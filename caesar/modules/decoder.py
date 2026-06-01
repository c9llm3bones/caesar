"""Decoder for protein structures in PyTorch.

Adapted from SALAD structure_autoencoder.
"""
# IN PROGRESS
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional, Tuple
from caesar.utils.geometry import Vec3Array, Rot3Array, Rigid3Array

from caesar.modules.encoder import (
    EncoderUpdate, 
    AADecoderPairFeatures
)
from caesar.modules.utils.geometry import (
    distance_one_hot, get_spatial_neighbours, index_align,
    sequence_relative_position,
    extract_neighbours_salad_compatible as extract_neighbours,
    get_neighbours, distance_rbf,
    index_mean,
    get_random_neighbours,
    index_sum,
    single_protein_sidechains,
    atom37_to_ncacocb, atom37_local_feature_channels, mask_atom37_local_positions,
)

from caesar.modules.basic import Linear, MLP, init_zeros, init_linear, gelu_salad

from caesar.modules.geometric import (
    SparseStructureAttention, SparseAttention, SparseSemiEquivariantPointAttention,
    direction_features,
    distance_features,
    extract_aa_frames,
    pair_vector_features,
    position_rotation_features,
)

from caesar.utils.loss import violation_loss
from caesar.utils.all_atom_multimer import get_atom14_mask

def _as_bool(value):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return False
        return bool(value.detach().bool().any().item())
    return bool(value)


def _assert_finite(name: str, tensor: torch.Tensor):
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return
    bad_count = int((~finite).sum().item())
    finite_vals = tensor[finite]
    if finite_vals.numel() > 0:
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
    else:
        min_val = float("nan")
        max_val = float("nan")
    raise RuntimeError(
        f"Non-finite tensor detected in {name}: bad_count={bad_count}, "
        f"finite_min={min_val}, finite_max={max_val}"
    )

def stop_gradient(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if hasattr(x, "map_tensor_fn"):
        return x.map_tensor_fn(lambda t: t.detach())
    raise TypeError(f"stop_gradient: unsupported type {type(x)}")

def _gather_rows(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather along dim 0 for arbitrary index shape.

    Uses modulo indexing to preserve legacy `-1` behavior from advanced indexing.
    """
    if index.dtype != torch.long:
        index = index.long()
    flat = index.reshape(-1).remainder(source.shape[0])
    gathered = torch.index_select(source, dim=0, index=flat)
    return gathered.reshape(*index.shape, *source.shape[1:])


def _gather_dim1(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Row-wise gather along dim 1 for tensors of shape (N, M, ...)."""
    if index.dtype != torch.long:
        index = index.long()
    safe = index.remainder(source.shape[1])
    view_shape = (*safe.shape, *([1] * (source.ndim - 2)))
    expand_shape = (*safe.shape, *source.shape[2:])
    gather_index = safe.view(view_shape).expand(expand_shape)
    return torch.gather(source, dim=1, index=gather_index)


def _gather_vec3(source: Vec3Array, index: torch.Tensor) -> Vec3Array:
    return Vec3Array(
        _gather_rows(source.x, index),
        _gather_rows(source.y, index),
        _gather_rows(source.z, index),
    )

class DecoderBlock(nn.Module):
    """Standard equivariant decoder block."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        if c.distogram_block == "mlp":
            self.distogram = QuickDistogram(c)
        elif c.distogram_block == "inner":
            self.distogram = InnerDistogram(c)
        elif c.distogram_block == None or c.distogram_block == "none":
            self.distogram = None
        else:
            raise ValueError(f"Unknown distogram_block={c.distogram_block}")

        self.ln_attn_main = nn.LayerNorm(c.local_size)
        self.ln_update = nn.LayerNorm(c.local_size)
        self.ln_pos = nn.LayerNorm(c.local_size)

        # (c) when distogram is enabled, salad does a
        # second attention pass with separate params
        if self.distogram is not None:
            self.ln_attn_dist = nn.LayerNorm(c.local_size)
        else:
            self.ln_attn_dist = None
            
        self.attn_main = SparseStructureAttention(c)
        self.attn_dist = SparseStructureAttention(c) if self.distogram is not None else None

        self.update = DecoderUpdate(c)
        self.pos_update = UpdatePositions()
        nr = int(getattr(c, "num_random_neighbours", 32))
        self.current_neigh = extract_neighbours(num_index=16, num_spatial=16, num_random=nr)
        self.dmap_neigh = extract_dmap_neighbours(count=32)

        self.pair_features_main = DecoderPairFeatures(c)
        self.pair_features_dist = DecoderPairFeatures(c) if self.distogram is not None else None

    def forward(
        self,
        features: torch.Tensor,                 # (N, local_size)
        pos: torch.Tensor,                      # (N, A, 3) 
        resi: torch.Tensor,                     # (N,)
        chain: torch.Tensor,                    # (N,)
        batch: torch.Tensor,                    # (N,)
        mask: torch.Tensor,                     # (N,)
        sup_neighbours: Optional[torch.Tensor] = None,  # (N, Ksup)
        pos_gt: Optional[torch.Tensor] = None,  
        generator: Optional[torch.Generator] = None,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.config
        N = features.shape[0]

        distogram_logits = None
        dmap = None
        dmap_neighbours = None

        if self.distogram is not None:
            if c.teacher_forcing_style:
                if sup_neighbours is None or pos_gt is None:
                    raise ValueError("teacher_forcing_style=True requires sup_neighbours and pos_gt")

                distogram_logits, _ = self.distogram(features, resi, chain, batch, sup_neighbours)

                cb = Vec3Array.from_array(pos_gt[:, -1])  # (N,3)
                dmap = (cb[:, None] - cb[None, :]).norm()  # (N,N)

            else:
                dist_full_logits, dmap = self.distogram(features, resi, chain, batch, None)  # logits (N,N,B), dmap (N,N)

                # salad uses axis_index; here it's just arange(N).
                if sup_neighbours is None:
                    raise ValueError("distogram_block enabled but sup_neighbours is None")

                sup_neighbours = sup_neighbours.long()
                distogram_logits = _gather_dim1(dist_full_logits, sup_neighbours)

            dmap_neighbours = self.dmap_neigh(dmap.detach(), resi, chain, batch, mask, generator=generator)
        else:
            distogram_logits = torch.zeros((1,), dtype=features.dtype, device=features.device)
            dmap = None

        current_neighbours = self.current_neigh(
            Vec3Array.from_array(pos), resi, chain, batch, mask,
            generator=generator
        )

        pair, pair_mask = self.pair_features_main(
            pos, dmap, current_neighbours, resi, chain, batch, mask
        )

        features = features + self.attn_main(
            self.ln_attn_main(features),
            pos / c.sigma_data,
            pair, pair_mask,
            current_neighbours, resi, chain, batch, mask
        )

        if self.distogram is not None:
            pair2, pair2_mask = self.pair_features_dist(
                pos, dmap, dmap_neighbours, resi, chain, batch, mask
            )
            features = features + self.attn_dist(
                self.ln_attn_dist(features),
                pos / c.sigma_data,
                pair2, pair2_mask,
                dmap_neighbours, resi, chain, batch, mask
            )

        features = features + self.update(
            self.ln_update(features),
            pos, chain, batch, mask
        )

        local_norm = self.ln_pos(features)
        symm = getattr(c, "symm", None)  
        pos = self.pos_update(pos, local_norm, scale=float(c.sigma_data), symm=symm)
        if pos.dtype != local_norm.dtype:
            pos = pos.to(dtype=local_norm.dtype)
        return features, pos, distogram_logits
    
class Decoder(nn.Module):
    """Protein structure decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        if getattr(c, 'equivariance', 'equivariant') == "nonequivariant":
            decoder_module = NonEquivariantDecoderStack
            decoder_block = NonEquivariantDecoderBlock
        elif getattr(c, 'equivariance', 'equivariant') == "semiequivariant":
            decoder_module = SemiEquivariantDecoderStack
            decoder_block = SemiEquivariantDecoderBlock
        else:
            decoder_module = DecoderStack
            decoder_block = DecoderBlock

        self.decoder_stack = decoder_module(c, decoder_block)
        self.aa_decoder = AADecoder(c)

        self.use_latent_diff = bool(
            getattr(c, 'input_diffusion', False) or getattr(c, 'latent_diffusion', False))
        if self.use_latent_diff:
            self.latent_ln = nn.LayerNorm(c.local_size)
            out_dim = c.local_size if getattr(c, 'noembed', False) else c.latent_size
            self.latent_decoder = MLP(
                size=c.local_size * 2,
                out_size=out_dim,
                bias=False,
                activation=gelu_salad,
                final_init=init_zeros(),
            )
        self.prev_local_ln = nn.LayerNorm(c.local_size, elementwise_affine=True)

        self.local_mlp = MLP(
            size=4 * c.local_size,
            out_size=c.local_size,
            bias=False,
            activation=gelu_salad,
            final_init=init_linear(),
        )

        self.local_ln = nn.LayerNorm(c.local_size, elementwise_affine=True)
        self.angle_pos = GetAnglePositions(local_dim=c.local_size)

        # atom37 main branch decoder
        self.decoder_stack37 = None
        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )
        if atom37_main:
            self.decoder_stack37 = DecoderStackAtom37(c, DecoderBlockAtom37)
            self.prev_local_ln_atom37 = nn.LayerNorm(c.local_size, elementwise_affine=True)
            self.local_mlp_atom37 = MLP(
                size=4 * c.local_size, out_size=c.local_size,
                activation=gelu_salad, bias=False, final_init=init_linear(),
            )
            self.local_ln_atom37 = nn.LayerNorm(c.local_size, elementwise_affine=True)
        
    def forward(self, data, prev, generator=None):
        if prev is None:
            prev = self.init_prev(data)
        c = self.config

        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )

        if atom37_main:
            local37, pos37, atom_mask37, resi, chain, batch, mask = self.prepare_features(data, prev)
            sup_neighbours = get_random_neighbours(c.fape_neighbours)(
                Vec3Array.from_array(data["pos_gt"][:, -1]), batch, mask, generator=generator)

            local37, pos37, trajectory37, sup_distogram = self.decoder_stack37(
                local37, pos37, atom_mask37, None,
                resi, chain, batch, mask,
                sup_neighbours=sup_neighbours, generator=generator)

            # derive backbone view from atom37
            pos = atom37_to_ncacocb(pos37)
            trajectory = torch.stack(
                [atom37_to_ncacocb(trajectory37[t]) for t in range(trajectory37.shape[0])], dim=0)
            local = local37

            result = {
                "latent": data["latent"],
                "trajectory": trajectory,
                "trajectory37": trajectory37,
                "sup_neighbours": sup_neighbours,
                "sup_distogram": sup_distogram,
                "local": local,
                "pos": pos,
                "pos37": pos37,
            }

            aa_logits, decoder_features, corrupt_aa = self.aa_decoder.decode_train(
                data["aa_gt"], local, pos, resi, chain, batch, mask)
            result.update({
                "aa": aa_logits,
                "aa_features": decoder_features,
                "corrupt_aa": corrupt_aa * data["mask"] * (data["aa_gt"] != 20),
                "aa_gt": data["aa_gt"],
            })

            aatype = data["aa_gt"]
            if getattr(c, "eval", False):
                aatype = aa_logits.argmax(dim=-1)
            raw_angles, angles, atom_pos = self.angle_pos(aatype, local, pos)
            result.update({"raw_angles": raw_angles, "angles": angles, "atom_pos": atom_pos})
            return result

        local, pos, resi, chain, batch, mask = self.prepare_features(data, prev)

        # supervision neighbours (FAPE)
        sup_neighbours = get_random_neighbours(c.fape_neighbours)(
            Vec3Array.from_array(data["pos_gt"][:, -1]), batch, mask, generator=generator
        )

        local, pos, trajectory, sup_distogram = self.decoder_stack(
            local, pos, resi, chain, batch, mask, sup_neighbours,
            generator=generator
        )

        result = {
            "latent": data["latent"],
            "trajectory": trajectory,
            "sup_neighbours": sup_neighbours,
            "sup_distogram": sup_distogram,
            "local": local,
            "pos": pos,
        }

        if self.use_latent_diff:
            latent_update = self.latent_decoder(
                torch.cat((self.latent_ln(local), data["latent"]), dim=-1)
            )

            time = data["time"]
            if getattr(c, 'vp_diffusion', False):
                predicted_latent = data["latent"] + latent_update
            else:
                predicted_latent = (
                    data["skip_latent"]
                    + latent_update * time[:, None] / torch.sqrt(1 + time[:, None] ** 2)
                )
            result["predicted_latent"] = predicted_latent

        aa_logits, decoder_features, corrupt_aa = self.aa_decoder.decode_train(
            data["aa_gt"], local, pos, resi, chain, batch, mask
        )

        result.update({
            "aa": aa_logits,
            "aa_features": decoder_features,
            "corrupt_aa": corrupt_aa * data["mask"] * (data["aa_gt"] != 20),
            "aa_gt": data["aa_gt"],
        })

        aatype = data["aa_gt"]
        if getattr(c, "eval", False):
            aatype = aa_logits.argmax(dim=-1)

        raw_angles, angles, atom_pos = self.angle_pos(aatype, local, pos)

        result.update({
            "raw_angles": raw_angles,
            "angles": angles,
            "atom_pos": atom_pos,
        })
        return result

    def init_prev(self, data):
        c = self.config
        local_dtype = data["latent"].dtype if "latent" in data else data["pos"].dtype
        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )
        if atom37_main:
            pos37 = data["pos37"]
            return {
                "pos37": pos37,
                "local": torch.zeros((pos37.shape[0], c.local_size),
                                      device=pos37.device, dtype=local_dtype),
            }
        return {
            "pos": data["pos"],
            "local": torch.zeros(
                (data["pos"].shape[0], c.local_size),
                device=data["pos"].device,
                dtype=local_dtype,
            ),
        }

    def prepare_features(self, data, prev):
        c = self.config

        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )

        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        latent = data["latent"]
        mask = data["mask"]

        if atom37_main:
            pos37 = prev["pos37"]
            atom_mask37 = data["all_atom_mask_37"].to(dtype=pos37.dtype)
            _, local_pos37 = extract_aa_frames(Vec3Array.from_array(pos37))
            local_pos37_arr = local_pos37.to_tensor()
            local_pos37_arr = mask_atom37_local_positions(local_pos37_arr, atom_mask37)

            local_features37 = [
                local_pos37_arr.reshape(local_pos37_arr.shape[0], -1),
                distance_rbf(
                    torch.sqrt(torch.clamp((local_pos37_arr ** 2).sum(dim=-1), min=1e-6)),
                    0.0, 22.0).reshape(local_pos37_arr.shape[0], -1),
                latent,
                self.prev_local_ln_atom37(prev["local"]),
            ]
            local_features37 = torch.cat(local_features37, dim=-1)
            local37 = self.local_mlp_atom37(local_features37)
            local37 = self.local_ln_atom37(local37)
            return local37, pos37, atom_mask37, resi, chain, batch, mask

        pos = prev["pos"]

        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))

        local_features = [
            local_pos.to_tensor().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1),
            latent
        ]

        if getattr(c, 'time_embedding', False) and getattr(c, 'latent_diffusion', False) and "time" in data:
            time = distance_rbf(data["time"], 0, 80.0, bins=200)
            local_features.append(time)

        local_features.append(self.prev_local_ln(prev["local"]))

        local_features = torch.cat(local_features, dim=-1)

        local = self.local_mlp(local_features)
        local = self.local_ln(local)

        return local, pos, resi, chain, batch, mask
    
    def loss(self, data, result, generator=None):
        """Compute Decoder losses from the input data and Decoder results."""
        c = self.config
        detect_nonfinite = _as_bool(data.get("detect_nonfinite", False))

        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]  
        mask_bool = mask.bool()
        # mask_f = mask.to(dtype=torch.float32)

        losses = {}
        total = None

        # AA NLL loss
        aa_gt = data["aa_gt"]
        aa_predict_mask = mask * (aa_gt != 20)
       
        aa_logp = result["aa"] 
        
        aa_gt_long = aa_gt.long()
        # (c) jax-equivalent one-hot semantics
        valid = (aa_gt_long >= 0) & (aa_gt_long < 20)
        aa_one_hot = F.one_hot(aa_gt_long.clamp(0, 19), num_classes=20).to(dtype=aa_logp.dtype)
        aa_one_hot = aa_one_hot * valid.unsqueeze(-1)
     
        aa_nll = -(aa_logp * aa_one_hot).sum(dim=-1)
        aa_nll = torch.where(aa_predict_mask.bool(), aa_nll, torch.zeros_like(aa_nll))
        aa_mask_sum = aa_predict_mask.to(dtype=aa_nll.dtype).sum()
        aa_nll = aa_nll.sum() / torch.clamp(aa_mask_sum, min=1.0)
        losses["aa"] = aa_nll
        losses["debug_mask_sum"] = mask.sum()
        losses["debug_aa_mask_sum"] = aa_predict_mask.sum()
        losses["weighted_aa"] = float(c.aa_weight) * aa_nll
        total = losses["weighted_aa"]

        # position losses
        mask_f = mask.to(dtype=aa_nll.dtype)
        base_weight = mask_f / torch.clamp(index_sum(mask_f, batch, mask_bool), min=1.0) / (batch.max() + 1)
        losses["debug_base_weight_sum"] = base_weight.sum()
        
        # sparse neighbour FAPE ** 2
        pair_mask = (batch[:, None] == batch[None, :])
        pair_mask = pair_mask * (mask[:, None] * mask[None, :])

        pos_gt = data["pos_gt"]
        pos_gt = torch.where(mask[:, None, None].bool(), pos_gt, torch.zeros_like(pos_gt))
        pos_gt = Vec3Array.from_array(pos_gt)

        frames_gt, _ = extract_aa_frames(stop_gradient(pos_gt))

        # CB distance 
        distance = data["dmap"]
        distance = torch.where(pair_mask.bool(), distance, torch.full_like(distance, float("inf")))
        # get random neighbours to compute sparse FAPE on
        neighbours = get_random_neighbours(c.fape_neighbours)(distance, batch, mask, generator=generator)
        losses["debug_neighbours_hash"] = neighbours.sum().to(dtype=aa_nll.dtype)
        gathered_mask = _gather_rows(mask, neighbours)
        mask_neighbours = (neighbours != -1) * mask[:, None] * gathered_mask
        pos_gt_local = frames_gt[:, None, None].apply_inverse_to_point(_gather_vec3(pos_gt, neighbours))
        
        traj = result["trajectory"]  # tensor, (T, N, A, 3)
        traj_vec = Vec3Array.from_array(traj)
        frames, _ = extract_aa_frames(traj_vec)  # (T, N) Rigid3Array
        safe_neighbours = neighbours.long().remainder(traj.shape[1])
        flat_neighbours = safe_neighbours.reshape(-1)
        traj_neighbours = torch.index_select(traj, dim=1, index=flat_neighbours).reshape(
            traj.shape[0], neighbours.shape[0], neighbours.shape[1], traj.shape[2], traj.shape[3]
        )
        traj_local = frames[:, :, None, None].apply_inverse_to_point(Vec3Array.from_array(traj_neighbours))
        fape_base = (traj_local - pos_gt_local).norm2()

        fape_clipped = torch.clamp(fape_base, 0.0, float(c.clip_fape))

        if getattr(c, "unclipped_weight", 0):
            fape_clipped = fape_base * float(c.unclipped_weight) + fape_clipped

        if getattr(c, "no_fape2", False):
            # use FAPE instead of FAPE ** 2
            fape_clipped = torch.sqrt(torch.clamp(fape_clipped, min=1e-6))

        fape_traj = fape_clipped.mean(dim=-1)  # [T,N,K] 
        fape_traj = torch.where(
            mask_neighbours[None].bool(),
            fape_traj,
            torch.zeros_like(fape_traj),
        )

        fape_traj = fape_traj.sum(dim=-1) / torch.clamp(mask_neighbours.sum(dim=1)[None], min=1.0)
        fape_traj = (fape_traj * base_weight).sum(dim=-1)
        losses["fape"] = fape_traj[-1] / 3.0
        losses["fape_trajectory"] = fape_traj.mean() / 3.0
        fape_loss = (float(c.fape_weight) * fape_traj[-1] + float(c.fape_trajectory_weight) * fape_traj.mean()) / 3.0
        losses["weighted_fape"] = fape_loss
        total = total + losses["weighted_fape"]

        # atom37 full-atom losses
        if getattr(c, 'atom37_parallel_mode', 'none') == 'parallel' and 'pos37' in result:
            pos37_pred = result['pos37']
            pos37_gt = data.get('pos37_gt', None)
            if pos37_gt is not None:
                atom37_mask = data.get('all_atom_mask_37', data['all_atom_mask'])
                if getattr(c, 'loss_all_atoms', False):
                    atom37_mask = torch.ones_like(atom37_mask) * mask[:, None]

                pos37_pred_v = Vec3Array.from_array(pos37_pred)
                pos37_gt_v = Vec3Array.from_array(pos37_gt)

                pos37_local_mode = getattr(c, 'pos37_local_mode', 'all_atom')
                local_neighbours37 = get_spatial_neighbours(count=c.local_neighbours)(
                    pos37_gt_v[:, 1], batch, mask)
                mask37_gt = _gather_rows(atom37_mask, local_neighbours37)

                if pos37_local_mode == 'sidechain':
                    # sidechain mask: exclude N(0), CA(1), C(2), O(4) backbone atoms
                    sc_mask37 = atom37_mask.clone()
                    sc_mask37[:, 0] = 0  # N
                    sc_mask37[:, 1] = 0  # CA
                    sc_mask37[:, 2] = 0  # C
                    sc_mask37[:, 4] = 0  # O
                    mask37_gt = mask37_gt * _gather_rows(sc_mask37, local_neighbours37)

                frames37_gt, _ = extract_aa_frames(pos37_gt_v)
                pos37_gt_local = frames37_gt[:, None, None].apply_inverse_to_point(
                    _gather_vec3(pos37_gt_v, local_neighbours37))
                frames37, _ = extract_aa_frames(pos37_pred_v)
                pos37_local = frames37[:, None, None].apply_inverse_to_point(
                    _gather_vec3(pos37_pred_v, local_neighbours37))

                pos37_loss = torch.where(
                    mask37_gt.bool(),
                    (pos37_local - pos37_gt_local).norm2(),
                    torch.zeros_like((pos37_local - pos37_gt_local).norm2())
                ).sum(dim=(1, 2))
                pos37_loss = pos37_loss / torch.clamp(
                    mask37_gt.sum(dim=(1, 2)).to(dtype=pos37_loss.dtype), min=1)
                pos37_loss = (torch.where(
                    mask.bool(), pos37_loss, torch.zeros_like(pos37_loss)
                ) * base_weight).sum() / 3.0
                losses['pos37'] = pos37_loss
                losses['pos37_local'] = pos37_loss
                weighted_pos37 = float(getattr(c, 'pos37_local_weight', 0.0)) * pos37_loss
                losses['weighted_pos37_local'] = weighted_pos37
                total = total + weighted_pos37

                # Aligned sidechain loss
                sc_aligned_w = float(getattr(c, 'atom37_sidechain_aligned_weight', 0.0))
                if sc_aligned_w > 0:
                    sc_mask37 = atom37_mask.clone()
                    sc_mask37[:, 0] = 0
                    sc_mask37[:, 1] = 0
                    sc_mask37[:, 2] = 0
                    sc_mask37[:, 4] = 0
                    pos37_gt_bb_aligned = index_align(
                        pos37_gt, pos37_pred, batch, mask)
                    sc_sqerr = ((pos37_pred - pos37_gt_bb_aligned) ** 2).sum(dim=-1)
                    sc_sqerr = torch.where(
                        sc_mask37.bool(), sc_sqerr, torch.zeros_like(sc_sqerr)).sum(dim=-1)
                    sc_count = torch.clamp(
                        sc_mask37.sum(dim=-1).to(dtype=sc_sqerr.dtype), min=1)
                    sc_sqerr = sc_sqerr / sc_count
                    sc_sqerr = (torch.where(
                        mask.bool(), sc_sqerr, torch.zeros_like(sc_sqerr)
                    ) * base_weight).sum() / 3.0
                    losses['atom37_sidechain_aligned_loss'] = sc_sqerr
                    weighted_sc = sc_aligned_w * sc_sqerr
                    losses['weighted_atom37_sidechain_aligned_loss'] = weighted_sc
                    total = total + weighted_sc

                # RMSD diagnostic
                losses['rmsd_no_align_atom37'] = _masked_rmsd_no_align(
                    pos37_pred, pos37_gt, atom37_mask, mask)

        # sup distogram loss
        if getattr(c, "distogram_block", None) not in (None, "none"):
            cb_gt = pos_gt[:, -1]  
            sup_neighbours = result["sup_neighbours"]
            gathered_mask = _gather_rows(mask, sup_neighbours)
            sup_mask = (sup_neighbours != -1) * mask[:, None] * gathered_mask
            dist_gt = (cb_gt[:, None] - _gather_vec3(cb_gt, sup_neighbours)).norm()
            dist_one_hot = distance_one_hot(dist_gt, 0.0, 22.0, 16)
            distogram_nll = -(result["sup_distogram"] * dist_one_hot[None]).sum(dim=-1)
            distogram_nll = torch.where(sup_mask.bool(), distogram_nll, torch.zeros_like(distogram_nll)).sum(dim=-1)
            distogram_nll = distogram_nll / torch.clamp(sup_mask.sum(dim=1), min=1.0)
            distogram_nll = (distogram_nll * base_weight).sum(dim=1)
            losses["debug_sup_mask_sum"] = sup_mask.sum()
            losses["distogram"] = distogram_nll[-1]
            losses["distogram_trajectory"] = distogram_nll.mean()
            losses["weighted_distogram"] = 10.0 * distogram_nll[-1] + 5.0 * distogram_nll.mean()
            total = total + losses["weighted_distogram"]

        # Kabsch RMSD loss
        if getattr(c, "kabsch_rmsd", False):
            traj = Vec3Array.from_array(result["trajectory"])
            last = traj[-1]
            pos_gt = stop_gradient(index_align(pos_gt, last, batch, mask))
            pos_loss_unclipped = (pos_gt[None] - traj).norm2()
            pos_loss_clipped = torch.clamp(pos_loss_unclipped, 0.0, 100.0)
            pos_loss = pos_loss_clipped
            if c.unclipped_weight:
                pos_loss = pos_loss_unclipped * float(c.unclipped_weight) + pos_loss
            pos_loss = pos_loss.mean(dim=-1)
            pos_loss = pos_loss * base_weight[None]
            pos_loss = pos_loss.sum(dim=-1)
            losses["kabsch_rmsd"] = pos_loss[-1] / 3.0
            losses["kabsch_rmsd_trajectory"] = pos_loss.mean() / 3.0
            pos_loss = (float(c.fape_weight) * pos_loss[-1] + float(c.fape_trajectory_weight) * pos_loss.mean()) / 3.0
            losses["weighted_kabsch_rmsd"] = pos_loss
            total = total + losses["weighted_kabsch_rmsd"]

        # local loss
        atom_pos = Vec3Array.from_array(result["atom_pos"])
        atom_pos_gt = Vec3Array.from_array(data["atom_pos"])

        local_neighbours = get_spatial_neighbours(count=c.local_neighbours)(
            pos_gt[:, -1], batch, mask
        )

        mask_gt = _gather_rows(data["atom_mask"], local_neighbours)

        frames_gt, _ = extract_aa_frames(atom_pos_gt)
        atom_pos_gt = frames_gt[:, None, None].apply_inverse_to_point(_gather_vec3(atom_pos_gt, local_neighbours))

        frames, _ = extract_aa_frames(atom_pos)
        atom_pos = frames[:, None, None].apply_inverse_to_point(_gather_vec3(atom_pos, local_neighbours))

        local_loss = torch.where(
            mask_gt.bool(),
            (atom_pos - atom_pos_gt).norm2(),
            torch.zeros_like((atom_pos - atom_pos_gt).norm2()),
        ).sum(dim=(1, 2))

        local_loss = local_loss / torch.clamp(mask_gt.sum(dim=(1, 2)), min=1)

        local_loss = (torch.where(mask.bool(), local_loss, torch.zeros_like(local_loss)) * base_weight).sum() / 3.0

        losses["local"] = local_loss
        losses["weighted_local"] = float(c.local_weight) * local_loss
        total = total + losses["weighted_local"]

        # VQ losses
        if getattr(c, "codebook_size", 0) and (not getattr(c, "is_decoder", False)):
            cl = result["codebook_losses"]
            losses.update(cl)
            if not getattr(c, "state", False):
                total = total + float(c.codebook_loss_scale) * (cl["codebook"] + cl["unassigned"])
            total = total + float(c.codebook_loss_scale) * float(c.codebook_b) * cl["commitment"]
                
        # additional denoising losses (not used in manuscript)
        if (getattr(c, "input_diffusion", False) or getattr(c, "latent_diffusion", False)) and getattr(c, "latent_loss_scale", 0):
            raw_loss = ((data["clean_latent"] - result["predicted_latent"]) ** 2).mean(dim=-1)  # [N]
            if getattr(c, "vp_diffusion", False):
                weighted_loss = (torch.where(mask_bool, raw_loss, torch.zeros_like(raw_loss)) * base_weight).sum()
            else:
                time = torch.clamp(data["time"], min=1e-2)
                raw_loss = raw_loss * (1.0 + time ** 2) / torch.clamp(time ** 2, min=1e-6)
                weighted_loss = (torch.where(mask_bool, raw_loss, torch.zeros_like(raw_loss)) * base_weight).sum()

            losses["latent"] = weighted_loss
            losses["weighted_latent"] = float(c.latent_loss_scale) * weighted_loss
            total = total + losses["weighted_latent"]

        # AlphaFold violation loss. (not used in manuscript)
        if getattr(c, "violation_scale", 0):
            res_mask = data["mask"].bool()
            pred_mask_dtype = result["atom_pos"].dtype
            pred_mask = get_atom14_mask(data["aa_gt"]).to(device=res_mask.device, dtype=pred_mask_dtype)
            pred_mask = pred_mask * res_mask[:, None].to(dtype=pred_mask_dtype)

            violation, _ = violation_loss(
                data["aa_gt"],
                data["residue_index"],
                result["atom_pos"],
                pred_mask,
                res_mask,
                clash_overlap_tolerance=1.5,
                violation_tolerance_factor=2.0,
                chain_index=data["chain_index"],
                batch_index=data["batch_index"],
                per_residue=False,
            )
            losses["violation"] = violation.mean()
            losses["weighted_violation"] = float(c.violation_scale) * violation.mean()
            total = total + losses["weighted_violation"]

        if detect_nonfinite:
            _assert_finite("decoder/aa", result["aa"])
            _assert_finite("decoder/trajectory", result["trajectory"])
            _assert_finite("decoder/atom_pos", result["atom_pos"])
            _assert_finite("decoder/total", total)

        losses["total"] = total
        return total, losses

class DecoderUpdate(nn.Module):
    """GeGLU update with global averaging for the decoder."""

    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__()
        self.config = config

        c = self.config
        local_size = int(c.local_size)

        self.pos_mlp = MLP(
            local_size * 2,
            local_size,
            activation=gelu_salad,
            final_init="zeros",
        )

        self.local_update = Linear(local_size * int(c.factor), initializer="linear", bias=False)

        self.local_gate = Linear(local_size * int(c.factor), initializer="relu", bias=False)
        self.chain_gate = Linear(local_size * int(c.factor), initializer="relu", bias=False)
        self.batch_gate = Linear(local_size * int(c.factor), initializer="relu", bias=False)

        self.out = Linear(local_size, initializer="zeros")

    def forward(self, local, pos, chain, batch, mask):
        c = self.config

        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))
        local_pos = local_pos.to_tensor() if hasattr(local_pos, "to_tensor") else local_pos.to_tensor()
        local_pos = local_pos.reshape(local_pos.shape[0], -1)

        local = local + self.pos_mlp(local_pos)

        local_update = self.local_update(local)
        local_gate = gelu_salad(self.local_gate(local))
        chain_gate = gelu_salad(self.chain_gate(local))
        batch_gate = gelu_salad(self.batch_gate(local))

        hidden = index_mean(batch_gate * local_update, batch, mask[..., None])
        hidden = hidden + index_mean(chain_gate * local_update, chain, mask[..., None])
        hidden = hidden + local_gate * local_update

        result = self.out(hidden)
        return result

class GetAnglePositions(nn.Module):
    def __init__(self, local_dim):
        super().__init__()

        self.mlp = MLP(
            local_dim * 2,
            7 * 2,
            bias=False,
            activation=gelu_salad,
            final_init="linear"
        )

    def forward(self, aa_gt, local, pos):
        frames, local_positions = extract_aa_frames(Vec3Array.from_array(pos))

        features = [
            local,
            local_positions.to_tensor().reshape(local_positions.shape[0], -1),
            distance_rbf(
                local_positions.norm(), 0.0, 10.0, 16
            ).reshape(local_positions.shape[0], -1),
            F.one_hot(aa_gt.long(), num_classes=21).to(local.dtype),
        ]

        raw_angles = self.mlp(torch.cat(features, dim=-1))

        raw_angles = raw_angles.reshape(-1, 7, 2)
        angles = raw_angles / torch.sqrt(
            torch.clamp((raw_angles ** 2).sum(dim=-1, keepdim=True), min=1e-6)
        )

        angle_pos, _ = single_protein_sidechains(aa_gt, frames, angles)
        angle_pos = angle_pos.to_tensor().reshape(-1, 14, 3)
        angle_pos = torch.cat(
            (pos[..., :4, :], angle_pos[..., 4:, :]), dim=-2
        )

        return raw_angles, angles, angle_pos

class AADecoderBlock(nn.Module):
    """Amino acid sequence decoder block."""

    def __init__(self, config, name: Optional[str] = "aa_decoder_block"):
        super().__init__()
        self.config = config
        c = self.config

        self.pair_features = AADecoderPairFeatures(c)

        self.aa_linear = Linear(int(c.pair_size), bias=False)
        self.ln_aa = nn.LayerNorm(int(c.pair_size), elementwise_affine=True)

        self.pair_mlp = MLP(
            2 * int(c.pair_size),
            int(c.pair_size),
            activation=gelu_salad,
            final_init="linear",
        )

        self.attn = SparseStructureAttention(c)
        self.ln_attn_in = nn.LayerNorm(int(c.local_size), elementwise_affine=True)

        self.update = EncoderUpdate(c)
        self.ln_update_in = nn.LayerNorm(int(c.local_size), elementwise_affine=True)

    def forward(self, aa, features, pos, neighbours, resi, chain, batch, mask):
        c = self.config

        pair, pair_mask = self.pair_features(
            Vec3Array.from_array(pos), neighbours, resi, chain, batch, mask
        )

        aa_oh = F.one_hot(aa.long(), 21).to(dtype=features.dtype)
        aa_emb = self.ln_aa(self.aa_linear(aa_oh))         
        pair = pair + _gather_rows(aa_emb, neighbours)

        pair = self.pair_mlp(pair)

        features = features + self.attn(
            self.ln_attn_in(features),
            pos,
            pair,
            pair_mask,
            neighbours,
            resi,
            chain,
            batch,
            mask,
        )

        features = features + self.update(
            self.ln_update_in(features),
            pos,
            chain,
            batch,
            mask,
        )

        return features
    
class AADecoderStack(nn.Module):
    """Amino acid sequence decoder stack."""

    def __init__(self, config, depth: Optional[int] = None, name: Optional[str] = "aa_decoder_stack"):
        super().__init__()
        self.config = config
        self.depth = int(depth or 3)

        self.blocks = nn.ModuleList([AADecoderBlock(config) for _ in range(self.depth)])
        self.ln = nn.LayerNorm(int(config.local_size), elementwise_affine=True)

    def forward(self, aa, local, pos, neighbours, resi, chain, batch, mask):
        x = local
        for block in self.blocks:
            x = block(aa, x, pos, neighbours, resi, chain, batch, mask)
        x = self.ln(x)
        return x
    
class AADecoder(nn.Module):
    """Amino acid sequence decoder module."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.local_size)
        self.proj = nn.Linear(config.local_size, 20, bias=False)
        nn.init.zeros_(self.proj.weight)

        self.stack = AADecoderStack(config, depth=config.aa_decoder_depth)

    def forward(self, aa, local, pos, resi, chain, batch, mask):
        neighbours = get_spatial_neighbours(32)(
            Vec3Array.from_array(pos)[:, -1], batch, mask
        )

        local = self.stack(
            aa, local, pos, neighbours,
            resi, chain, batch, mask
        )

        local = self.norm(local)
        logits = self.proj(local)
        return logits, local

    def decode_train(self, aa, local, pos, resi, chain, batch, mask):
        aa = torch.full_like(aa, 20)
        logits, features = self.forward(
            aa, local, pos, resi, chain, batch, mask
        )
        logits = F.log_softmax(logits, dim=-1)
        return logits, features, torch.ones_like(mask)

    
class DecoderStack(nn.Module):
    def __init__(self, config, block_cls):
        super().__init__()
        self.blocks = nn.ModuleList([
            block_cls(config) for _ in range(config.depth)
        ])

    def forward(self, local, pos, resi, chain, batch, mask, sup_neighbours, generator=None):
        trajectory = []
        sup_distograms = []

        for block in self.blocks:
            local, pos, sup_dist = block(
                local, pos, resi, chain, batch, mask,
                sup_neighbours=sup_neighbours,
                generator=generator,
            )
            trajectory.append(pos)
            sup_distograms.append(sup_dist)

        trajectory = torch.stack(trajectory, dim=0)
        sup_distograms = torch.stack(sup_distograms, dim=0)

        return local, pos, trajectory, sup_distograms
    

class SemiEquivariantDecoderBlock(nn.Module):
    """Partly non-equivariant decoder block (+dgram in the manuscript)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        if c.distogram_block == "mlp":
            self.dist = QuickDistogram(c)    
            self.has_dist = True
        elif c.distogram_block == "inner":
            self.dist = InnerDistogram(c)         
            self.has_dist = True
        elif c.distogram_block == "none":
            self.dist = None
            self.has_dist = False
        else:
            raise ValueError(f"Unknown distogram_block: {c.distogram_block}")

        self.dmap_neigh = extract_dmap_neighbours(count=32)
        nr = int(getattr(c, "num_random_neighbours", 32))
        self.current_neigh = extract_neighbours(num_index=16, num_spatial=16, num_random=nr)

        self.pair_feat_main = HybridDecoderPairFeatures(c, center_features=False)
        self.pair_feat_dist = HybridDecoderPairFeatures(c, center_features=False) if self.has_dist else None

        self.attn_main = SparseSemiEquivariantPointAttention(config=c, size=c.key_size, heads=c.heads)
        self.attn_dist = SparseSemiEquivariantPointAttention(config=c, size=c.key_size, heads=c.heads) if self.has_dist else None

        self.update = NonEquivariantDecoderUpdate(c)

        self.ln_attn_main = nn.LayerNorm(c.local_size)
        self.ln_attn_dist = nn.LayerNorm(c.local_size) if self.has_dist else None
        self.ln_update_in = nn.LayerNorm(c.local_size)
        self.ln_pos = nn.LayerNorm(c.local_size)

        self.strict_minus1_indexing = True


    def forward(
        self,
        features: torch.Tensor,
        pos: torch.Tensor,
        resi: torch.Tensor,
        chain: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        sup_neighbours: Optional[torch.Tensor] = None,
        pos_gt: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.config
        N = features.shape[0]

        if not self.has_dist:
            raise ValueError('SemiEquivariantDecoderBlock: salad doesn\'t suppose that has_dist=None')
            ### mirror "none" case; original code would actually crash earlier for this class
            # distogram_logits = torch.zeros((1,), dtype=torch.float32, device=features.device)
            # dmap = None
            # dmap_neighbours = None
        else:
            # original: distogram_logits, dmap = dist(features, resi, chain, batch, None)
            dist_full_logits, dmap = self.dist(features, resi, chain, batch, None)  # (N,N,B), (N,N)

            if sup_neighbours is None:
                raise ValueError("SemiEquivariantDecoderBlock requires sup_neighbours when distogram is enabled.")

            sup_neighbours = sup_neighbours.long()
            distogram_logits = _gather_dim1(dist_full_logits, sup_neighbours)
            
            dmap_neighbours = self.dmap_neigh(dmap.detach(), resi, chain, batch, mask, generator=generator)

        current_neighbours = self.current_neigh(
            Vec3Array.from_array(pos),
            resi, chain, batch, mask,
            generator=generator
        )
        
        pair, pair_mask = self.pair_feat_main(
            pos, dmap, current_neighbours,
            resi, chain, batch, mask
        )
        features = features + self.attn_main(
            self.ln_attn_main(features),
            pair,
            pos / c.sigma_data,
            current_neighbours,
            pair_mask
        )

        if self.has_dist:
            pair2, pair2_mask = self.pair_feat_dist(
                pos, dmap, dmap_neighbours,
                resi, chain, batch, mask
            )
            features = features + self.attn_dist(
                self.ln_attn_dist(features),
                pair2,
                pos / c.sigma_data,
                dmap_neighbours,
                pair2_mask
            )

        features = features + self.update(
            self.ln_update_in(features),
            chain, batch, mask
        )

        local_norm = self.ln_pos(features)

        pos = semiequivariant_update_positions(
            pos, local_norm,
            scale=c.sigma_data,
            symm=c.symm
        )

        if pos.dtype != local_norm.dtype:
            pos = pos.to(dtype=local_norm.dtype)
        return features, pos, distogram_logits

class SemiEquivariantDecoderStack(nn.Module):
    """Stack of partly non-equivariant Decoder blocks using data augmentation."""

    def __init__(self, config, block_cls):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            block_cls(config) for _ in range(config.depth)
        ])

    def forward(self, local, pos,
                resi, chain, batch, mask,
                sup_neighbours, generator=None):
        c = self.config

        # apply data augmentation when training non-equivariant Decoders
        pos = structure_augmentation(
            pos / c.sigma_data, batch, mask, generator=generator
        ) * c.sigma_data

        trajectory = []
        sup_distograms = []

        for block in self.blocks:
            local, pos, sup_dist = block(
                local, pos,
                resi, chain, batch, mask,
                sup_neighbours=sup_neighbours,
                generator=generator,
            )
            trajectory.append(pos)
            sup_distograms.append(sup_dist)

        trajectory = torch.stack(trajectory, dim=0)
        sup_distograms = torch.stack(sup_distograms, dim=0)

        return local, pos, trajectory, sup_distograms

class NonEquivariantDecoderStack(nn.Module):
    """Stack of fully non-equivariant decoder blocks."""

    def __init__(self, config, block_cls):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList([
            block_cls(config) for _ in range(config.depth)
        ])

        self.input_proj = nn.Linear(
            config.local_size + 5 * 3,
            config.local_size,
            bias=False
        )

        self.pos_proj = nn.Linear(config.local_size, 5 * 3, bias=False)
        self.norm = nn.LayerNorm(config.local_size)

    def forward(self, local, pos,
                resi, chain, batch, mask,
                sup_neighbours, generator=None):
        c = self.config

        aug_pos = structure_augmentation(
            pos / c.sigma_data, batch, mask, generator=generator
        )

        localpos = torch.cat(
            [local, aug_pos.reshape(pos.shape[0], -1)],
            dim=-1
        )
        localpos = self.input_proj(localpos)

        trajectory = []
        sup_distograms = []

        for block in self.blocks:
            localpos, sup_dist = block(
                localpos,
                resi, chain, batch, mask,
                sup_neighbours
            )
            trajectory.append(localpos)
            sup_distograms.append(sup_dist)

        trajectory = torch.stack(trajectory, dim=0)
        sup_distograms = torch.stack(sup_distograms, dim=0)

        traj = self.norm(trajectory)
        traj = self.pos_proj(traj)
        traj = traj.view(
            traj.shape[0], traj.shape[1], 5, 3
        )
        traj = traj * c.sigma_data

        pos = traj[-1]
        return local, pos, traj, sup_distograms


class HybridDecoderPairFeatures(nn.Module):
    """Pair features for the partially nonequivariant Decoder (NEQ)."""

    def __init__(self, c, center_features: bool = False):
        super().__init__()
        self.c = c
        self.center_features = bool(center_features) 

        self.relpos = sequence_relative_position(32, one_hot=True)

        self.p_relpos = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dmap   = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dist   = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_neigh  = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dirs   = Linear(c.pair_size, bias=False, initializer="linear")

        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)
        self.mlp = MLP(c.pair_size * 2, c.pair_size, activation=gelu_salad, final_init="linear")

    def forward(
        self,
        pos: torch.Tensor,
        dmap: Optional[torch.Tensor],
        neighbours: torch.Tensor,
        resi: torch.Tensor,
        chain: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.c
        neighbours = neighbours.long()

        gathered_mask = _gather_rows(mask, neighbours)
        pair_mask = mask[:, None] * gathered_mask
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)

        if dmap is not None:
            dmap = _gather_dim1(dmap, neighbours)

        pair = self.p_relpos(self.relpos(resi, chain, batch, neighbours))

        if dmap is not None:
            dmap_feat = distance_rbf(dmap)
            pair = pair + self.p_dmap(torch.where(pair_mask[..., None].bool(), dmap_feat, 0.0))

        pos_v = Vec3Array.from_array(pos)

        pair = pair + self.p_dist(distance_features(pos_v, neighbours, d_min=0.0, d_max=22.0))

        pos_arr = pos_v.to_tensor() if hasattr(pos_v, "to_tensor") else pos_v.to_tensor()  # (N, A, 3)
        K = neighbours.shape[1]

        local_neighbourhood = torch.cat(
            (
                pos_arr[:, None].expand(-1, K, -1, -1),
                _gather_rows(pos_arr, neighbours),
            ),
            dim=-2,
        ) / float(c.sigma_data)

        center_neighbourhood = local_neighbourhood - pos_arr[:, None, 1, None]
        center_neighbourhood = (Vec3Array.from_array(center_neighbourhood).normalized())
        center_neighbourhood = center_neighbourhood.to_tensor() if hasattr(center_neighbourhood, "to_tensor") else center_neighbourhood.to_tensor()

        local_neighbourhood = local_neighbourhood.reshape(*neighbours.shape, -1)
        center_neighbourhood = center_neighbourhood.reshape(*neighbours.shape, -1)

        pair = pair + self.p_neigh(torch.cat((local_neighbourhood, center_neighbourhood), dim=-1))

        dirs = (pos_v[:, None, :, None] - _gather_vec3(pos_v, neighbours)[:, :, None, :]).to_tensor()
        dirs = dirs.reshape(*neighbours.shape, -1)
        pair = pair + self.p_dirs(dirs)

        pair = self.ln(pair)
        pair = self.mlp(pair)
        return pair, pair_mask

class NonequivariantDecoderPairFeatures(nn.Module):
    """Pair features for a fully non-equivariant decoder."""

    def __init__(self, c):
        super().__init__()
        self.c = c

        self.relpos = sequence_relative_position(32, one_hot=True)

        self.p_relpos = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dmap   = Linear(c.pair_size, bias=False, initializer="linear")

        self.p_local_i = Linear(c.pair_size, bias=False)  
        self.p_local_j = Linear(c.pair_size, bias=False)

        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)
        self.mlp = MLP(c.pair_size * 2, c.pair_size, activation=gelu_salad, final_init="linear")

    def forward(
        self,
        local: torch.Tensor,                 # (N, local_dim)
        dmap: Optional[torch.Tensor],        # (N, N) or None
        neighbours: torch.Tensor,            # (N, K) 
        resi: torch.Tensor,                  # (N,)
        chain: torch.Tensor,                 # (N,)
        batch: torch.Tensor,                 # (N,)
        mask: torch.Tensor,                  # (N,) 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.c
        neighbours = neighbours.long()

        gathered_mask = _gather_rows(mask, neighbours)
        pair_mask = mask[:, None] * gathered_mask
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)

        if dmap is not None:
            dmap = _gather_dim1(dmap, neighbours)

        pair = self.p_relpos(self.relpos(resi, chain, batch, neighbours))

        if dmap is not None:
            pair = pair + self.p_dmap(
                torch.where(pair_mask[..., None].bool(), distance_rbf(dmap), 0.0)
            )

        pair = pair + self.p_local_i(local)[:, None]
        pair = pair + _gather_rows(self.p_local_j(local), neighbours)

        pair = self.ln(pair)
        pair = self.mlp(pair)
        return pair, pair_mask
    
def structure_augmentation_params(pos, batch, mask, generator=None):
    """Sample random rotation and translation parameters for data augmentation."""
    # center positions
    batch_l = batch.long()
    center = index_mean(pos[:, 1], batch_l, mask[:, None])
    # centering + random translation
    noise = torch.randn((batch.shape[0], 3), device=pos.device, dtype=pos.dtype, generator=generator)
    translation = noise[batch_l]
    # random rotation
    rotation = random_rotation(batch, device=pos.device, dtype=pos.dtype, generator=generator)
    return center, translation, rotation

def apply_structure_augmentation(pos, center, translation, rotation):
    """Apply a rotation and translation to an array of atom positions."""
    # center positions
    pos = pos - center[:, None]
    # apply transformation
    pos = rotation[:, None].apply_to_point(Vec3Array.from_array(pos)).to_tensor()
    pos = pos + translation[:, None]
    return pos

def apply_inverse_structure_augmentation(pos, center, translation, rotation):
    """Apply the inverse of a rotation and translation to an array of atom positions."""
    # invert translation
    pos = pos - translation[:, None]
    # invert rotation
    pos = rotation[:, None].apply_inverse_to_point(Vec3Array.from_array(pos)).to_tensor()
    # move to center
    pos = pos + center[:, None]
    return pos

def structure_augmentation(pos, batch, mask, generator=None):
    """Randomly rotate and translate an array of atom positions."""
    # get random augmentation parameters
    center, translation, rotation = structure_augmentation_params(pos, batch, mask, generator=generator)
    # apply random augmentation
    pos = apply_structure_augmentation(pos, center, translation, rotation)
    return pos

def random_rotation(batch, device=None, dtype=None, generator=None):
    """Sample a random rotation."""
    batch = batch.long()
    if batch.numel() == 0:
        x = torch.empty((0, 3), device=device, dtype=dtype)
        y = torch.empty((0, 3), device=device, dtype=dtype)
    else:
        num_items = int(batch.max().item()) + 1
        x0 = torch.randn((num_items, 3), device=device, dtype=dtype, generator=generator)
        y0 = torch.randn((num_items, 3), device=device, dtype=dtype, generator=generator)
        x = torch.index_select(x0, dim=0, index=batch)
        y = torch.index_select(y0, dim=0, index=batch)
    return Rot3Array.from_two_vectors(Vec3Array.from_array(x), Vec3Array.from_array(y))

def extract_dmap_neighbours(count: int = 32):
    """Extract nearest neighbours from a distance map."""

    def inner(distance, resi, chain, batch, mask, *, generator: torch.Generator | None = None):
        same_item = batch[:, None].eq(batch[None, :])
        mask_bool = mask.bool()

        weight = -3.0 * torch.log(distance + 1e-6)

        u = torch.rand(
            weight.shape,
            device=weight.device,
            dtype=weight.dtype,
            generator=generator,
        )
        u = u * (1.0 - 2e-6) + 1e-6

        gumbel = torch.log(-torch.log(u))

        weight = weight - gumbel

        distance2 = -weight

        inf = torch.tensor(torch.inf, device=distance2.device, dtype=distance2.dtype)
        distance2 = torch.where(same_item, distance2, inf)

        mask2 = same_item & mask_bool[:, None] & mask_bool[None, :]

        return get_neighbours(count)(distance2, mask2)

    return inner

class DecoderPairFeatures(nn.Module):
    """Pair features for the equivariant Decoder."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.relpos = sequence_relative_position(32, one_hot=True, pseudo_chains=True)

        self.p_relpos = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dmap = (
            Linear(c.pair_size, bias=False, initializer="linear")
            if getattr(c, "distogram_block", "none") != "none"
            else None
        )
        self.p_dist   = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dir    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_rot    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_vec    = Linear(c.pair_size, bias=False, initializer="linear")

        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)

        self.mlp = MLP(c.pair_size * 2, c.pair_size, activation=gelu_salad, final_init="linear")

    def forward(
        self,
        pos: torch.Tensor,                 # (N, A, 3)
        dmap: Optional[torch.Tensor],      # (N, N) 
        neighbours: torch.Tensor,          # (N, K) 
        resi: torch.Tensor,                # (N,)
        chain: torch.Tensor,               # (N,)
        batch: torch.Tensor,               # (N,)
        mask: torch.Tensor,                # (N,) 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.c
        neighbours = neighbours.long()
        
        gathered_mask = _gather_rows(mask, neighbours)
        pair_mask = mask[:, None] * gathered_mask
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)

        if dmap is not None:
            dmap = _gather_dim1(dmap, neighbours)

        pair = self.p_relpos(self.relpos(resi, chain, batch, neighbours))

        if dmap is not None:
            dmap_feat = distance_rbf(dmap)  
            pair = pair + self.p_dmap(torch.where(pair_mask[..., None].bool(), dmap_feat, 0.0))

        pos_v = Vec3Array.from_array(pos)

        pair = pair + self.p_dist(distance_features(pos_v, neighbours, d_min=0.0, d_max=22.0))
        pair = pair + self.p_dir(direction_features(pos_v, neighbours))
        pair = pair + self.p_rot(position_rotation_features(pos_v, neighbours))
        pair = pair + self.p_vec(pair_vector_features(pos_v, neighbours))

        pair = self.ln(pair)
        pair = self.mlp(pair)

        return pair, pair_mask
    
def _masked_rmsd_no_align(pred, target, atom_mask, residue_mask=None):
    """Compute RMSD without alignment, over masked atoms."""
    sq = ((pred - target) ** 2).sum(dim=-1)
    m = atom_mask.to(dtype=pred.dtype)
    if residue_mask is not None:
        m = m * residue_mask[:, None].to(dtype=pred.dtype)
    denom = torch.clamp(m.sum(), min=1.0)
    return torch.sqrt((sq * m).sum() / denom)


class DecoderPairFeaturesAtom37(nn.Module):
    """Pair features for the atom37 decoder branch."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.relpos = sequence_relative_position(32, one_hot=True, pseudo_chains=True)
        self.p_relpos = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dmap = (
            Linear(c.pair_size, bias=False, initializer="linear")
            if getattr(c, "distogram_block", "none") != "none" else None
        )
        self.p_dist = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dir  = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_rot  = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_vec  = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_local_self  = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_local_self_j = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_local_neigh = Linear(c.pair_size, bias=False, initializer="linear")
        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)
        self.mlp = MLP(c.pair_size * 2, c.pair_size, activation=gelu_salad, final_init="linear")

    def forward(self, pos37: torch.Tensor, dmap, neighbours: torch.Tensor,
                resi, chain, batch, mask,
                atom_mask37: torch.Tensor = None):
        c = self.c
        if neighbours.dtype != torch.long:
            neighbours = neighbours.long()

        gathered_mask = _gather_rows(mask, neighbours)
        pair_mask = mask[:, None] * gathered_mask
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)

        if dmap is not None:
            dmap_gathered = _gather_dim1(dmap, neighbours)
        else:
            dmap_gathered = None

        # use NCACOCB view for geometric features
        pos_view = Vec3Array.from_array(atom37_to_ncacocb(pos37))
        pair = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        if dmap_gathered is not None and self.p_dmap is not None:
            dmap_feat = distance_rbf(dmap_gathered)
            pair = pair + self.p_dmap(torch.where(pair_mask[..., None].bool(), dmap_feat, 0.0))
        pair = pair + self.p_dist(distance_features(pos_view, neighbours, d_min=0.0, d_max=22.0))
        pair = pair + self.p_dir(direction_features(pos_view, neighbours))
        pair = pair + self.p_rot(position_rotation_features(pos_view, neighbours))
        pair = pair + self.p_vec(pair_vector_features(pos_view, neighbours))
        if atom_mask37 is not None:
            pos37_v = Vec3Array.from_array(pos37)
            frames37, local_pos37 = extract_aa_frames(pos37_v)
            local_self_feat = atom37_local_feature_channels(local_pos37.to_tensor(), atom_mask37)
            local_neigh_pos = frames37[:, None, None].apply_inverse_to_point(
                _gather_vec3(pos37_v, neighbours)).to_tensor()
            neigh_mask37 = _gather_rows(atom_mask37, neighbours)
            local_neigh_feat = atom37_local_feature_channels(local_neigh_pos, neigh_mask37)
            pair = pair + self.p_local_self(local_self_feat)[:, None]
            pair = pair + self.p_local_self_j(_gather_rows(local_self_feat, neighbours))
            pair = pair + self.p_local_neigh(local_neigh_feat)
        pair = self.ln(pair)
        pair = self.mlp(pair)
        return pair, pair_mask


class DecoderUpdateAtom37(nn.Module):
    """GeGLU update with global averaging for the atom37 decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        local_size = int(c.local_size)

        self.pos_mlp = MLP(local_size * 2, local_size, activation=gelu_salad, final_init="zeros")
        self.local_update = Linear(local_size * int(c.factor), initializer="linear", bias=False)
        self.local_gate  = Linear(local_size * int(c.factor), initializer="relu", bias=False)
        self.chain_gate  = Linear(local_size * int(c.factor), initializer="relu", bias=False)
        self.batch_gate  = Linear(local_size * int(c.factor), initializer="relu", bias=False)
        self.out = Linear(local_size, initializer="zeros")

    def forward(self, local, pos37, chain, batch, mask,
                atom_mask37: torch.Tensor = None):
        _, local_pos37 = extract_aa_frames(Vec3Array.from_array(pos37))
        local_pos37_arr = local_pos37.to_tensor()
        if atom_mask37 is not None:
            local_pos37_arr = mask_atom37_local_positions(local_pos37_arr, atom_mask37)
        local_pos_flat = local_pos37_arr.reshape(local_pos37_arr.shape[0], -1)
        local = local + self.pos_mlp(local_pos_flat)
        local_update = self.local_update(local)
        local_gate  = gelu_salad(self.local_gate(local))
        chain_gate  = gelu_salad(self.chain_gate(local))
        batch_gate  = gelu_salad(self.batch_gate(local))
        hidden = index_mean(batch_gate * local_update, batch, mask[..., None])
        hidden = hidden + index_mean(chain_gate * local_update, chain, mask[..., None])
        hidden = hidden + local_gate * local_update
        return self.out(hidden)


def inject_pos_backbone_into_pos37(pos37: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Copy backbone [N, CA, C, O] from NCACOCB pos into atom37 coordinates.

    pos is NCACOCB format: [N=0, CA=1, C=2, O=3, CB=4].
    pos37 is atom37 format: [N=0, CA=1, C=2, CB=3, O=4, ...].
    O lives at index 3 in NCACOCB but at index 4 in atom37.
    """
    pos37 = pos37.clone()
    pos37[..., 0, :] = pos[..., 0, :]   # N
    pos37[..., 1, :] = pos[..., 1, :]   # CA
    pos37[..., 2, :] = pos[..., 2, :]   # C
    pos37[..., 4, :] = pos[..., 3, :]   # O: NCACOCB[3] → atom37[4]
    return pos37


class UpdatePositionsAtom37(nn.Module):
    """Equivariant position update for atom37 coordinates."""
    def __init__(self):
        super().__init__()
        self.proj: Optional[Linear] = None
        self._A: Optional[int] = None

    def forward(self, pos37: torch.Tensor, local_norm: torch.Tensor, *,
                scale: float = 10.0, symm=None,
                backbone_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        A = int(pos37.shape[-2])
        if self.proj is None:
            self._A = A
            self.proj = Linear(A * 3, initializer="zeros", bias=False,
                               device=local_norm.device, dtype=local_norm.dtype)
        else:
            if self._A != A:
                raise ValueError(f"UpdatePositionsAtom37 initialized with A={self._A}, got A={A}")
        frames37, local_pos37 = extract_aa_frames(Vec3Array.from_array(pos37))
        pos_update = float(scale) * self.proj(local_norm)
        if symm is not None:
            pos_update = symm(pos_update)
        pos_update = Vec3Array.from_array(pos_update.reshape(pos_update.shape[0], A, 3))
        local_pos37 = local_pos37 + pos_update
        pos37 = frames37[..., None].apply_to_point(local_pos37).to_tensor()
        if backbone_pos is not None:
            pos37 = inject_pos_backbone_into_pos37(pos37, backbone_pos)
        return pos37


class DecoderBlockAtom37(nn.Module):
    """Equivariant decoder block for the atom37 branch."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        if c.distogram_block == "mlp":
            self.distogram = QuickDistogram(c)
        elif c.distogram_block == "inner":
            self.distogram = InnerDistogram(c)
        elif c.distogram_block in (None, "none"):
            self.distogram = None
        else:
            raise ValueError(f"Unknown distogram_block={c.distogram_block}")

        self.ln_attn_main  = nn.LayerNorm(c.local_size)
        self.ln_attn_dist  = nn.LayerNorm(c.local_size) if self.distogram is not None else None
        self.ln_update     = nn.LayerNorm(c.local_size)
        self.ln_pos        = nn.LayerNorm(c.local_size)

        self.attn_main = SparseStructureAttention(c)
        self.attn_dist = SparseStructureAttention(c) if self.distogram is not None else None

        self.update = DecoderUpdateAtom37(c)
        self.pos_update = UpdatePositionsAtom37()

        nr = int(getattr(c, "num_random_neighbours", 32))
        self.current_neigh = extract_neighbours(num_index=16, num_spatial=16, num_random=nr)
        self.dmap_neigh = extract_dmap_neighbours(count=32)

        self.pair_features_main = DecoderPairFeaturesAtom37(c)
        self.pair_features_dist = DecoderPairFeaturesAtom37(c) if self.distogram is not None else None

    def forward(self, features: torch.Tensor, pos37: torch.Tensor,
                atom_mask37: torch.Tensor, backbone_pos,
                resi, chain, batch, mask,
                sup_neighbours=None, generator=None):
        c = self.config
        # CB from atom37 view for neighbours
        pos37_v = Vec3Array.from_array(pos37)
        distogram_logits = None
        dmap_neighbours = None
        if self.distogram is not None:
            dist_full_logits, dmap37 = self.distogram(features, resi, chain, batch, None)
            if sup_neighbours is None:
                raise ValueError("sup_neighbours required when distogram is enabled")
            distogram_logits = _gather_dim1(dist_full_logits, sup_neighbours.long())
        else:
            cb37 = Vec3Array.from_array(atom37_to_ncacocb(pos37)[:, -1])
            dmap37 = (cb37[:, None] - cb37[None, :]).norm()
            distogram_logits = torch.zeros((1,), dtype=features.dtype, device=features.device)
        dmap_neighbours = self.dmap_neigh(dmap37.detach(), resi, chain, batch, mask,
                                           generator=generator)

        current_neighbours = self.current_neigh(pos37_v, resi, chain, batch, mask, generator=generator)

        pair, pair_mask = self.pair_features_main(
            pos37, dmap37, current_neighbours, resi, chain, batch, mask, atom_mask37)
        features = features + self.attn_main(
            self.ln_attn_main(features),
            pos37 / float(c.sigma_data), pair, pair_mask,
            current_neighbours, resi, chain, batch, mask)

        if self.pair_features_dist is not None:
            pair2, pair2_mask = self.pair_features_dist(
                pos37, dmap37, dmap_neighbours, resi, chain, batch, mask, atom_mask37)
            features = features + self.attn_dist(
                self.ln_attn_dist(features),
                pos37 / float(c.sigma_data), pair2, pair2_mask,
                dmap_neighbours, resi, chain, batch, mask)

        features = features + self.update(
            self.ln_update(features), pos37, chain, batch, mask, atom_mask37)

        local_norm = self.ln_pos(features)
        symm = getattr(c, "symm", None)
        pos37 = self.pos_update(pos37, local_norm, scale=float(c.sigma_data), symm=symm)
        if pos37.dtype != local_norm.dtype:
            pos37 = pos37.to(dtype=local_norm.dtype)
        return features, pos37, distogram_logits


class DecoderStackAtom37(nn.Module):
    """Atom37 decoder stack."""

    def __init__(self, config, block_cls):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([block_cls(config) for _ in range(config.depth)])

    def forward(self, local37, pos37, atom_mask37, backbone_pos,
                resi, chain, batch, mask,
                sup_neighbours=None, generator=None):
        trajectory37 = []
        sup_distograms = []
        for block in self.blocks:
            local37, pos37, sup_dist = block(
                local37, pos37, atom_mask37, backbone_pos,
                resi, chain, batch, mask,
                sup_neighbours=sup_neighbours, generator=generator)
            trajectory37.append(pos37)
            sup_distograms.append(sup_dist)
        trajectory37 = torch.stack(trajectory37, dim=0)
        sup_distograms = torch.stack(sup_distograms, dim=0)
        return local37, pos37, trajectory37, sup_distograms


class QuickDistogram(nn.Module):
    """Per-layer distogram module.

    Not used in the manuscript (too slow to be viable).
    """
    def __init__(self, config, bins: int = 16, start: float = 0.0, stop: float = 22.0):
        super().__init__()
        self.config = config
        self.bins = int(bins)

        self.left = Linear(32, bias=False)
        self.right = Linear(32, bias=False)

        self.relpos = sequence_relative_position(32, one_hot=True)
        self.relpos_proj = Linear(32, bias=False)

        self.ln = nn.LayerNorm(32, elementwise_affine=True)
        self.head = MLP(size=64, out_size=self.bins, depth=2, activation=gelu_salad, final_init="zeros")

        step = (stop - start) / self.bins
        bin_centers = torch.arange(self.bins, dtype=torch.get_default_dtype()) * step + step / 2.0
        self.register_buffer("bin_centers", bin_centers, persistent=False)

    def forward(
        self,
        features: torch.Tensor,         # (N, C)
        resi: torch.Tensor,             # (N,)
        chain: torch.Tensor,            # (N,)
        batch: torch.Tensor,            # (N,)
        neighbours: torch.Tensor,       # (N, K) 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = features.shape[0]
        neighbours = neighbours.long()

        dl = self.left(features)   # (N, 32)
        dr = self.right(features)  # (N, 32)

        ### safe gather for neighbours (handle -1)
        # valid = neighbours != -1
        # neigh = neighbours.clamp_min(0)
        # dcode = dl[:, None, :] + dr[neigh]
        
        dcode = dl[:, None, :] + _gather_rows(dr, neighbours)
        
        pair_resi = self.relpos(resi, chain, batch, neighbours)
        dcode = dcode + self.relpos_proj(pair_resi)

        dcode = self.ln(dcode)

        # logits: (N, K, bins)
        distogram_logits = self.head(dcode)
        distogram_logits = F.log_softmax(distogram_logits, dim=-1)

        # (c) mask out invalid neighbours (neighbour can be -1)
        # if valid is not None:
        #   distogram_logits = distogram_logits.masked_fill(~valid[..., None], -1e9)

        probs = torch.softmax(distogram_logits, dim=-1)
        bin_centers = self.bin_centers.to(device=probs.device, dtype=probs.dtype)
        dmap = (probs * bin_centers).sum(dim=-1)

        return distogram_logits, dmap


class InnerDistogram(nn.Module):
    """Light-weight per-layer distogram module. (+dist in the manuscript)"""
    def __init__(self, config, bins: int = 16, heads: int = 8, start: float = 0.0, stop: float = 22.0):
        super().__init__()
        self.config = config
        self.bins = int(bins)
        self.heads = int(heads)

        self.code_proj = Linear(self.heads * self.bins, bias=False)
        self.gate_proj = Linear(self.heads * self.bins, bias=False)

        self.inner_weight = nn.Parameter(torch.zeros(self.heads, self.heads, self.bins))

        step = (stop - start) / self.bins
        bin_centers = torch.arange(self.bins, dtype=torch.get_default_dtype()) * step + step / 2.0
        self.register_buffer("bin_centers", bin_centers, persistent=False)

    def forward(
        self,
        features: torch.Tensor,   # (N, C)
        resi: torch.Tensor,       
        chain: torch.Tensor,      
        batch: torch.Tensor,      
        neighbours: torch.Tensor  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = features.shape[0]

        dcode = self.code_proj(features).reshape(N, self.heads, self.bins)  # (N,8,16)
        dgate = self.gate_proj(features).reshape(N, self.heads, self.bins)  # (N,8,16)
        dgate = gelu_salad(dgate)

        logits = torch.einsum("iax,jbx,abx->ijx", dcode, dgate, self.inner_weight)

        logits = 0.5 * (logits + logits.transpose(0, 1))

        logits = F.log_softmax(logits, dim=-1)

        probs = torch.softmax(logits, dim=-1)
        bin_centers = self.bin_centers.to(device=probs.device, dtype=probs.dtype)
        dmap = (probs * bin_centers).sum(dim=-1)
        return logits, dmap


class NonEquivariantDecoderBlock(nn.Module):
    """Fully non-equivariant decoder block."""

    def __init__(self, config, name: Optional[str] = "decoder_block"):
        super().__init__()
        self.config = config
        c = self.config

        distogram_block = {
            "mlp": QuickDistogram,
            "inner": InnerDistogram,
            "none": None,
        }[c.distogram_block]
        if distogram_block is None:
            raise ValueError('NonEquivariantDecoderBlock requires distogram_block != "none"')

        self.dist = distogram_block(c)

        self.dmap_neigh = extract_dmap_neighbours(count=32)

        self.ln_pair_in = nn.LayerNorm(c.local_size, elementwise_affine=True)
        self.ln_attn_in = nn.LayerNorm(c.local_size, elementwise_affine=True)
        self.ln_upd_in = nn.LayerNorm(c.local_size, elementwise_affine=True)

        self.pair_features = NonequivariantDecoderPairFeatures(c)

        self.attn = SparseAttention(size=c.key_size, heads=c.heads)
        self.update = NonEquivariantDecoderUpdate(c)

    def forward(
        self,
        features: torch.Tensor,
        resi: torch.Tensor,
        chain: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        sup_neighbours: Optional[torch.Tensor] = None,
        pos_gt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = features.shape[0]
        if sup_neighbours is None:
            raise ValueError("sup_neighbours is required")

        sup_neighbours = sup_neighbours.long()

        dist_full_logits, dmap = self.dist(features, resi, chain, batch, None)  # (N,N,B), (N,N)

        distogram_logits = _gather_dim1(dist_full_logits, sup_neighbours)

        dmap_neighbours = self.dmap_neigh(dmap.detach(), resi, chain, batch, mask)

        pair, pair_mask = self.pair_features(
            self.ln_pair_in(features),
            dmap,
            dmap_neighbours,
            resi,
            chain,
            batch,
            mask,
        )

        features = features + self.attn(
            self.ln_attn_in(features),
            pair,
            dmap_neighbours,
            pair_mask,
        )

        features = features + self.update(
            self.ln_upd_in(features),
            chain,
            batch,
            mask,
        )

        return features, distogram_logits

class NonEquivariantDecoderUpdate(nn.Module):
    """Fully non-equivariant update for the decoder. 
    
    Not used in the manuscript
    """

    def __init__(self, config, name: Optional[str] = "light_global_update"):
        super().__init__()
        self.config = config

        c = self.config
        local_size = int(c.local_size)
        factor = int(c.factor)

        self.local_update = Linear(local_size * factor, initializer="linear", bias=False)
        self.local_gate = Linear(local_size * factor, initializer="relu", bias=False)
        self.chain_gate = Linear(local_size * factor, initializer="relu", bias=False)
        self.batch_gate = Linear(local_size * factor, initializer="relu", bias=False)

        self.out = Linear(local_size, initializer="zeros")

    def forward(self, local, chain, batch, mask):
        local_update = self.local_update(local)
        local_gate = gelu_salad(self.local_gate(local))
        chain_gate = gelu_salad(self.chain_gate(local))
        batch_gate = gelu_salad(self.batch_gate(local))

        hidden = index_mean(batch_gate * local_update, batch, mask[..., None])
        hidden = hidden + index_mean(chain_gate * local_update, chain, mask[..., None])
        hidden = hidden + local_gate * local_update

        result = self.out(hidden)
        return result
    

class UpdatePositions(nn.Module):
    """Equivariant position update."""
    def __init__(self):
        super().__init__()
        self.proj: Optional[Linear] = None
        self._A: Optional[int] = None

    def forward(
        self,
        pos: torch.Tensor,  
        local_norm: torch.Tensor, 
        *,
        scale: float = 10.0,
        symm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        A = int(pos.shape[-2])

        if self.proj is None:
            self._A = A
            self.proj = Linear(
                A * 3,
                initializer="zeros",
                bias=False,
                device=local_norm.device,
                dtype=local_norm.dtype,
            )
        else:
            if self._A != A:
                raise ValueError(
                    f"UpdatePositions initialized with A={self._A}, but got A={A}. "
                )

        frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))

        pos_update = float(scale) * self.proj(local_norm)

        if symm is not None:
            pos_update = symm(pos_update)

        pos_update = Vec3Array.from_array(pos_update.reshape(pos_update.shape[0], A, 3))
        local_pos = local_pos + pos_update

        pos_out = frames[..., None].apply_to_point(local_pos).to_tensor()
        return pos_out

# TODO: move to __init__ of SemiEquivariantDecoderBlock 
def semiequivariant_update_positions(
    pos: torch.Tensor,
    local_norm: torch.Tensor,
    scale: float = 10.0,
    symm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Partially equivariant position update. (NEQ)"""

    pos_update = float(scale) * Linear(
        pos.shape[-2] * 3,
        initializer="zeros",
        bias=False,
        device=local_norm.device,
        dtype=local_norm.dtype,
    )(local_norm)

    pos_update = pos_update.reshape(*pos_update.shape[:-1], -1, 3)
    if symm is not None:
        pos_update = symm(pos_update)

    pos = pos + pos_update
    return pos
