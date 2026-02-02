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
    sequence_relative_position, extract_neighbours,
    get_neighbours, distance_rbf, 
    index_mean, 
    get_random_neighbours, 
    index_sum,
    single_protein_sidechains)

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

DEBUG = True

def stop_gradient(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if hasattr(x, "map_tensor_fn"):
        return x.map_tensor_fn(lambda t: t.detach())
    raise TypeError(f"stop_gradient: unsupported type {type(x)}")
    
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

                sup_neighbours = sup_neighbours.to(torch.long)
                idx = torch.arange(sup_neighbours.shape[0], device=sup_neighbours.device, dtype=torch.long)[:, None]
                distogram_logits = dist_full_logits[idx, sup_neighbours]

            dmap_neighbours = self.dmap_neigh(dmap.detach(), resi, chain, batch, mask, generator=generator)
        else:
            distogram_logits = torch.zeros((1,), dtype=torch.float32, device=features.device)
            dmap = None

        current_neighbours = self.current_neigh(
            Vec3Array.from_array(pos), resi, chain, batch, mask
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
        return features, pos.to(local_norm.dtype), distogram_logits
    
class Decoder(nn.Module):
    """Protein structure decoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        if c.equivariance == "nonequivariant":
            decoder_module = NonEquivariantDecoderStack
            decoder_block = NonEquivariantDecoderBlock
        elif c.equivariance == "semiequivariant":
            decoder_module = SemiEquivariantDecoderStack
            decoder_block = SemiEquivariantDecoderBlock
        else:
            decoder_module = DecoderStack
            decoder_block = DecoderBlock

        self.decoder_stack = decoder_module(c, decoder_block)
        self.aa_decoder = AADecoder(c)

        self.use_latent_diff = bool(c.input_diffusion or c.latent_diffusion)
        if self.use_latent_diff:
            self.latent_ln = nn.LayerNorm(c.local_size) 
            out_dim = c.local_size if c.noembed else c.latent_size
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
        
    def forward(self, data, prev, generator=None):
        if prev is None:
            prev = self.init_prev(data)
        c = self.config

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
            if c.vp_diffusion:
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
        if c.eval:  
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
        return {
            "pos": data["pos"],
            "local": torch.zeros(
                (data["pos"].shape[0], c.local_size),
                device=data["pos"].device,
                dtype=torch.float32,
            ),
        }

    def prepare_features(self, data, prev):
        c = self.config

        pos = prev["pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        latent = data["latent"]
        mask = data["mask"]

        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))

        local_features = [
            local_pos.to_tensor().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1),
            latent
        ]

        if c.time_embedding and c.latent_diffusion and "time" in data:
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

        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]  
        mask_bool = mask.bool()
        mask_f = mask.to(dtype=torch.float32)

        losses = {}
        total = torch.zeros((), device=mask.device, dtype=torch.float32)

        # AA NLL loss
        aa_gt = data["aa_gt"]
        aa_predict_mask = mask * (aa_gt != 20)
       
        aa_logp = result["aa"] 
        
        aa_gt_long = aa_gt.to(torch.long)
        # (c) jax-equivalent one-hot semantics
        valid = (aa_gt_long >= 0) & (aa_gt_long < 20)
        aa_one_hot = F.one_hot(aa_gt_long.clamp(0, 19), num_classes=20).to(aa_logp.dtype)
        aa_one_hot = aa_one_hot * valid.unsqueeze(-1).to(aa_one_hot.dtype)
     
        aa_nll = -(aa_logp * aa_one_hot).sum(dim=-1)
        aa_nll = torch.where(aa_predict_mask.to(torch.bool), aa_nll, torch.zeros_like(aa_nll))
        aa_nll = aa_nll.sum() / torch.clamp(aa_predict_mask.to(torch.float32).sum(), min=1.0)
        losses["aa"] = aa_nll
        total = total + float(c.aa_weight) * aa_nll

        # position losses
        base_weight = mask / torch.clamp(index_sum(mask.to(torch.float32), batch, mask), min=1.0) / (batch.max() + 1)
        
        # sparse neighbour FAPE ** 2
        pair_mask = (batch[:, None] == batch[None, :])
        pair_mask = pair_mask * (mask[:, None] * mask[None, :])

        pos_gt = data["pos_gt"]
        pos_gt = torch.where(mask[:, None, None].to(torch.bool), pos_gt, torch.zeros_like(pos_gt))
        pos_gt = Vec3Array.from_array(pos_gt)

        frames_gt, _ = extract_aa_frames(stop_gradient(pos_gt))

        # CB distance 
        distance = data["dmap"]
        distance = torch.where(pair_mask.to(torch.bool), distance, torch.full_like(distance, float("inf")))
        # get random neighbours to compute sparse FAPE on
        neighbours = get_random_neighbours(c.fape_neighbours)(distance, batch, mask, generator=generator)
        mask_neighbours = (neighbours != -1) * mask[:, None] * mask[neighbours]
        pos_gt_local = frames_gt[:, None, None].apply_inverse_to_point(pos_gt[neighbours])
        
        traj = result["trajectory"]  # tensor, (T, N, A, 3)
        T = traj.shape[0]

        fape_base_list = []
        # (c) torch equivalent for jax.vmap
        for t in range(T):
            traj_t = Vec3Array.from_array(traj[t])      # (N, A, 3)
            frames_t, _ = extract_aa_frames(traj_t)            # (N,) Rigid3Array
            traj_local_t = frames_t[:, None, None].apply_inverse_to_point(traj_t[neighbours])  # (N, K, A)

            fape_base_t = (traj_local_t - pos_gt_local).norm2()  # (N, K)
            fape_base_list.append(fape_base_t)
        fape_base = torch.stack(fape_base_list, dim=0)  # (T, N, K)

        fape_clipped = torch.clamp(fape_base, 0.0, float(c.clip_fape))

        if getattr(c, "unclipped_weight", 0):
            fape_clipped = fape_base * float(c.unclipped_weight) + fape_clipped

        if getattr(c, "no_fape2", False):
            # use FAPE instead of FAPE ** 2
            fape_clipped = torch.sqrt(torch.clamp(fape_clipped, min=1e-6))

        fape_traj = fape_clipped.mean(dim=-1)  # [T,N,K] 
        fape_traj = torch.where(
            mask_neighbours[None].to(torch.bool),
            fape_traj,
            torch.zeros_like(fape_traj),
        )

        fape_traj = fape_traj.sum(dim=-1) / torch.clamp(mask_neighbours.sum(dim=1)[None], min=1.0)
        fape_traj = (fape_traj * base_weight).sum(dim=-1)
        losses["fape"] = fape_traj[-1] / 3.0
        losses["fape_trajectory"] = fape_traj.mean() / 3.0
        fape_loss = (float(c.fape_weight) * fape_traj[-1] + float(c.fape_trajectory_weight) * fape_traj.mean()) / 3.0
        total = total + fape_loss
       
        # sup distogram loss
        if getattr(c, "distogram_block", None) not in (None, "none"):
            cb_gt = pos_gt[:, -1]  
            sup_neighbours = result["sup_neighbours"]
            sup_mask = (sup_neighbours != -1) * mask[:, None] * mask[sup_neighbours]
            dist_gt = (cb_gt[:, None] - cb_gt[sup_neighbours]).norm()
            dist_one_hot = distance_one_hot(dist_gt, 0.0, 22.0, 16)
            distogram_nll = -(result["sup_distogram"] * dist_one_hot[None]).sum(dim=-1)
            distogram_nll = torch.where(sup_mask.to(torch.bool), distogram_nll, torch.zeros_like(distogram_nll)).sum(dim=-1)
            distogram_nll = distogram_nll / torch.clamp(sup_mask.sum(dim=1), min=1.0)
            distogram_nll = (distogram_nll * base_weight).sum(dim=1)
            losses["distogram"] = distogram_nll[-1]
            losses["distogram_trajectory"] = distogram_nll.mean()
            total = total + 10.0 * distogram_nll[-1] + 5.0 * distogram_nll.mean()
            if DEBUG:
                print("kabsch_rmsd loss:", distogram_nll[-1].item())
 
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
            total = total + pos_loss
            if DEBUG:
                print("kabsch_rmsd loss:", pos_loss.item())

        # local loss
        atom_pos = Vec3Array.from_array(result["atom_pos"])
        atom_pos_gt = Vec3Array.from_array(data["atom_pos"])

        local_neighbours = get_spatial_neighbours(count=c.local_neighbours)(
            pos_gt[:, -1], batch, mask
        )

        mask_gt = data["atom_mask"][local_neighbours]

        frames_gt, _ = extract_aa_frames(atom_pos_gt)
        atom_pos_gt = frames_gt[:, None, None].apply_inverse_to_point(atom_pos_gt[local_neighbours])

        frames, _ = extract_aa_frames(atom_pos)
        atom_pos = frames[:, None, None].apply_inverse_to_point(atom_pos[local_neighbours])

        local_loss = torch.where(
            mask_gt.to(torch.bool),
            (atom_pos - atom_pos_gt).norm2(),
            torch.zeros_like((atom_pos - atom_pos_gt).norm2()),
        ).sum(dim=(1, 2))

        local_loss = local_loss / torch.clamp(mask_gt.sum(dim=(1, 2)), min=1)

        local_loss = (torch.where(mask.to(torch.bool), local_loss, torch.zeros_like(local_loss)) * base_weight).sum() / 3.0

        losses["local"] = local_loss
        total = total + float(c.local_weight) * local_loss

        # VQ losses
        if getattr(c, "codebook_size", 0) and (not getattr(c, "is_decoder", False)):
            cl = result["codebook_losses"]
            losses.update(cl)
            if not getattr(c, "state", False):
                total = total + float(c.codebook_loss_scale) * (cl["codebook"] + cl["unassigned"])
            total = total + float(c.codebook_loss_scale) * float(c.codebook_b) * cl["commitment"]
            if DEBUG:
                print("codebook loss:", (cl["codebook"] + cl["unassigned"]).item())
                print("commitment loss:", cl["commitment"].item())
                
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
            total = total + float(c.latent_loss_scale) * weighted_loss
            if DEBUG:
                print("latent loss:", weighted_loss.item())

        # AlphaFold violation loss. (not used in manuscript)
        if getattr(c, "violation_scale", 0):
            res_mask = data["mask"].bool()
            pred_mask = get_atom14_mask(data["aa_gt"]).to(res_mask.device).to(torch.float32) * res_mask[:, None].to(torch.float32)

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
            total = total + float(c.violation_scale) * violation.mean()
            if DEBUG:
                print("violation loss:", violation.mean().item())

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
            F.one_hot(aa_gt.to(torch.long), num_classes=21).to(local.dtype),
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

        aa_oh = F.one_hot(aa.to(torch.long), 21).to(dtype=features.dtype)
        aa_emb = self.ln_aa(self.aa_linear(aa_oh))         
        pair = pair + aa_emb[neighbours]                  

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
            idx = torch.arange(N, device=sup_neighbours.device)[:, None]

            distogram_logits = dist_full_logits[idx, sup_neighbours]
            
            dmap_neighbours = self.dmap_neigh(dmap.detach(), resi, chain, batch, mask, generator=generator)

        current_neighbours = self.current_neigh(
            Vec3Array.from_array(pos),
            resi, chain, batch, mask
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

        return features, pos.to(local_norm.dtype), distogram_logits

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
        neighbours = neighbours.to(torch.long)

        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)

        index = torch.arange(neighbours.shape[0], device=neighbours.device)

        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]

        pair = self.p_relpos(self.relpos(resi, chain, batch, neighbours))

        if dmap is not None:
            dmap_feat = distance_rbf(dmap)
            pair = pair + self.p_dmap(torch.where(pair_mask[..., None].to(torch.bool), dmap_feat, 0.0))

        pos_v = Vec3Array.from_array(pos)

        pair = pair + self.p_dist(distance_features(pos_v, neighbours, d_min=0.0, d_max=22.0))

        pos_arr = pos_v.to_tensor() if hasattr(pos_v, "to_tensor") else pos_v.to_tensor()  # (N, A, 3)
        K = neighbours.shape[1]

        local_neighbourhood = torch.cat(
            (
                pos_arr[:, None].expand(-1, K, -1, -1),
                pos_arr[neighbours],
            ),
            dim=-2,
        ) / float(c.sigma_data)

        center_neighbourhood = local_neighbourhood - pos_arr[:, None, 1, None]
        center_neighbourhood = (Vec3Array.from_array(center_neighbourhood).normalized())
        center_neighbourhood = center_neighbourhood.to_tensor() if hasattr(center_neighbourhood, "to_tensor") else center_neighbourhood.to_tensor()

        local_neighbourhood = local_neighbourhood.reshape(*neighbours.shape, -1)
        center_neighbourhood = center_neighbourhood.reshape(*neighbours.shape, -1)

        pair = pair + self.p_neigh(torch.cat((local_neighbourhood, center_neighbourhood), dim=-1))

        dirs = (pos_v[:, None, :, None] - pos_v[neighbours, None, :]).to_tensor()
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
        neighbours = neighbours.to(torch.long)

        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)

        index = torch.arange(neighbours.shape[0], device=neighbours.device)

        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]

        pair = self.p_relpos(self.relpos(resi, chain, batch, neighbours))

        if dmap is not None:
            pair = pair + self.p_dmap(
                torch.where(pair_mask[..., None].to(torch.bool), distance_rbf(dmap), 0.0)
            )

        pair = pair + self.p_local_i(local)[:, None]
        pair = pair + self.p_local_j(local)[neighbours]

        pair = self.ln(pair)
        pair = self.mlp(pair)
        return pair, pair_mask
    
def structure_augmentation_params(pos, batch, mask, generator=None):
    """Sample random rotation and translation parameters for data augmentation."""
    # center positions
    batch_l = batch.to(torch.long)
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
    N = batch.shape[0]
    x0 = torch.randn((N, 3), device=device, dtype=dtype, generator=generator)
    y0 = torch.randn((N, 3), device=device, dtype=dtype, generator=generator)
    x = x0[batch]
    y = y0[batch]
    return Rot3Array.from_two_vectors(Vec3Array.from_array(x), Vec3Array.from_array(y))

def extract_dmap_neighbours(count: int = 32):
    """Extract nearest neighbours from a distance map."""

    def inner(distance, resi, chain, batch, mask, *, generator: torch.Generator | None = None):
        same_item = batch[:, None].eq(batch[None, :])

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

        mask2 = (mask[:, None] * mask[None, :]) * same_item.to(mask.dtype)

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
        neighbours = neighbours.to(torch.long)
        
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)

        index = torch.arange(neighbours.shape[0], device=neighbours.device)

        if dmap is not None:
            dmap = dmap[index[:, None], neighbours]

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
        bin_centers = torch.arange(self.bins, dtype=torch.float32) * step + step / 2.0
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
        neighbours = neighbours.to(torch.long)

        dl = self.left(features)   # (N, 32)
        dr = self.right(features)  # (N, 32)

        ### safe gather for neighbours (handle -1)
        # valid = neighbours != -1
        # neigh = neighbours.clamp_min(0)
        # dcode = dl[:, None, :] + dr[neigh]
        
        dcode = dl[:, None, :] + dr[neighbours]
        
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
        dmap = (probs * self.bin_centers.to(probs.dtype)).sum(dim=-1)

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
        bin_centers = torch.arange(self.bins, dtype=torch.float32) * step + step / 2.0
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
        dmap = (probs * self.bin_centers.to(probs.dtype)).sum(dim=-1)
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

        sup_neighbours = sup_neighbours.to(torch.long)

        dist_full_logits, dmap = self.dist(features, resi, chain, batch, None)  # (N,N,B), (N,N)

        idx = torch.arange(N, device=sup_neighbours.device)[:, None]
        distogram_logits = dist_full_logits[idx, sup_neighbours] # -1 maybe

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
            self.proj = Linear(A * 3, initializer="zeros", bias=False)
        else:
            if self._A != A:
                raise ValueError(
                    f"UpdatePositions initialized with A={self._A}, but got A={A}. "
                )

        frames, local_pos = extract_aa_frames(Vec3Array.from_array(pos))

        pos_update = float(scale) * self.proj(local_norm)

        if symm is not None:
            pos_update = symm(pos_update)

        pos_update = Vec3Array.from_array(pos_update.view(pos_update.shape[0], A, 3))
        local_pos = local_pos + pos_update

        pos_out = frames[..., None].apply_to_point(local_pos).to_tensor()
        return pos_out

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
    )(local_norm)

    pos_update = pos_update.reshape(*pos_update.shape[:-1], -1, 3)
    if symm is not None:
        pos_update = symm(pos_update)

    pos = pos + pos_update
    return pos
