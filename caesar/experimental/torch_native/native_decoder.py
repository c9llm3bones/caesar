"""Tensor-native backbone decoder stack modules."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from caesar.experimental.torch_native.native_encoder import (
    DIRECTION5_FEATURES,
    EncoderUpdateNative,
    DISTANCE5_FEATURES,
    LOCAL_FRAME5_FEATURES,
    PairFeaturesNative,
    PAIR_VECTOR5_FEATURES,
    RELPOS32_FEATURES,
    ROTATION_FEATURES,
    extract_neighbours_native,
    topk_neighbours,
)
from caesar.experimental.torch_native.native_geometric import (
    SparseStructureAttentionNative,
    direction_features,
    distance_features,
    gather_nodes,
    pair_vector_features,
    position_rotation_features,
)
from caesar.experimental.torch_native.tensor_geometry import (
    apply_frame,
    distance_rbf,
    frames_from_ncac,
    index_mean,
    invert_apply_frame,
    local_positions,
    sequence_relative_position,
)
from caesar.aflib.common import residue_constants as rc
from caesar.experimental.torch_native.basic import Linear, MLP, gelu_salad, init_linear


def gather_dim1(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()
    safe = index.remainder(source.shape[1])
    view_shape = (*safe.shape, *([1] * (source.ndim - 2)))
    expand_shape = (*safe.shape, *source.shape[2:])
    gather_index = safe.view(view_shape).expand(expand_shape)
    return torch.gather(source, dim=1, index=gather_index)


def extract_dmap_neighbours_native(distance: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor, count: int = 32) -> torch.Tensor:
    same_item = batch[:, None].eq(batch[None, :])
    mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
    pair_mask = same_item & mask_bool[:, None] & mask_bool[None, :]
    dist = torch.where(same_item, distance, torch.full_like(distance, float("inf")))
    return topk_neighbours(dist, pair_mask, int(count))


def extract_spatial_neighbours_native(pos: torch.Tensor, batch: torch.Tensor, mask: torch.Tensor, count: int = 32) -> torch.Tensor:
    """Fixed-shape spatial neighbours for tensor-native decoder side paths."""
    distance = torch.linalg.norm(pos[:, None] - pos[None, :], dim=-1)
    same_item = batch[:, None].eq(batch[None, :])
    mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
    pair_mask = same_item & mask_bool[:, None] & mask_bool[None, :]
    distance = torch.where(same_item, distance, torch.full_like(distance, float("inf")))
    return topk_neighbours(distance, pair_mask, int(count))


def _rc_tensor(table, aatype: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(table, device=aatype.device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    flat = aatype.long().reshape(-1).remainder(tensor.shape[0])
    gathered = torch.index_select(tensor, dim=0, index=flat)
    return gathered.reshape(*aatype.shape, *tensor.shape[1:])


def _compose_frames(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    rot = torch.matmul(left[..., :3, :3], right[..., :3, :3])
    trans = apply_frame(left[..., :3, :3], left[..., :3, 3], right[..., :3, 3])
    out = torch.zeros((*rot.shape[:-2], 4, 4), device=rot.device, dtype=rot.dtype)
    out[..., :3, :3] = rot
    out[..., :3, 3] = trans
    out[..., 3, 3] = 1.0
    return out


def torsion_angles_to_frames_native(
    aatype: torch.Tensor,
    backbone_frames: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Tensor-native OpenFold torsion frame construction."""
    default_frames = _rc_tensor(rc.restype_rigid_group_default_frame, aatype, dtype=angles.dtype)

    sin_angles = torch.cat((torch.zeros_like(aatype, dtype=angles.dtype).unsqueeze(-1), angles[..., 0]), dim=-1)
    cos_angles = torch.cat((torch.ones_like(aatype, dtype=angles.dtype).unsqueeze(-1), angles[..., 1]), dim=-1)
    zeros = torch.zeros_like(sin_angles)
    ones = torch.ones_like(sin_angles)
    rot_x = torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=-1),
            torch.stack((zeros, cos_angles, -sin_angles), dim=-1),
            torch.stack((zeros, sin_angles, cos_angles), dim=-1),
        ),
        dim=-2,
    )

    all_frames = default_frames.clone()
    all_frames[..., :3, :3] = torch.matmul(default_frames[..., :3, :3], rot_x)
    chi1_frame = all_frames[..., 4, :, :]
    chi2_frame = _compose_frames(chi1_frame, all_frames[..., 5, :, :])
    chi3_frame = _compose_frames(chi2_frame, all_frames[..., 6, :, :])
    chi4_frame = _compose_frames(chi3_frame, all_frames[..., 7, :, :])
    frames_to_backbone = torch.cat(
        (
            all_frames[..., 0:5, :, :],
            chi2_frame[..., None, :, :],
            chi3_frame[..., None, :, :],
            chi4_frame[..., None, :, :],
        ),
        dim=-3,
    )
    return _compose_frames(backbone_frames[..., None, :, :], frames_to_backbone)


def frames_to_atom14_pos_native(aatype: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """Place atom14 literature positions using tensor-native rigid frames."""
    group_idx = _rc_tensor(rc.restype_atom14_to_rigid_group, aatype).long()
    group_mask = F.one_hot(group_idx, num_classes=8).to(dtype=frames.dtype)
    atom_frames = (frames[:, None] * group_mask[..., None, None]).sum(dim=-3)
    lit_positions = _rc_tensor(rc.restype_atom14_rigid_group_positions, aatype, dtype=frames.dtype)
    pred = apply_frame(atom_frames[..., :3, :3], atom_frames[..., :3, 3], lit_positions)
    atom_mask = _rc_tensor(rc.restype_atom14_mask, aatype, dtype=frames.dtype)
    return pred * atom_mask[..., None]


class QuickDistogramNative(nn.Module):
    """Tensor-native equivalent of decoder.QuickDistogram."""

    def __init__(self, config, bins: int = 16, start: float = 0.0, stop: float = 22.0):
        super().__init__()
        self.config = config
        self.bins = int(bins)
        self.left = Linear(config.local_size, 32, bias=False)
        self.right = Linear(config.local_size, 32, bias=False)
        self.relpos = sequence_relative_position(32, one_hot=True)
        self.relpos_proj = Linear(RELPOS32_FEATURES, 32, bias=False)
        self.ln = nn.LayerNorm(32, elementwise_affine=True)
        self.head = MLP(32, size=64, out_size=self.bins, depth=2, activation=gelu_salad, final_init="zeros")

        step = (stop - start) / self.bins
        bin_centers = torch.arange(self.bins, dtype=torch.get_default_dtype()) * step + step / 2.0
        self.register_buffer("bin_centers", bin_centers, persistent=False)

    def forward(self, features, resi, chain, batch, neighbours):
        neighbours = neighbours.long()
        dl = self.left(features)
        dr = self.right(features)
        dcode = dl[:, None, :] + gather_nodes(dr, neighbours)
        dcode = dcode + self.relpos_proj(self.relpos(resi, chain, batch, neighbours))
        dcode = self.ln(dcode)
        distogram_logits = F.log_softmax(self.head(dcode), dim=-1)
        probs = torch.softmax(distogram_logits, dim=-1)
        bin_centers = self.bin_centers.to(device=probs.device, dtype=probs.dtype)
        dmap = (probs * bin_centers).sum(dim=-1)
        return distogram_logits, dmap


class InnerDistogramNative(nn.Module):
    """Tensor-native equivalent of decoder.InnerDistogram."""

    def __init__(self, config, bins: int = 16, heads: int = 8, start: float = 0.0, stop: float = 22.0):
        super().__init__()
        self.config = config
        self.bins = int(bins)
        self.heads = int(heads)
        self.code_proj = Linear(config.local_size, self.heads * self.bins, bias=False)
        self.gate_proj = Linear(config.local_size, self.heads * self.bins, bias=False)
        self.inner_weight = nn.Parameter(torch.zeros(self.heads, self.heads, self.bins))

        step = (stop - start) / self.bins
        bin_centers = torch.arange(self.bins, dtype=torch.get_default_dtype()) * step + step / 2.0
        self.register_buffer("bin_centers", bin_centers, persistent=False)

    def forward(self, features, resi, chain, batch, neighbours):
        del resi, chain, batch, neighbours
        n = features.shape[0]
        dcode = self.code_proj(features).reshape(n, self.heads, self.bins)
        dgate = gelu_salad(self.gate_proj(features).reshape(n, self.heads, self.bins))
        logits = torch.einsum("iax,jbx,abx->ijx", dcode, dgate, self.inner_weight)
        logits = 0.5 * (logits + logits.transpose(0, 1))
        logits = F.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        bin_centers = self.bin_centers.to(device=probs.device, dtype=probs.dtype)
        dmap = (probs * bin_centers).sum(dim=-1)
        return logits, dmap


class DecoderPairFeaturesNative(nn.Module):
    """Tensor-native equivalent of decoder.DecoderPairFeatures."""

    def __init__(self, c: Any):
        super().__init__()
        self.c = c
        self.relpos = sequence_relative_position(32, one_hot=True, pseudo_chains=True)
        self.p_relpos = Linear(RELPOS32_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_dmap = Linear(64, c.pair_size, bias=False, initializer="linear")
        self.p_dist = Linear(DISTANCE5_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_dir = Linear(DIRECTION5_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_rot = Linear(ROTATION_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_vec = Linear(PAIR_VECTOR5_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)
        self.mlp = MLP(c.pair_size, c.pair_size * 2, c.pair_size, activation=gelu_salad, final_init="linear")

    def forward(self, pos, dmap, neighbours, resi, chain, batch, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        neighbours = neighbours.long()
        gathered_mask = gather_nodes(mask, neighbours)
        pair_mask = mask[:, None] * gathered_mask
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)

        dmap = gather_dim1(dmap, neighbours)

        pair = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        dmap_feat = distance_rbf(dmap)
        pair = pair + self.p_dmap(torch.where(pair_mask[..., None].bool(), dmap_feat, 0.0))
        pair = pair + self.p_dist(distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair = pair + self.p_dir(direction_features(pos, neighbours))
        pair = pair + self.p_rot(position_rotation_features(pos, neighbours))
        pair = pair + self.p_vec(pair_vector_features(pos, neighbours))
        return self.mlp(self.ln(pair)), pair_mask


class DecoderUpdateNative(nn.Module):
    """Tensor-native equivalent of decoder.DecoderUpdate."""

    def __init__(self, config: Any, name: Optional[str] = "light_global_update"):
        super().__init__()
        del name
        self.config = config
        local_size = int(config.local_size)
        hidden = local_size * int(config.factor)
        self.pos_mlp = MLP(LOCAL_FRAME5_FEATURES, local_size * 2, local_size, activation=gelu_salad, final_init="zeros")
        self.local_update = Linear(local_size, hidden, initializer="linear", bias=False)
        self.local_gate = Linear(local_size, hidden, initializer="relu", bias=False)
        self.chain_gate = Linear(local_size, hidden, initializer="relu", bias=False)
        self.batch_gate = Linear(local_size, hidden, initializer="relu", bias=False)
        self.out = Linear(hidden, local_size, initializer="zeros")

    def forward(self, local, pos, chain, batch, mask):
        _, _, local_pos = local_positions(pos)
        local = local + self.pos_mlp(local_pos.reshape(local_pos.shape[0], -1))

        local_update = self.local_update(local)
        local_gate = gelu_salad(self.local_gate(local))
        chain_gate = gelu_salad(self.chain_gate(local))
        batch_gate = gelu_salad(self.batch_gate(local))
        hidden = index_mean(batch_gate * local_update, batch, mask[..., None])
        hidden = hidden + index_mean(chain_gate * local_update, chain, mask[..., None])
        hidden = hidden + local_gate * local_update
        return self.out(hidden)


class UpdatePositionsNative(nn.Module):
    """Tensor-native equivalent of decoder.UpdatePositions."""

    def __init__(self, local_size: int, atom_count: int = 5):
        super().__init__()
        self.atom_count = int(atom_count)
        self.proj = Linear(local_size, self.atom_count * 3, initializer="zeros", bias=False)

    def forward(self, pos, local_norm, *, scale: float = 10.0):
        a = int(pos.shape[-2])
        if a != self.atom_count:
            raise ValueError(f"UpdatePositionsNative initialized with A={self.atom_count}, but got A={a}.")

        rot, trans, local_pos = local_positions(pos)
        pos_update = float(scale) * self.proj(local_norm)
        local_pos = local_pos + pos_update.reshape(pos_update.shape[0], a, 3)
        return apply_frame(rot[:, None], trans[:, None], local_pos)


class DecoderBlockNative(nn.Module):
    """Tensor-native equivalent of the standard equivariant decoder block."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        c = config
        if c.distogram_block != "inner":
            raise ValueError("DecoderBlockNative v1 expects distogram_block='inner'")
        self.distogram = InnerDistogramNative(c)

        self.ln_attn_main = nn.LayerNorm(c.local_size)
        self.ln_update = nn.LayerNorm(c.local_size)
        self.ln_pos = nn.LayerNorm(c.local_size)
        self.ln_attn_dist = nn.LayerNorm(c.local_size)
        self.attn_main = SparseStructureAttentionNative(c)
        self.attn_dist = SparseStructureAttentionNative(c)
        self.update = DecoderUpdateNative(c)
        self.pos_update = UpdatePositionsNative(local_size=c.local_size, atom_count=5)
        self.pair_features_main = DecoderPairFeaturesNative(c)
        self.pair_features_dist = DecoderPairFeaturesNative(c)

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
        generator=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del generator
        c = self.config
        del pos_gt
        sup_neighbours = sup_neighbours.long()
        dist_full_logits, dmap = self.distogram(features, resi, chain, batch, None)
        distogram_logits = gather_dim1(dist_full_logits, sup_neighbours)
        dmap_neighbours = extract_dmap_neighbours_native(dmap.detach(), batch, mask, count=32)

        current_neighbours = extract_neighbours_native(
            pos,
            resi,
            chain,
            batch,
            mask,
            num_index=16,
            num_spatial=16,
            num_random=0,
        )

        pair, pair_mask = self.pair_features_main(pos, dmap, current_neighbours, resi, chain, batch, mask)
        features = features + self.attn_main(
            self.ln_attn_main(features),
            pos / c.sigma_data,
            pair,
            pair_mask,
            current_neighbours,
            resi,
            chain,
            batch,
            mask,
        )

        pair2, pair2_mask = self.pair_features_dist(pos, dmap, dmap_neighbours, resi, chain, batch, mask)
        features = features + self.attn_dist(
            self.ln_attn_dist(features),
            pos / c.sigma_data,
            pair2,
            pair2_mask,
            dmap_neighbours,
            resi,
            chain,
            batch,
            mask,
        )

        features = features + self.update(self.ln_update(features), pos, chain, batch, mask)
        pos = self.pos_update(pos, self.ln_pos(features), scale=float(c.sigma_data))
        return features, pos, distogram_logits


class DecoderStackNative(nn.Module):
    def __init__(self, config: Any, block_cls=DecoderBlockNative):
        super().__init__()
        self.blocks = nn.ModuleList([block_cls(config) for _ in range(config.depth)])

    def forward(self, local, pos, resi, chain, batch, mask, sup_neighbours, generator=None):
        del generator
        trajectory = []
        sup_distograms = []
        for block in self.blocks:
            local, pos, sup_dist = block(
                local,
                pos,
                resi,
                chain,
                batch,
                mask,
                sup_neighbours=sup_neighbours,
                generator=None,
            )
            trajectory.append(pos)
            sup_distograms.append(sup_dist)
        return local, pos, torch.stack(trajectory, dim=0), torch.stack(sup_distograms, dim=0)


class AADecoderBlockNative(nn.Module):
    """Tensor-native equivalent of decoder.AADecoderBlock."""

    def __init__(self, config: Any, name: Optional[str] = "aa_decoder_block"):
        super().__init__()
        del name
        self.config = config
        c = config
        self.pair_features = PairFeaturesNative(c)
        self.aa_linear = Linear(21, int(c.pair_size), bias=False)
        self.ln_aa = nn.LayerNorm(int(c.pair_size), elementwise_affine=True)
        self.pair_mlp = MLP(
            int(c.pair_size),
            2 * int(c.pair_size),
            int(c.pair_size),
            activation=gelu_salad,
            final_init="linear",
        )
        self.attn = SparseStructureAttentionNative(c)
        self.ln_attn_in = nn.LayerNorm(int(c.local_size), elementwise_affine=True)
        self.update = EncoderUpdateNative(c)
        self.ln_update_in = nn.LayerNorm(int(c.local_size), elementwise_affine=True)

    def forward(self, aa, features, pos, neighbours, resi, chain, batch, mask):
        pair, pair_mask = self.pair_features(pos, neighbours, resi, chain, batch, mask)
        aa_oh = F.one_hot(aa.long(), 21).to(dtype=features.dtype)
        aa_emb = self.ln_aa(self.aa_linear(aa_oh))
        pair = pair + gather_nodes(aa_emb, neighbours)
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
        features = features + self.update(self.ln_update_in(features), pos, chain, batch, mask)
        return features


class AADecoderStackNative(nn.Module):
    """Tensor-native equivalent of decoder.AADecoderStack."""

    def __init__(self, config: Any, depth: Optional[int] = None, name: Optional[str] = "aa_decoder_stack"):
        super().__init__()
        del name
        self.config = config
        self.depth = int(depth or 3)
        self.blocks = nn.ModuleList([AADecoderBlockNative(config) for _ in range(self.depth)])
        self.ln = nn.LayerNorm(int(config.local_size), elementwise_affine=True)

    def forward(self, aa, local, pos, neighbours, resi, chain, batch, mask):
        x = local
        for block in self.blocks:
            x = block(aa, x, pos, neighbours, resi, chain, batch, mask)
        return self.ln(x)


class AADecoderNative(nn.Module):
    """Tensor-native amino acid decoder with legacy-compatible parameter names."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.local_size)
        self.proj = nn.Linear(config.local_size, 20, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.stack = AADecoderStackNative(config, depth=config.aa_decoder_depth)

    def forward(self, aa, local, pos, resi, chain, batch, mask):
        neighbours = extract_spatial_neighbours_native(pos[:, -1], batch, mask, count=32)
        local = self.stack(aa, local, pos, neighbours, resi, chain, batch, mask)
        local = self.norm(local)
        logits = self.proj(local)
        return logits, local

    def decode_train(self, aa, local, pos, resi, chain, batch, mask):
        aa = torch.full_like(aa, 20)
        logits, features = self.forward(aa, local, pos, resi, chain, batch, mask)
        logits = F.log_softmax(logits, dim=-1)
        return logits, features, torch.ones_like(mask)


class GetAnglePositionsNative(nn.Module):
    """Tensor-native angle head; OpenFold atom14 placement stays at the boundary."""

    def __init__(self, local_dim):
        super().__init__()
        self.mlp = MLP(
            local_dim + LOCAL_FRAME5_FEATURES + 5 * 16 + 21,
            local_dim * 2,
            7 * 2,
            bias=False,
            activation=gelu_salad,
            final_init="linear",
        )

    def forward(self, aa_gt, local, pos):
        rot, trans, local_pos = local_positions(pos)
        features = [
            local,
            local_pos.reshape(local_pos.shape[0], -1),
            distance_rbf(torch.linalg.norm(local_pos, dim=-1), 0.0, 10.0, 16).reshape(local_pos.shape[0], -1),
            F.one_hot(aa_gt.long(), num_classes=21).to(local.dtype),
        ]
        raw_angles = self.mlp(torch.cat(features, dim=-1))
        raw_angles = raw_angles.reshape(-1, 7, 2)
        angles = raw_angles / torch.sqrt(torch.clamp((raw_angles**2).sum(dim=-1, keepdim=True), min=1e-6))

        backbone_frames = torch.zeros((*rot.shape[:-2], 4, 4), device=rot.device, dtype=rot.dtype)
        backbone_frames[..., :3, :3] = rot
        backbone_frames[..., :3, 3] = trans
        backbone_frames[..., 3, 3] = 1.0
        all_frames_to_global = torsion_angles_to_frames_native(aa_gt, backbone_frames, angles)
        angle_pos = frames_to_atom14_pos_native(aa_gt, all_frames_to_global)
        angle_pos = torch.cat((pos[..., :4, :], angle_pos[..., 4:, :]), dim=-2)
        return raw_angles, angles, angle_pos


class DecoderNative(nn.Module):
    """Backbone-only decoder container for the experimental native path."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        if config.equivariance not in (None, "equivariant"):
            raise ValueError("DecoderNative v1 supports only equivariant backbone configs")
        if config.atom37_parallel_mode != "none" or config.atom37_main_branch:
            raise ValueError("DecoderNative v1 supports backbone-only configs")

        self.decoder_stack = DecoderStackNative(config)
        self.aa_decoder = AADecoderNative(config)
        self.use_latent_diff = False
        self.prev_local_ln = nn.LayerNorm(config.local_size, elementwise_affine=True)
        local_input = LOCAL_FRAME5_FEATURES + 5 * 64 + config.latent_size + config.local_size
        self.local_mlp = MLP(
            local_input,
            size=4 * config.local_size,
            out_size=config.local_size,
            bias=False,
            activation=gelu_salad,
            final_init=init_linear(),
        )
        self.local_ln = nn.LayerNorm(config.local_size, elementwise_affine=True)
        self.angle_pos = GetAnglePositionsNative(local_dim=config.local_size)

    def init_prev(self, data):
        local_dtype = data["latent"].dtype if "latent" in data else data["pos"].dtype
        return {
            "pos": data["pos"],
            "local": torch.zeros(
                (data["pos"].shape[0], self.config.local_size),
                device=data["pos"].device,
                dtype=local_dtype,
            ),
        }


def prepare_decoder_features_native(decoder, data: dict[str, torch.Tensor], prev: dict[str, torch.Tensor]):
    """Tensor-native replacement for backbone Decoder.prepare_features."""
    resi = data["residue_index"]
    chain = data["chain_index"]
    batch = data["batch_index"]
    latent = data["latent"]
    mask = data["mask"]
    pos = prev["pos"]

    _, _, local_pos = local_positions(pos)
    local_features = [
        local_pos.reshape(local_pos.shape[0], -1),
        distance_rbf(torch.linalg.norm(local_pos, dim=-1), 0.0, 22.0).reshape(local_pos.shape[0], -1),
        latent,
    ]
    local_features.append(decoder.prev_local_ln(prev["local"]))

    local = decoder.local_mlp(torch.cat(local_features, dim=-1))
    local = decoder.local_ln(local)
    return local, pos, resi, chain, batch, mask
