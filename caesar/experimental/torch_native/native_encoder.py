"""Tensor-native backbone encoder modules."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from caesar.experimental.torch_native.native_geometric import (
    SparseStructureAttentionNative,
    direction_features,
    distance_features,
    gather_nodes,
    local_frame_features,
    pair_vector_features,
    position_rotation_features,
)
from caesar.experimental.torch_native.tensor_geometry import (
    distance_rbf,
    frames_from_ncac,
    sequence_relative_position,
)
from caesar.experimental.torch_native.basic import Linear, MLP, gelu_salad, init_linear, init_relu, init_zeros


RELPOS32_FEATURES = 66
RELPOS8_FEATURES = 18
DISTANCE5_FEATURES = 5 * 5 * 16
DIRECTION5_FEATURES = 5 * 3
ROTATION_FEATURES = 9
PAIR_VECTOR5_FEATURES = 2 * 5 * 3 * 2
LOCAL_FRAME5_FEATURES = 5 * 3


def extract_neighbours_native(
    pos: torch.Tensor,
    resi: torch.Tensor,
    chain: torch.Tensor,
    batch: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_index: int,
    num_spatial: int,
    num_random: int = 0,
) -> torch.Tensor:
    """Tensor-native deterministic neighbour extraction.

    This matches the deterministic branch used by the experimental backbone
    path. Random neighbours are intentionally rejected here so sampling stays
    outside native hot paths.
    """
    if int(num_random) != 0:
        raise ValueError("extract_neighbours_native expects num_random=0")
    if pos.ndim == 3:
        ca = pos[:, 1]
    elif pos.ndim == 2:
        ca = pos
    else:
        raise ValueError(f"Unsupported pos shape: {tuple(pos.shape)}")

    same_batch = batch[:, None] == batch[None, :]
    same_chain = chain[:, None] == chain[None, :]
    mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
    valid = same_batch & mask_bool[:, None] & mask_bool[None, :]

    within = (resi[:, None] - resi[None, :]).abs() < int(num_index)
    within = within & same_batch & same_chain

    distance = torch.linalg.norm(ca[:, None] - ca[None, :], dim=-1)
    inf = torch.full_like(distance, float("inf"))
    distance = torch.where(within, inf, distance)
    distance = torch.where(valid, distance, inf)

    if int(num_spatial) > 0:
        sorted_distance = torch.sort(distance, dim=-1).values
        cutoff = sorted_distance[:, : int(num_spatial)][:, -1]
        within = within | (distance < cutoff)

    total = int(num_index) + int(num_spatial)
    selected = torch.where(within, torch.full_like(distance, -10_000.0), inf)
    selected = torch.where(valid, selected, inf)
    return topk_neighbours(selected, valid, total)


def topk_neighbours(distance: torch.Tensor, mask: torch.Tensor, count: int) -> torch.Tensor:
    n = distance.shape[0]
    if count <= 0:
        return torch.empty((n, 0), device=distance.device, dtype=torch.long)
    distance = torch.where(mask.bool(), distance.float(), torch.full_like(distance.float(), float("inf")))
    col_index = torch.arange(n, device=distance.device, dtype=distance.dtype)[None, :]
    sortable = torch.where(torch.isfinite(distance), distance + col_index * 1e-6, distance)
    k = min(int(count), n)
    idx = torch.topk(sortable, k=k, dim=-1, largest=False, sorted=True).indices
    valid = torch.gather(distance, dim=1, index=idx) < float("inf")
    idx = torch.where(valid, idx, torch.full_like(idx, -1))
    if k < int(count):
        pad = torch.full((n, int(count) - k), -1, device=distance.device, dtype=torch.long)
        idx = torch.cat((idx, pad), dim=-1)
    return idx


class PairFeaturesNative(nn.Module):
    """Tensor-native equivalent of encoder.AADecoderPairFeatures."""

    def __init__(self, c: Any):
        super().__init__()
        self.c = c
        self.relpos = sequence_relative_position(32, one_hot=True, pseudo_chains=True)
        self.p_relpos = Linear(RELPOS32_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_dist = Linear(DISTANCE5_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_dir = Linear(DIRECTION5_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_rot = Linear(ROTATION_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_vec = Linear(PAIR_VECTOR5_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)

    def forward(self, pos, neighbours, resi, chain, batch, mask):
        neighbours = neighbours.long()
        mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
        neigh_mask = gather_nodes(mask_bool, neighbours)
        pair_mask = mask_bool[:, None] & neigh_mask & (neighbours != -1)

        pair = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        pair = pair + self.p_dist(distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair = pair + self.p_dir(direction_features(pos, neighbours))
        pair = pair + self.p_rot(position_rotation_features(pos, neighbours))
        pair = pair + self.p_vec(pair_vector_features(pos, neighbours))
        return self.ln(pair), pair_mask


class InitLocalFeaturesNative(nn.Module):
    """Tensor-native equivalent of encoder.InitLocalFeatures."""

    def __init__(self, c: Any):
        super().__init__()
        self.c = c
        self.relpos = sequence_relative_position(8, one_hot=True, pseudo_chains=True)
        self.p_relpos = Linear(RELPOS8_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_dist = Linear(DISTANCE5_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_dir = Linear(DIRECTION5_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_rot = Linear(ROTATION_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.p_vec = Linear(PAIR_VECTOR5_FEATURES, c.pair_size, bias=False, initializer="linear")
        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)
        self.mlp = MLP(c.pair_size, size=c.pair_size * 2, out_size=c.pair_size, activation=gelu_salad)
        self.to_local = Linear(c.pair_size, c.local_size, bias=False)

    def forward(self, pos, neighbours, resi, chain, batch, mask):
        neighbours = neighbours.long()
        mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
        neigh_mask = gather_nodes(mask_bool, neighbours)
        pair_mask_bool = mask_bool[:, None] & neigh_mask & (neighbours != -1)

        pair = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        pair = pair + self.p_dist(distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair = pair + self.p_dir(direction_features(pos, neighbours))
        pair = pair + self.p_rot(position_rotation_features(pos, neighbours))
        pair = pair + self.p_vec(pair_vector_features(pos, neighbours))
        pair = self.mlp(self.ln(pair))
        pair = torch.where(pair_mask_bool.unsqueeze(-1), pair, torch.zeros_like(pair))

        denom = pair_mask_bool.sum(dim=-1, keepdim=True).to(dtype=pair.dtype).clamp_min(1.0)
        local = pair.sum(dim=1) / denom
        return self.to_local(local)


class EncoderUpdateNative(nn.Module):
    """Tensor-native equivalent of encoder.EncoderUpdate."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        c = config
        self.localpos_mlp = MLP(
            LOCAL_FRAME5_FEATURES,
            size=c.local_size * 2,
            out_size=c.local_size,
            activation=gelu_salad,
            final_init=init_zeros(),
        )
        hidden = c.local_size * c.factor
        self.update_proj = Linear(c.local_size, hidden, initializer=init_linear(), bias=False)
        self.gate_proj = Linear(c.local_size, hidden, initializer=init_relu(), bias=False)
        self.out_proj = Linear(hidden, c.local_size, initializer=init_zeros(), bias=True)

    def forward(self, local, pos, chain=None, batch=None, mask=None):
        del chain, batch, mask
        local = local + self.localpos_mlp(local_frame_features(pos))
        upd = self.update_proj(local)
        gate = gelu_salad(self.gate_proj(local))
        return self.out_proj(gate * upd)


class EncoderBlockNative(nn.Module):
    """Tensor-native backbone encoder block."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        c = config
        self.pair_features = PairFeaturesNative(c)
        self.pair_mlp = MLP(c.pair_size, 2 * c.pair_size, c.pair_size, activation=gelu_salad, final_init=init_linear())
        self.attn = SparseStructureAttentionNative(c)
        self.update = EncoderUpdateNative(c)
        self.ln_attn = nn.LayerNorm(c.local_size)
        self.ln_update = nn.LayerNorm(c.local_size)

    def forward(self, features, pos, resi, chain, batch, mask, *, generator=None):
        del generator
        c = self.config
        neighbours = extract_neighbours_native(
            pos,
            resi,
            chain,
            batch,
            mask,
            num_index=16,
            num_spatial=16,
            num_random=0,
        )
        pair, pair_mask = self.pair_features(pos, neighbours, resi, chain, batch, mask)
        pair = self.pair_mlp(pair)
        features = features + self.attn(
            self.ln_attn(features),
            pos,
            pair,
            pair_mask,
            neighbours,
            resi,
            chain,
            batch,
            mask,
        )
        features = features + self.update(self.ln_update(features), pos, chain, batch, mask)
        return features


class EncoderStackNative(nn.Module):
    def __init__(self, config: Any, depth=None):
        super().__init__()
        self.config = config
        self.depth = depth or 3
        self.blocks = nn.ModuleList([EncoderBlockNative(config) for _ in range(self.depth)])
        self.ln_final = nn.LayerNorm(config.local_size)

    def forward(self, local, pos, resi, chain, batch, mask, *, generator=None):
        del generator
        for block in self.blocks:
            local = block(local, pos, resi, chain, batch, mask, generator=None)
        return self.ln_final(local)


class EncoderNative(nn.Module):
    """Backbone-only tensor-native encoder with legacy-compatible state names."""

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        c = config
        if c.atom37_parallel_mode != "none" or c.atom37_main_branch:
            raise ValueError("EncoderNative supports backbone-only configs")
        self.init_local = InitLocalFeaturesNative(c)
        self.stack = EncoderStackNative(c, depth=c.encoder_depth)
        self.local_mlp = MLP(
            c.local_size,
            size=4 * c.local_size,
            out_size=c.local_size,
            activation=gelu_salad,
            bias=False,
            final_init=init_linear(),
        )
        self.local_ln = nn.LayerNorm(c.local_size)
        self.ln_final = nn.LayerNorm(c.local_size)
        self.to_latent = Linear(c.local_size, c.latent_size, bias=False, initializer=init_linear())

    def prepare_features(self, data, *, generator=None, trace=None):
        del generator
        pos = data["pos_input"]

        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask_bool"]
        if mask.dtype != torch.bool:
            mask = mask != 0

        neighbours = extract_neighbours_native(
            pos,
            resi,
            chain,
            batch,
            mask,
            num_index=5,
            num_spatial=5,
            num_random=0,
        )
        local_features = self.init_local(pos, neighbours, resi, chain, batch, mask)
        local = self.local_mlp(local_features)
        local = self.local_ln(local)

        if trace is not None:
            trace["enc/neighbours0"] = neighbours.detach()
            trace["enc/local_features0"] = local_features.detach()
            trace["enc/local0"] = local.detach()
        return local, pos, resi, chain, batch, mask

    def forward(self, data, *, generator=None, trace=None):
        del generator
        local, pos, resi, chain, batch, mask = self.prepare_features(data, generator=None, trace=trace)
        local = self.stack(local, pos, resi, chain, batch, mask, generator=None)
        local = self.ln_final(local)
        return self.to_latent(local)
