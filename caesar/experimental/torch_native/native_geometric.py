"""Torch-native geometric feature wrappers.

These functions are the first replacement surface for pieces currently living
in ``caesar.modules.geometric``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from caesar.experimental.torch_native.basic import Linear, MLP, gelu_salad, init_linear
from caesar.experimental.torch_native.tensor_geometry import (
    apply_frame,
    distance_rbf,
    direction_features,
    distance_features,
    frame_tensor_from_rot_trans,
    frames_from_ncac,
    gather_nodes,
    invert_apply_frame,
    local_frame_features,
    paired_distance_features,
    paired_rotation_features,
    pair_vector_features,
    position_rotation_features,
    rot_trans_from_frame_tensor,
    sequence_relative_position,
)


def _config_int(config, name: str, default: int) -> int:
    value = getattr(config, name, None)
    return int(default if value is None else value)


class LinearToPointsNative(nn.Module):
    """Parameter-compatible point projection helper with tensor-only forward."""

    def __init__(self, size: int = 8, init: str = "linear", in_features: int = 32):
        super().__init__()
        self.size = int(size)
        self.init = init
        self.in_features = int(in_features)
        self.proj = Linear(self.in_features, self.size * 3, bias=False, initializer=self.init)

    def forward(self, data: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        rot, trans = rot_trans_from_frame_tensor(frames)
        raw = self.proj(data).reshape(*data.shape[:-1], self.size, 3)
        return apply_frame(rot[..., None, :, :], trans[..., None, :], raw)


class SparseInvariantPointAttentionNative(nn.Module):
    """Tensor-native equivalent of ``modules.geometric.SparseInvariantPointAttention``."""

    def __init__(
        self,
        size: int = 32,
        heads: int = 4,
        query_points: int = 8,
        value_points: int = 8,
        final_init: str = "zeros",
        normalize: bool = False,
        local_size: int = 32,
        pair_size: int = 16,
    ):
        super().__init__()
        self.size = int(size)
        self.heads = int(heads)
        self.query_points = int(query_points)
        self.value_points = int(value_points)
        self.final_init = final_init
        self.normalize = bool(normalize)
        self.local_size = int(local_size)
        self.pair_size = int(pair_size)

        self.local_norm = nn.LayerNorm(self.local_size, elementwise_affine=True) if self.normalize else nn.Identity()
        self.ln_q = nn.LayerNorm(self.size, elementwise_affine=True)
        self.ln_k = nn.LayerNorm(self.size, elementwise_affine=True)
        self.qkv = Linear(self.local_size, self.heads * 3 * self.size, bias=False, name="qkv")
        self.qkv_global = LinearToPointsNative(
            self.heads * (2 * self.query_points + self.value_points),
            in_features=self.local_size,
        )
        self.bias = Linear(self.pair_size, self.heads, bias=False, name="bias")

        gamma0 = torch.log(torch.exp(torch.tensor(1.0)) - torch.tensor(1.0))
        self.gamma = nn.Parameter(gamma0.repeat(self.heads))
        out_in = self.heads * (self.pair_size + self.size + self.value_points * 3 + self.value_points)
        self.project_out = Linear(out_in, self.local_size, initializer=self.final_init, name="project_out")

    def forward(self, local, pair, frames, neighbours, mask):
        local = self.local_norm(local)

        rot, trans = rot_trans_from_frame_tensor(frames)
        qkv = self.qkv(local).view(*local.shape[:-1], self.heads, 3 * self.size)
        q, k, v = torch.split(qkv, self.size, dim=-1)
        q = self.ln_q(q)
        k = self.ln_k(k)

        raw_points = self.qkv_global.proj(local)
        raw_points = raw_points.reshape(*raw_points.shape[:-1], self.heads, -1, 3)
        qkv_g = apply_frame(rot[:, None, None], trans[:, None, None], raw_points)
        q_g, k_g, v_g = torch.split(
            qkv_g,
            [self.query_points, self.query_points, self.value_points],
            dim=-2,
        )

        bias = self.bias(pair)

        w_c = (2.0 / (9.0 * self.query_points)) ** 0.5
        w_l = (1.0 / 3.0) ** 0.5
        dfactor = F.softplus(self.gamma.view(1, 1, self.heads)) * w_c / 2.0

        neighbours = neighbours.long()
        k_neigh = gather_nodes(k, neighbours)
        v_neigh = gather_nodes(v, neighbours)
        k_g_neigh = gather_nodes(k_g, neighbours)
        v_g_neigh = gather_nodes(v_g, neighbours)

        dist = dfactor * (q_g[:, None] - k_g_neigh).pow(2).sum(dim=(-1, -2))
        dot = (1.0 / self.size) ** 0.5 * (q.unsqueeze(1) * k_neigh).sum(dim=-1)
        attn_logits = w_l * (dot + bias - dist)

        pair_mask = mask * (neighbours != -1)
        attn_logits = torch.where(
            pair_mask[..., None].bool(),
            attn_logits,
            torch.full_like(attn_logits, -1e9),
        )

        attn = torch.softmax(attn_logits, dim=1)
        attn = torch.where(pair_mask[..., None].bool(), attn, torch.zeros_like(attn))

        out_pair = torch.einsum("nkh,nkc->nhc", attn, pair)
        out_scalar = torch.einsum("nkh,nkhc->nhc", attn, v_neigh)
        out_point_global = torch.einsum("nkh,nkhvd->nhvd", attn, v_g_neigh)
        out_point = invert_apply_frame(rot[:, None, None], trans[:, None, None], out_point_global)
        out_norm = torch.linalg.norm(out_point, dim=-1)

        concat = torch.cat(
            [
                out_pair.reshape(*out_pair.shape[:-2], -1),
                out_scalar.reshape(*out_scalar.shape[:-2], -1),
                out_point.reshape(*out_point.shape[:-3], -1),
                out_norm.reshape(*out_point.shape[:-3], -1),
            ],
            dim=-1,
        )

        out = self.project_out(concat)
        return out


class SparseStructureAttentionNative(nn.Module):
    """Tensor-native replacement surface for ``geometric.SparseStructureAttention``."""

    def __init__(self, config, normalize: bool = True):
        super().__init__()
        self.config = config
        self.normalize = bool(normalize)
        final_init = config.update_init if getattr(config, "update_init", None) else "zeros"
        attn_cls = SparseInvariantMultiQueryAttentionNative if getattr(config, "multi_query", False) else SparseInvariantPointAttentionNative
        if attn_cls is SparseInvariantMultiQueryAttentionNative:
            self.attn = attn_cls(
                config=config,
                heads=config.heads,
                size=config.key_size,
                final_init=final_init,
                normalize=self.normalize,
            )
        else:
            self.attn = attn_cls(
                heads=config.heads,
                size=config.key_size,
                final_init=final_init,
                normalize=self.normalize,
                local_size=_config_int(config, "local_size", 32),
                pair_size=_config_int(config, "pair_size", 16),
            )

    def forward(
        self,
        local: torch.Tensor,
        pos: torch.Tensor,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        neighbours: torch.Tensor,
        resi: torch.Tensor,
        chain: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        del resi, chain, batch, mask
        rot, trans = frames_from_ncac(pos)
        frames = frame_tensor_from_rot_trans(rot, trans)
        return self.attn(local, pair, frames, neighbours, pair_mask)


class SparseStructureMessageNative(nn.Module):
    """Tensor-native equivalent of ``modules.geometric.SparseStructureMessage``."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pair_mlp = MLP(
            int(config.pair_size),
            2 * int(config.pair_size),
            out_size=None,
            depth=3,
            activation=gelu_salad,
            final_init="zeros",
        )

    def forward(self, local, pos, pair, pair_mask, neighbours, resi, chain, batch, mask):
        del local, pos, neighbours, resi, chain, batch, mask
        pair = self.pair_mlp(pair)
        local_update = torch.where(pair_mask[..., None].bool(), pair, torch.zeros_like(pair)).sum(dim=1)
        return local_update / pair.shape[1]


class SparseInvariantMultiQueryAttentionNative(nn.Module):
    """Tensor-native equivalent of ``modules.geometric.SparseInvariantMultiQueryAttention``."""

    def __init__(
        self,
        *,
        config,
        size: int = 32,
        heads: int = 4,
        query_points: int = 8,
        value_points: int = 8,
        final_init: str = "zeros",
        normalize: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        del name
        self.config = config
        self.size = int(size)
        self.heads = int(heads)
        self.query_points = int(query_points)
        self.value_points = int(value_points)
        self.final_init = final_init
        self.normalize = bool(normalize)

        local_size = _config_int(config, "local_size", 32)
        pair_size = _config_int(config, "pair_size", 16)
        self.q_proj = Linear(local_size, self.heads * self.size, bias=True, initializer="linear")
        self.k_proj = Linear(local_size, self.size, bias=True, initializer="linear")
        self.v_proj = Linear(local_size, self.size, bias=True, initializer="linear")
        self.qp_proj = Linear(local_size, self.heads * self.query_points * 3, bias=True, initializer="linear")
        self.kp_proj = Linear(local_size, self.query_points * 3, bias=True, initializer="linear")
        self.vp_proj = Linear(local_size, self.query_points * 3, bias=True, initializer="linear")
        self.pair_bias = Linear(pair_size, self.heads, bias=False, initializer="linear")
        gamma0 = torch.log(torch.exp(torch.tensor(1.0)) - torch.tensor(1.0))
        self.gamma = nn.Parameter(gamma0.repeat(self.heads))
        out_in = self.heads * (self.size + pair_size + self.query_points * 3)
        self.out_proj = Linear(out_in, local_size, bias=False, initializer=final_init)

    def _component(self, x: torch.Tensor, proj: Linear, heads: int) -> torch.Tensor:
        y = proj(x)
        return y.view(y.shape[0], heads, self.size)

    def _points_global(self, x: torch.Tensor, proj: Linear, frames: torch.Tensor, heads: int) -> torch.Tensor:
        rot, trans = rot_trans_from_frame_tensor(frames)
        y = proj(x).view(x.shape[0], heads, self.query_points, 3)
        return apply_frame(rot[:, None, None], trans[:, None, None], y)

    def _to_local(self, x_global: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        rot, trans = rot_trans_from_frame_tensor(frames)
        return invert_apply_frame(rot[:, None, None], trans[:, None, None], x_global)

    def forward(self, local, pair, frames, neighbours, mask):
        neighbours = neighbours.long()
        mask_bool = mask.bool()

        n, _ = neighbours.shape
        h = self.heads
        c = self.size
        q_count = self.query_points

        q = self._component(local, self.q_proj, heads=h)
        k = self._component(local, self.k_proj, heads=1)
        v = self._component(local, self.v_proj, heads=1)
        q = F.layer_norm(q, (c,), weight=None, bias=None)
        k = F.layer_norm(k, (c,), weight=None, bias=None)

        qp = self._points_global(local, self.qp_proj, frames, heads=h)
        kp = self._points_global(local, self.kp_proj, frames, heads=1)
        vp = self._points_global(local, self.vp_proj, frames, heads=1)

        k_g = gather_nodes(k, neighbours).expand(-1, -1, h, -1)
        v_g = gather_nodes(v, neighbours).expand(-1, -1, h, -1)
        kp_g = gather_nodes(kp, neighbours).expand(-1, -1, h, -1, -1)
        vp_g = gather_nodes(vp, neighbours).expand(-1, -1, h, -1, -1)

        w_c = (2.0 / (9.0 * q_count)) ** 0.5
        w_l = (1.0 / 3.0) ** 0.5
        scale = F.softplus(self.gamma.view(1, 1, h)) * w_c / 2.0

        dot = (q.unsqueeze(1) * k_g).sum(dim=-1) / (c**0.5)
        dist = ((qp.unsqueeze(1) - kp_g) ** 2).sum(dim=(-1, -2))
        bias = self.pair_bias(pair)
        attn_logits = w_l * (dot - scale * dist + bias)
        attn_logits = torch.where(mask_bool.unsqueeze(-1), attn_logits, torch.full_like(attn_logits, -1e9))
        attn = torch.softmax(attn_logits, dim=1)
        attn = torch.where(mask_bool.unsqueeze(-1), attn, torch.zeros_like(attn))

        local_update = torch.einsum("nkh,nkhc->nhc", attn, v_g).reshape(n, -1)
        pair_update = torch.einsum("nkh,nkc->nhc", attn, pair).reshape(n, -1)
        point_global = torch.einsum("nkh,nkhqd->nhqd", attn, vp_g)
        point_local = self._to_local(point_global, frames).reshape(n, -1)
        return self.out_proj(torch.cat((local_update, pair_update, point_local), dim=-1))


class SemiEquivariantSparseStructureAttentionNative(nn.Module):
    """Tensor-native equivalent of ``modules.geometric.SemiEquivariantSparseStructureAttention``."""

    def __init__(self, config, normalize: bool = False):
        super().__init__()
        self.normalize = bool(normalize)
        self.config = config
        final_init = config.update_init if getattr(config, "update_init", None) else "zeros"
        self.attn = SparseSemiEquivariantPointAttentionNative(
            config=config,
            heads=config.heads,
            size=config.key_size,
            final_init=final_init,
            normalize=self.normalize,
        )

    def forward(self, local, pos, pair, pair_mask, neighbours, resi, chain, batch, mask):
        del resi, chain, batch, mask
        return self.attn(local, pair, pos, neighbours, pair_mask)


class SparseAttentionNative(nn.Module):
    """Tensor-native equivalent of ``modules.geometric.SparseAttention``."""

    def __init__(
        self,
        size: int = 32,
        heads: int = 4,
        final_init: str = "zeros",
        normalize: bool = False,
        local_size: int = 32,
        pair_size: int = 16,
    ):
        super().__init__()
        self.size = int(size)
        self.heads = int(heads)
        self.final_init = final_init
        self.normalize = bool(normalize)
        self.local_size = int(local_size)
        self.pair_size = int(pair_size)

        self.qkv = Linear(self.local_size, self.heads * 3 * self.size, bias=False, name="qkv")
        self.ln_q = nn.LayerNorm(self.size, elementwise_affine=True)
        self.ln_k = nn.LayerNorm(self.size, elementwise_affine=True)
        self.bias = Linear(self.pair_size, self.heads, bias=False, name="bias")
        self.local_norm = nn.LayerNorm(self.local_size, elementwise_affine=True) if self.normalize else nn.Identity()
        out_in = self.heads * (self.pair_size + self.size)
        self.project_out = Linear(out_in, self.local_size, initializer=self.final_init, name="project_out")

    def forward(self, local, pair, neighbours, mask):
        local = self.local_norm(local)

        qkv = self.qkv(local).view(*local.shape[:-1], self.heads, 3 * self.size)
        q, k, v = torch.split(qkv, self.size, dim=-1)
        q = self.ln_q(q)
        k = self.ln_k(k)

        bias = self.bias(pair)

        neighbours = neighbours.long()
        k_neigh = gather_nodes(k, neighbours)
        v_neigh = gather_nodes(v, neighbours)
        pair_mask = mask * (neighbours != -1)

        dot = (1.0 / self.size) ** 0.5 * (q.unsqueeze(1) * k_neigh).sum(dim=-1)
        attn_logits = (1.0 / 2.0) ** 0.5 * (dot + bias)
        attn_logits = torch.where(
            pair_mask[..., None].bool(),
            attn_logits,
            torch.full_like(attn_logits, -1e9),
        )

        attn = torch.softmax(attn_logits, dim=1)
        attn = torch.where(pair_mask[..., None].bool(), attn, torch.zeros_like(attn))

        out_pair = torch.einsum("nkh,nkc->nhc", attn, pair)
        out_scalar = torch.einsum("nkh,nkhc->nhc", attn, v_neigh)
        concat = torch.cat(
            (
                out_pair.reshape(*out_pair.shape[:-2], -1),
                out_scalar.reshape(*out_scalar.shape[:-2], -1),
            ),
            dim=-1,
        )

        out = self.project_out(concat)
        return out


class DenseNonEquivariantPointAttentionNative(nn.Module):
    """Tensor-native equivalent of ``modules.geometric.DenseNonEquivariantPointAttention``."""

    def __init__(
        self,
        size: int = 32,
        heads: int = 4,
        query_points: int = 8,
        value_points: int = 8,
        final_init: str = "zeros",
        normalize: bool = False,
        use_pair: bool = False,
        local_size: int = 32,
        name: Optional[str] = "ada_point_attention",
    ):
        super().__init__()
        del name
        self.size = int(size)
        self.heads = int(heads)
        self.query_points = int(query_points)
        self.value_points = int(value_points)
        self.final_init = final_init
        self.use_pair = bool(use_pair)
        self.normalize = bool(normalize)
        self.local_size = int(local_size)

        h = self.heads
        c = self.size
        q = self.query_points
        v = self.value_points

        self.local_norm = nn.LayerNorm(self.local_size, elementwise_affine=True) if self.normalize else nn.Identity()
        self.q_proj = Linear(self.local_size, c * h, initializer="linear")
        self.k_proj = Linear(self.local_size, c * h, initializer="linear")
        self.v_proj = Linear(self.local_size, c * h, initializer="linear")
        self.ln_q = nn.LayerNorm(c, elementwise_affine=True)
        self.ln_k = nn.LayerNorm(c, elementwise_affine=True)
        self.qp_proj = Linear(self.local_size, q * h * 3, initializer="zeros")
        self.kp_proj = Linear(self.local_size, q * h * 3, initializer="zeros")
        self.vp_proj = Linear(self.local_size, v * h * 3, initializer="zeros")
        self.relpos = sequence_relative_position(32, one_hot=True)
        self.pair_lin = Linear(16 + 3 * q * h, c + 3 * v, bias=False, initializer="linear")
        self.bias_lin = Linear(16 + 3 * q * h, h, initializer="linear")

        emb = torch.empty(66, c + 3 * v)
        init_linear()(emb)
        self.resi_embedding = nn.Parameter(emb)
        gamma0 = torch.log(torch.exp(torch.tensor(1.0)) - torch.tensor(1.0))
        self.gamma = nn.Parameter(gamma0.repeat(h))
        self.out = Linear(h * (c + 3 * v), self.local_size, bias=False, initializer="zeros")

    def forward(self, local, pos, resi, chain, batch, mask):
        local = self.local_norm(local)

        h = self.heads
        c = self.size
        q = self.query_points
        v_count = self.value_points

        same_batch = batch[:, None].eq(batch[None, :])
        mask_bool = mask.bool()
        pair_mask = mask_bool[:, None] & mask_bool[None, :] & same_batch

        def attention_component(x: torch.Tensor, proj: Linear) -> torch.Tensor:
            y = proj(x)
            return y.view(*y.shape[:-1], h, c)

        def attention_point(x: torch.Tensor, proj: Linear, point_count: int) -> torch.Tensor:
            y = proj(x).view(*x.shape[:-1], h, point_count, 3)
            return y + pos[:, 1][:, None, None]

        query = self.ln_q(attention_component(local, self.q_proj))
        key = self.ln_k(attention_component(local, self.k_proj))
        value = attention_component(local, self.v_proj)
        query_points = attention_point(local, self.qp_proj, q)
        key_points = attention_point(local, self.kp_proj, q)
        value_points = attention_point(local, self.vp_proj, v_count)
        value = torch.cat((value, value_points.reshape(*value.shape[:-1], -1)), dim=-1)

        inner_product_attention = torch.einsum(
            "ihc,jhc->ijh",
            query * ((1.0 / query.shape[-1]) ** 0.5),
            key,
        )
        point_attention = -((query_points[:, None] - key_points[None, :]) ** 2).sum(dim=(-1, -2))

        resi_dist = (resi[:, None] - resi[None, :]).clamp(-32, 32)
        other_chain = chain[:, None].ne(chain[None, :])
        ca = 10.0 * pos[:, 1]
        dist = torch.linalg.norm(ca[:, None] - ca[None, :], dim=-1)
        dist = distance_rbf(dist, bins=16)
        rel = (query_points[:, None] - key_points[None, :]).reshape(dist.shape[0], dist.shape[1], -1)

        rdist = (resi_dist + 32).long()
        rdist = torch.where(other_chain, torch.full_like(rdist, 65), rdist)
        rdist_emb = self.resi_embedding[rdist]
        pair_input = torch.cat((rel, dist), dim=-1)
        pair = rdist_emb + self.pair_lin(pair_input)
        bias = self.bias_lin(pair_input)
        pair = pair[:, :, None, :] + value[None, :]

        w_c = (2.0 / (9.0 * q)) ** 0.5
        point_scale = F.softplus(self.gamma.view(1, 1, h)) * w_c / 2.0
        attn = (inner_product_attention + point_scale * point_attention + bias) * ((1.0 / 3.0) ** 0.5)
        attn = torch.where(pair_mask[..., None], attn, torch.full_like(attn, -1e9))
        attn = torch.softmax(attn, dim=1)
        attn = torch.where(pair_mask[..., None], attn, torch.zeros_like(attn))

        result = torch.einsum("ijh,ijhc->ihc", attn, pair).reshape(local.shape[0], -1)
        out = self.out(result)
        return out


class SparseSemiEquivariantPointAttentionNative(nn.Module):
    """Tensor-native equivalent of ``modules.geometric.SparseSemiEquivariantPointAttention``."""

    def __init__(
        self,
        *,
        config,
        size: int = 32,
        heads: int = 4,
        query_points: int = 8,
        value_points: int = 8,
        final_init: str = "zeros",
        normalize: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__()
        del name
        self.config = config
        self.size = int(size)
        self.heads = int(heads)
        self.query_points = int(query_points)
        self.value_points = int(value_points)
        self.final_init = final_init
        self.normalize = bool(normalize)

        local_size = _config_int(config, "local_size", 32)
        self.local_norm = nn.LayerNorm(local_size, elementwise_affine=True) if self.normalize else nn.Identity()
        self.qkv = Linear(local_size, self.heads * 3 * self.size, bias=False, initializer="linear", name="qkv")
        self.ln_q = nn.LayerNorm(self.size, elementwise_affine=True)
        self.ln_k = nn.LayerNorm(self.size, elementwise_affine=True)
        self.qkv_g = Linear(
            local_size,
            self.heads * (2 * self.query_points + self.value_points) * 3,
            bias=True,
            initializer="linear",
        )
        pair_size = _config_int(config, "pair_size", 16)
        self.bias = Linear(pair_size, self.heads, bias=False, initializer="linear", name="bias")
        gamma0 = torch.log(torch.exp(torch.tensor(1.0)) - torch.tensor(1.0))
        self.gamma = nn.Parameter(gamma0.repeat(self.heads))
        out_in = self.heads * (
            pair_size + self.size + self.value_points * 3 + self.value_points
        )
        self.project_out = Linear(out_in, local_size, bias=True, initializer=final_init, name="project_out")

    def forward(
        self,
        local: torch.Tensor,
        pair: torch.Tensor,
        pos: torch.Tensor,
        neighbours: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        local = self.local_norm(local)

        h = self.heads
        c = self.size
        q = self.query_points
        v_count = self.value_points

        qkv = self.qkv(local).view(*local.shape[:-1], h, 3 * c)
        query, key, value = torch.split(qkv, c, dim=-1)
        query = self.ln_q(query)
        key = self.ln_k(key)

        base = pos[:, 1, :]
        qkv_g = self.qkv_g(local).view(local.shape[0], -1, 3)
        qkv_g = qkv_g + base[:, None, :]
        qkv_g = qkv_g.view(qkv_g.shape[0], h, (2 * q + v_count), 3)
        q_g, k_g, v_g = torch.split(qkv_g, [q, q, v_count], dim=-2)

        bias = self.bias(pair)

        neighbours = neighbours.long()
        pair_mask = mask * (neighbours != -1)
        pair_mask_bool = pair_mask.bool()

        k_n = gather_nodes(key, neighbours)
        v_n = gather_nodes(value, neighbours)
        k_gn = gather_nodes(k_g, neighbours)
        v_gn = gather_nodes(v_g, neighbours)

        w_c = (2.0 / (9.0 * q)) ** 0.5
        w_l = (1.0 / 3.0) ** 0.5
        dfactor = F.softplus(self.gamma.view(1, 1, h)) * w_c / 2.0
        dist = dfactor * (q_g.unsqueeze(1) - k_gn).pow(2).sum(dim=(-1, -2))
        dot = (1.0 / c) ** 0.5 * (query.unsqueeze(1) * k_n).sum(dim=-1)
        attn_logits = w_l * (dot + bias - dist)
        attn_logits = torch.where(
            pair_mask_bool.unsqueeze(-1),
            attn_logits,
            torch.full_like(attn_logits, -1e9),
        )

        attn = torch.softmax(attn_logits, dim=1)
        attn = torch.where(pair_mask_bool.unsqueeze(-1), attn, torch.zeros_like(attn))
        out_pair = torch.einsum("nkh,nkc->nhc", attn, pair)
        out_scalar = torch.einsum("nkh,nkhc->nhc", attn, v_n)
        out_point = torch.einsum("nkh,nkhvd->nhvd", attn, v_gn)
        out_point = out_point - base[:, None, None, :]
        out_norm = torch.linalg.norm(out_point, dim=-1)

        out = torch.cat(
            (
                out_pair.reshape(out_pair.shape[0], -1),
                out_scalar.reshape(out_scalar.shape[0], -1),
                out_point.reshape(out_point.shape[0], -1),
                out_norm.reshape(out_norm.shape[0], -1),
            ),
            dim=-1,
        )
        out = self.project_out(out)
        return out


__all__ = [
    "DenseNonEquivariantPointAttentionNative",
    "LinearToPointsNative",
    "SemiEquivariantSparseStructureAttentionNative",
    "SparseAttentionNative",
    "SparseInvariantMultiQueryAttentionNative",
    "SparseInvariantPointAttentionNative",
    "SparseStructureMessageNative",
    "SparseSemiEquivariantPointAttentionNative",
    "SparseStructureAttentionNative",
    "direction_features",
    "distance_features",
    "gather_nodes",
    "local_frame_features",
    "paired_distance_features",
    "paired_rotation_features",
    "pair_vector_features",
    "position_rotation_features",
]
