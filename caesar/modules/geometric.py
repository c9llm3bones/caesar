from typing import Optional
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from caesar.utils.geometry import Vec3Array
from caesar.modules.basic import Linear, MLP, init_linear
from caesar.modules.utils.geometry import (
    distance_rbf, extract_aa_frames, 
    sequence_relative_position, 
    index_mean
)
from caesar.utils.all_atom_multimer import make_transform_from_reference

from caesar.utils.geometry import Vec3Array, Rigid3Array

def distance_features(pos, neighbours=None, d_min=0.0, d_max=22.0, num_rbf=16):
    """Compute residue pair distance features.
    
    Args:
        pos: residue atom positions of shape (N, M, 3).
        neighbours: optional neighbour array.
        d_min: minimum distance for binning. Default: 0.0.
        d_max: maximum distance for binning. Default: 22.0.
        num_rbf: number of radial basis functions.
    Returns:
        5 * 5 * 16 radial basis functions per residue pair.
        If a neighbour array was provided, the output has shape (N, K, 400)
        otherwise it has shape (N, N, 400).
    """
    if neighbours is None:
        dist = (pos[:, None, :5, None] - pos[None, :, None, :5]).norm()
    else:
        dist = (pos[:, None, :5, None] - pos[neighbours, None, :5]).norm()
    dist = dist.reshape(*dist.shape[:-2], -1)
    result = distance_rbf(dist, d_min, d_max, num_rbf).reshape(*dist.shape[:-1], -1)
    return result

def direction_features(pos, neighbours=None, d_min=0.0, d_max=22.0, num_rbf=16):
    """Computes residue relative direction features.
    
    Args:
        pos: residue atom positions of shape (N, atoms, 3).
        neighbours: optional neighbour array.
    Returns:
        Direction vectors from the center of each residue frame to
        all atoms in each of its neighbours (N, K, atoms, 3).
    """
    frames, _ = extract_aa_frames(pos)
    if neighbours is None:
        local_pos = frames[:, None, None].apply_inverse_to_point(pos[None, :])
        dirs = local_pos.normalized().to_tensor()
    else:
        local_pos = frames[:, None, None].apply_inverse_to_point(pos[neighbours])
        dirs = local_pos.normalized().to_tensor()
    result = dirs.reshape(*dirs.shape[:2], -1)
    return result

def type_position_features(local, pos, batch, mask, size=32, scale=10.0,
                           learned_offset=False, neighbours=None):
    frames, _ = extract_aa_frames(pos)
    pair_mask = (neighbours != -1) * mask[:, None]
    def type_features(type_pos):
        type_pos = Vec3Array.from_array(type_pos)
        if len(type_pos.shape) == 3:
            type_pos = frames[:, None, None].apply_inverse_to_point(type_pos)
        else:
            type_pos = frames[:, None].apply_inverse_to_point(type_pos)
        type_dir = type_pos.normalized().to_tensor().reshape(local.shape[0], size * 3)
        type_dist = distance_rbf(type_pos.norm(), 0.0, 22.0, 10).reshape(type_pos.shape[0], -1)
        type_pos = type_pos.to_tensor().reshape(type_pos.shape[0], size * 3) / scale
        return torch.concatenate((type_dir, type_dist, type_pos), axis=-1)
    # compute type weight
    base_type_weight = Linear(size, bias=False, initializer="linear")(local)
    base_type_weight = torch.nn.functional.gelu(base_type_weight)
    # local type positions
    type_weight = base_type_weight[neighbours]
    type_weight = torch.where((neighbours != -1)[..., None], type_weight, 0)
    pos = pos.to_tensor()
    entry_pos = pos[:, 1, None]
    if learned_offset:
        entry_pos = LinearToPoints(size, init="zeros")(local, frames)
    local_type_pos = entry_pos[neighbours] * type_weight[..., None]
    local_type_pos = torch.where(pair_mask[..., None, None], local_type_pos, 0)
    local_type_pos = local_type_pos.sum(axis=1) / torch.maximum(pair_mask.sum(axis=-1)[..., None, None], 1)
    # global type positions
    global_type_pos = index_mean(
        entry_pos * base_type_weight[..., None],
        batch, mask[:, None, None])
    return torch.concatenate((type_features(local_type_pos),
                            type_features(global_type_pos)), axis=-1)

def paired_distance_features(x, y, d_min=0.0, d_max=22.0, num_rbf=16):
    """Compute distance features for a pair of structures x and y.
    
    Args:
        x, y: residue atom positions of shape (N, atoms, 3).
        d_min: minimum distance for binning. Default: 0.0.
        d_max: maximum distance for binning. Default: 22.0.
        num_rbf: number of radial basis functions.
    Returns:
        5 * 5 * 16 radial basis functions per residue pair.
    """
    dist = (x[..., :, None, :] - y[..., None, :, :]).norm()
    dist = dist.reshape(*dist.shape[:-2], -1)
    return distance_rbf(dist, d_min, d_max, num_rbf).reshape(*dist.shape[:-1], -1)

def rotation_features(frames, neighbours=None):
    """Relative rotation features for a set of residue frames and neighbours.
    
    Args:
        frames (Rigid3Array): local coordinate frames for a set of residues.
        neighbours (Optional[array]): neighbours of each residue
    Returns:
        Entries of the relative rotation matrix between
        pairs of neighbouring residues (N, K, 9).
    """
    if neighbours is None:
        rot = frames[:, None].inverse().rotation @ frames[None, :].rotation
    else:
        rot = frames[:, None].inverse().rotation @ frames[neighbours].rotation
    rot = rot.to_tensor().reshape(*rot.shape, -1)
    return rot

def position_rotation_features(pos: Vec3Array, neighbours=None):
    """Relative cotation features computed from atom positions.
    
    Args:
        pos (Vec3Array): residue backbone atom coordinates.
        neighbours (Optional[array]): neighbours of each residue
    Returns:
        Entries of the relative rotation matrix between
        pairs of neighbouring residues (N, K, 9).
    """
    frames, _ = extract_aa_frames(pos)
    if neighbours is None:
        rot = frames[:, None].inverse().rotation @ frames[None, :].rotation
    else:
        rot = frames[:, None].inverse().rotation @ frames[neighbours].rotation
    rot = rot.to_tensor().reshape(*rot.shape, -1)
    return rot

def paired_rotation_features(x, y):
    """Compute rotation features for a pair of structures x and y.
    
    Args:
        x, y: residue atom positions of shape (N, atoms, 3).
        neighbours (Optional[array]): neighbours of each residue
    Returns:
        Entries of the relative rotation matrix between
        pairs of residues in x and y.
    """
    x_frames = make_transform_from_reference(x[..., 0], x[..., 1], x[..., 2])
    y_frames = make_transform_from_reference(y[..., 0], y[..., 1], y[..., 2])
    rot = x_frames.inverse().rotation[:, None] @ y_frames.rotation[None, :]
    return rot.to_tensor().reshape(*rot.shape, -1)

def pair_vector_features(pos, neighbours=None, scale: float = 0.1):
    """Compute local coordinate features (PyTorch, close to JAX original).

    Args:
        pos: Vec3Array with shape (N, A) OR torch.Tensor with shape (N, A, 3).
        neighbours: optional LongTensor (N, K). If None -> (N, N) broadcast arange.
        scale: position scale factor.

    Returns:
        torch.Tensor of shape (N, K, 12*A) = concat([direction(=6*A), coords(=6*A)], -1).
    """
    if torch.is_tensor(pos):
        # assert pos.ndim == 3 and pos.shape[-1] == 3, f"expected (N,A,3), got {tuple(pos.shape)}"
        pos = Vec3Array.from_array(pos)
    else:
        assert isinstance(pos, Vec3Array), "pos must be Vec3Array or torch.Tensor (N,A,3)"

    device = pos.x.device
    N = pos.shape[0]
    A = pos.shape[1]

    if neighbours is None:
        neighbours = torch.arange(N, device=device, dtype=torch.long)[None, :].expand(N, N)
    else:
        neighbours = torch.as_tensor(neighbours, device=device, dtype=torch.long)

    frames, _ = extract_aa_frames(pos)  

    pair_vectors = torch.cat((
        # broadcast_to(frames.apply_inverse_to_point(pos[:,None])) to (N,K,A,3)
        frames[:, None, None].apply_inverse_to_point(pos[:, None]).to_tensor()
            .expand(neighbours.shape[0], neighbours.shape[1], A, 3),
        # frames.apply_inverse_to_point(pos[neighbours]) -> (N,K,A,3)
        frames[:, None, None].apply_inverse_to_point(pos[neighbours]).to_tensor(),
    ), dim=-2)  

    pair_vectors = Vec3Array.from_array(pair_vectors)  # (N,K,2A)

    direction = pair_vectors.normalized().to_tensor().reshape(*pair_vectors.shape[:-1], -1)
    coords = (scale * pair_vectors).to_tensor().reshape(*pair_vectors.shape[:-1], -1)

    return torch.cat((direction, coords), dim=-1)


class LinearToPoints(nn.Module):
    """Linear layer returning points in 3D space."""
    def __init__(self, size: int = 8, init: str = "linear"):
        super().__init__()
        self.size = int(size)
        self.init = init

        self.proj = Linear(self.size * 3, bias=False, initializer=self.init)

    def forward(self, data: torch.Tensor, frames):
        dtype = data.dtype

        raw = self.proj(data).to(torch.float32)

        raw = raw.reshape(*raw.shape[:-1], self.size, 3)

        points = Vec3Array.from_array(raw)                
        points = frames[..., None].apply_to_point(points)

        return points.to_tensor().to(dtype)

class SparseStructureMessage(nn.Module):
    """Message passing wrapper."""

    def __init__(self, config,):
        super().__init__()
        self.config = config
        self._pair_mlp: Optional[nn.Module] = None 

    def forward(self, local, pos, pair, pair_mask, neighbours, resi, chain, batch, mask):
        c = self.config

        pos = Vec3Array.from_array(pos.to(dtype=torch.float32))

        if self._pair_mlp is None:
            self._pair_mlp = MLP(
                2 * int(c.pair_size),
                int(local.shape[-1]),
                depth=3,
                activation=F.gelu,
                final_init="zeros",
            ).to(pair.device)

        pair = self._pair_mlp(pair)

        local_update = torch.where(pair_mask[..., None].to(torch.bool), pair, torch.zeros_like(pair)).sum(dim=1)
        local_update = local_update / pair.shape[1]
        return local_update
    
class SparseStructureAttention(nn.Module):
    """Sparse attention wrapper."""

    def __init__(self, config, normalize: bool = True):
        super().__init__()
        self.config = config
        self.normalize = normalize

        c = self.config
        final_init = c.update_init if getattr(c, "update_init", None) else "zeros"

        Attn = SparseInvariantMultiQueryAttention if getattr(c, "multi_query", False) else SparseInvariantPointAttention
        self.attn = Attn(
            heads=c.heads,
            size=c.key_size,
            final_init=final_init,
            normalize=self.normalize,
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
        frames, _ = extract_aa_frames(Vec3Array.from_array(pos))

        if hasattr(frames, "to_tensor"):
            frames_t = frames.to_tensor()
        elif hasattr(frames, "to_array"):
            frames_t = frames.to_array()
        else:
            frames_t = frames 

        local_update = self.attn(
            local,
            pair,
            frames_t,
            neighbours,
            pair_mask,
        )
        return local_update

class SemiEquivariantSparseStructureAttention(nn.Module):
    """Sparse attention wrapper"""

    def __init__(
        self,
        config,
        normalize: bool = False
    ):
        super().__init__()
        self.normalize = bool(normalize)
        self.config = config

        c = self.config
        final_init = c.update_init if getattr(c, "update_init", None) else "zeros"

        self.attn = SparseSemiEquivariantPointAttention(
            heads=c.heads,
            size=c.key_size,
            final_init=final_init,
            normalize=self.normalize,
        )

    def forward(self, local, pos, pair, pair_mask, neighbours, resi, chain, batch, mask):
        c = self.config
        local_update = self.attn(local, pair, pos, neighbours, pair_mask)
        return local_update
    
class SparseInvariantMultiQueryAttention(nn.Module):
    """Sparse IPA with multi-query attention."""

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
        self.config = config
        self.size = int(size)
        self.heads = int(heads)
        self.query_points = int(query_points)
        self.value_points = int(value_points)
        self.final_init = final_init
        self.normalize = bool(normalize)

        H, C, Q = self.heads, self.size, self.query_points

        self.q_proj = Linear(H * C, bias=True, initializer="linear")
        self.k_proj = Linear(1 * C, bias=True, initializer="linear")  # heads=1
        self.v_proj = Linear(1 * C, bias=True, initializer="linear")  # heads=1

        self.qp_proj = Linear(H * Q * 3, bias=True, initializer="linear")
        self.kp_proj = Linear(1 * Q * 3, bias=True, initializer="linear")  # heads=1
        self.vp_proj = Linear(1 * Q * 3, bias=True, initializer="linear")  # heads=1

        self.pair_bias = Linear(H, bias=False, initializer="linear")

        gamma0 = torch.log(torch.exp(torch.tensor(1.0)) - torch.tensor(1.0))
        self.gamma = nn.Parameter(gamma0.repeat(H))  # (H,)

        local_size = int(getattr(config, "local_size"))
        self.out_proj = Linear(local_size, bias=False, initializer=final_init)

    def _component(self, x: torch.Tensor, proj: Linear, heads: int) -> torch.Tensor:
        y = proj(x)  # (N, heads*size)
        return y.view(y.shape[0], heads, self.size)

    def _points_global(self, x: torch.Tensor, proj: Linear, frames: Rigid3Array, heads: int) -> torch.Tensor:
        N, H, Q = x.shape[0], heads, self.query_points
        y = proj(x).view(N, H, Q, 3)                       # local
        pts = Vec3Array.from_array(y)
        pts_g = frames[:, None, None].apply_to_point(pts)  # global
        return pts_g.to_tensor() if hasattr(pts_g, "to_tensor") else pts_g.to_array()

    def _to_local(self, x_global: torch.Tensor, frames: Rigid3Array) -> torch.Tensor:
        pts = Vec3Array.from_array(x_global)
        pts_l = frames[:, None, None].apply_inverse_to_point(pts)
        return pts_l.to_tensor() if hasattr(pts_l, "to_tensor") else pts_l.to_array()

    def forward(
        self,
        local: torch.Tensor,        # (N, C_local)
        pair: torch.Tensor,         # (N, K, C_pair)
        frames: torch.Tensor,        
        neighbours: torch.Tensor,   # (N, K)
        mask: torch.Tensor,         # (N, K)
    ) -> torch.Tensor:
        frames_r = Rigid3Array.from_array(frames.to(dtype=torch.float32))

        neighbours = neighbours.to(torch.long)
        mask_bool = mask.to(torch.bool)

        N, K = neighbours.shape
        H, C, Q = self.heads, self.size, self.query_points

        q = self._component(local, self.q_proj, heads=H)
        k = self._component(local, self.k_proj, heads=1)
        v = self._component(local, self.v_proj, heads=1)

        q = F.layer_norm(q, (C,), weight=None, bias=None)
        k = F.layer_norm(k, (C,), weight=None, bias=None)

        qp = self._points_global(local, self.qp_proj, frames_r, heads=H)   # (N,H,Q,3)
        kp = self._points_global(local, self.kp_proj, frames_r, heads=1)   # (N,1,Q,3)
        vp = self._points_global(local, self.vp_proj, frames_r, heads=1)   # (N,1,Q,3)

        k_g = k[neighbours]    # (N,K,1,C)
        v_g = v[neighbours]    # (N,K,1,C)
        kp_g = kp[neighbours]  # (N,K,1,Q,3)
        vp_g = vp[neighbours]  # (N,K,1,Q,3)

        k_g = k_g.expand(-1, -1, H, -1)             # (N,K,H,C)
        v_g = v_g.expand(-1, -1, H, -1)             # (N,K,H,C)
        kp_g = kp_g.expand(-1, -1, H, -1, -1)       # (N,K,H,Q,3)
        vp_g = vp_g.expand(-1, -1, H, -1, -1)       # (N,K,H,Q,3)

        w_C = (2.0 / (9.0 * Q)) ** 0.5
        w_L = (1.0 / 3.0) ** 0.5

        scale = F.softplus(self.gamma.view(1, 1, H)) * w_C / 2.0  # (1,1,H)

        attn_logits = (q.unsqueeze(1) * k_g).sum(dim=-1)          # (N,K,H)
        attn_logits = attn_logits / (C ** 0.5)

        dist = ((qp.unsqueeze(1) - kp_g) ** 2).sum(dim=(-1, -2))  # (N,K,H)

        bias = self.pair_bias(pair)                                # (N,K,H)

        attn_logits = w_L * (attn_logits - scale * dist + bias)

        neg_inf = torch.full_like(attn_logits, -1e9)
        attn_logits = torch.where(mask_bool.unsqueeze(-1), attn_logits, neg_inf)

        attn = torch.softmax(attn_logits, dim=1)
        attn = torch.where(mask_bool.unsqueeze(-1), attn, torch.zeros_like(attn))

        local_update = torch.einsum("nkh,nkhc->nhc", attn, v_g).reshape(N, -1)   # (N,H*C)
        pair_update  = torch.einsum("nkh,nkc->nhc",  attn, pair).reshape(N, -1) # (N,H*C_pair)

        point_global = torch.einsum("nkh,nkhqd->nhqd", attn, vp_g)              # (N,H,Q,3)
        point_local = self._to_local(point_global, frames_r).reshape(N, -1)    # (N,H*Q*3)

        result = torch.cat([local_update, pair_update, point_local], dim=-1)
        return self.out_proj(result)


class SparseAttention(nn.Module):
    """Sparse attention without point bias."""

    def __init__(
        self,
        size: int = 32,
        heads: int = 4,
        final_init: str = "zeros",
        normalize: bool = False
    ):
        super().__init__()
        self.size = int(size)
        self.heads = int(heads)
        self.final_init = final_init
        self.normalize = bool(normalize)

        self.qkv = Linear(self.heads * 3 * self.size, bias=False, name="qkv")

        self.ln_q = nn.LayerNorm(self.size, elementwise_affine=True)
        self.ln_k = nn.LayerNorm(self.size, elementwise_affine=True)

        self.bias = Linear(self.heads, bias=False, name="bias")

        self._ln_local: Optional[nn.LayerNorm] = None
        self._project_out: Optional[Linear] = None

    def forward(self, local, pair, neighbours, mask):
        if self.normalize:
            if self._ln_local is None:
                self._ln_local = nn.LayerNorm(local.shape[-1], elementwise_affine=True).to(local.device)
            local = self._ln_local(local)

        qkv = self.qkv(local).view(*local.shape[:-1], self.heads, 3 * self.size)
        q, k, v = torch.split(qkv, self.size, dim=-1)

        q = self.ln_q(q)
        k = self.ln_k(k)

        bias = self.bias(pair)
        if bias.dim() < local.dim():
            extension = (local.dim() - bias.dim()) * [1]
            bias = bias.view(bias.shape[0], *extension, *bias.shape[1:])

        w_L = (1.0 / 2.0) ** 0.5

        if neighbours is not None:
            neighbours = neighbours.to(dtype=torch.long)

        dot = (1.0 / self.size) ** 0.5 * (q.unsqueeze(1) * k[neighbours]).sum(dim=-1)
        attn_logits = w_L * (dot + bias)

        if neighbours is None:
            pair_mask = mask
        else:
            pair_mask = mask * (neighbours != -1)

        attn_logits = torch.where(
            pair_mask[..., None].to(torch.bool),
            attn_logits,
            torch.full_like(attn_logits, -1e9),
        )

        attn = torch.softmax(attn_logits, dim=1)  
        attn = torch.where(pair_mask[..., None].to(torch.bool), attn, torch.zeros_like(attn))

        out_pair = torch.einsum("nkh,nkc->nhc", attn, pair)
        out_scalar = torch.einsum("nkh,nkhc->nhc", attn, v[neighbours])

        x = torch.cat(
            [
                out_pair.reshape(*out_pair.shape[:-2], -1),
                out_scalar.reshape(*out_scalar.shape[:-2], -1),
            ],
            dim=-1,
        )

        if self._project_out is None:
            self._project_out = Linear(int(local.shape[-1]), initializer=self.final_init, name="project_out").to(local.device)

        out = self._project_out(x)
        return out.to(dtype=local.dtype)

class DenseNonEquivariantPointAttention(nn.Module):
    """EXPERIMENTAL. Non-equivariant point attention."""

    def __init__(
        self,
        size: int = 32,
        heads: int = 4,
        query_points: int = 8,
        value_points: int = 8,
        final_init: str = "zeros",
        normalize: bool = False,
        use_pair: bool = False,
        name: Optional[str] = "ada_point_attention",
    ):
        super().__init__()
        self.size = int(size)
        self.heads = int(heads)
        self.query_points = int(query_points)
        self.value_points = int(value_points)
        self.final_init = final_init
        self.use_pair = bool(use_pair)
        self.normalize = bool(normalize)

        H, C = self.heads, self.size
        Q, V = self.query_points, self.value_points

        self._ln_local: Optional[nn.LayerNorm] = None

        self.q_proj = Linear(C * H, initializer="linear")
        self.k_proj = Linear(C * H, initializer="linear")
        self.v_proj = Linear(C * H, initializer="linear")

        self.ln_q = nn.LayerNorm(C, elementwise_affine=True)
        self.ln_k = nn.LayerNorm(C, elementwise_affine=True)

        self.qp_proj = Linear(Q * H * 3, initializer="zeros")
        self.kp_proj = Linear(Q * H * 3, initializer="zeros")
        self.vp_proj = Linear(Q * H * 3, initializer="zeros")

        self.relpos = sequence_relative_position(32, one_hot=True)

        self.pair_lin = Linear(C + 3 * V, bias=False, initializer="linear")
        self.bias_lin = Linear(H, initializer="linear")

        # resi embedding table
        emb = torch.empty(66, C + 3 * V)
        init_linear()(emb)  
        self.resi_embedding = nn.Parameter(emb)

        gamma0 = torch.log(torch.exp(torch.tensor(1.0)) - torch.tensor(1.0))
        self.gamma = nn.Parameter(gamma0.repeat(H))  # (H,)

        self._out: Optional[Linear] = None

    def forward(self, local, pos, resi, chain, batch, mask):
        if self.normalize:
            if self._ln_local is None:
                self._ln_local = nn.LayerNorm(local.shape[-1], elementwise_affine=True).to(local.device)
            local = self._ln_local(local)

        H, C = self.heads, self.size
        Q, V = self.query_points, self.value_points

        same_batch = batch[:, None].eq(batch[None, :])
        pair_mask = (mask[:, None] * mask[None, :] * same_batch.to(mask.dtype)).to(torch.bool)

        def attention_component(x, proj: Linear):
            y = proj(x)  # (N, H*C)
            return y.view(*y.shape[:-1], H, C)

        def attention_point(x, proj: Linear):
            y = proj(x)  # (N, H*Q*3)
            y = y.view(*y.shape[:-1], H, Q, 3)
            y = y + pos[:, 1][:, None, None]  
            return y

        query = self.ln_q(attention_component(local, self.q_proj))
        key = self.ln_k(attention_component(local, self.k_proj))
        value = attention_component(local, self.v_proj)

        query_points = attention_point(local, self.qp_proj)
        key_points = attention_point(local, self.kp_proj)
        value_points = attention_point(local, self.vp_proj)

        value = torch.cat((value, value_points.reshape(*value.shape[:-1], -1)), dim=-1)

        inner_product_attention = torch.einsum(
            "ihc,jhc->ijh",
            query * ((1.0 / query.shape[-1]) ** 0.5),
            key,
        )

        point_attention = -((query_points[:, None] - key_points[None, :]) ** 2).sum(dim=(-1, -2))

        resi_dist = (resi[:, None] - resi[None, :]).clamp(-32, 32)
        other_chain = chain[:, None].ne(chain[None, :])

        ca = Vec3Array.from_array(10.0 * pos[:, 1])
        dist = (ca[:, None] - ca[None, :]).norm()
        dist = distance_rbf(dist, bins=16)

        rel = (query_points[:, None] - key_points[None, :]).reshape(dist.shape[0], dist.shape[1], -1)

        rdist = (resi_dist + 32).to(torch.long)
        rdist = torch.where(other_chain, torch.full_like(rdist, 65), rdist)
        rdist_emb = self.resi_embedding[rdist]  # (N, N, C + 3*V)

        pair = rdist_emb + self.pair_lin(torch.cat((rel, dist), dim=-1))
        bias = self.bias_lin(torch.cat((rel, dist), dim=-1))  # (N, N, H)

        pair = pair[:, :, None, :] + value[None, :]  # (N, N, H, C+3*V)

        w_C = (2.0 / (9.0 * Q)) ** 0.5
        point_scale = F.softplus(self.gamma.view(1, 1, H)) * w_C / 2.0

        attn = (inner_product_attention + point_scale * point_attention + bias) * ((1.0 / 3.0) ** 0.5)
        attn = torch.where(pair_mask[..., None], attn, torch.full_like(attn, -1e9))
        attn = torch.softmax(attn, dim=1)
        attn = torch.where(pair_mask[..., None], attn, torch.zeros_like(attn))

        result = torch.einsum("ijh,ijhc->ihc", attn, pair).reshape(local.shape[0], -1)

        if self._out is None:
            self._out = Linear(int(local.shape[-1]), bias=False, initializer="zeros").to(local.device)

        return self._out(result)

class SparseSemiEquivariantPointAttention(nn.Module):
    """Sparse IPA with additional non-equivariant features."""

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
        self.config = config
        self.size = int(size)
        self.heads = int(heads)
        self.query_points = int(query_points)
        self.value_points = int(value_points)
        self.final_init = final_init
        self.normalize = bool(normalize)

        H, C = self.heads, self.size
        Q, V = self.query_points, self.value_points

        local_size = int(getattr(config, "local_size"))

        self.ln_local = nn.LayerNorm(local_size, elementwise_affine=True)

        self.qkv = Linear(H * 3 * C, bias=False, initializer="linear", name="qkv")

        self.ln_q = nn.LayerNorm(C, elementwise_affine=True)
        self.ln_k = nn.LayerNorm(C, elementwise_affine=True)

        self.qkv_g = Linear(H * (2 * Q + V) * 3, bias=True, initializer="linear")

        self.bias = Linear(H, bias=False, initializer="linear", name="bias")

        gamma0 = torch.log(torch.exp(torch.tensor(1.0)) - torch.tensor(1.0))
        self.gamma = nn.Parameter(gamma0.repeat(H))  # (H,)

        self.project_out = Linear(local_size, bias=True, initializer=final_init, name="project_out")

    def forward(
        self,
        local: torch.Tensor,                 
        pair: torch.Tensor,                  
        pos: torch.Tensor,                   
        neighbours: Optional[torch.Tensor],  
        mask: torch.Tensor,                  
    ) -> torch.Tensor:
        if self.normalize:
            local = self.ln_local(local)

        H, C = self.heads, self.size
        Q, V = self.query_points, self.value_points

        qkv = self.qkv(local)
        qkv = qkv.view(*local.shape[:-1], H, 3 * C)
        q, k, v = torch.split(qkv, C, dim=-1)

        q = self.ln_q(q)
        k = self.ln_k(k)

        qkv_g = self.qkv_g(local)  # (..., H*(2Q+V)*3)

        # (c) maybe bug
        qkv_g = qkv.view(qkv.shape[0], -1, 3) 

        base = pos[:, 1, :]  # (N, 3)
        qkv_g = qkv_g + base[:, None, :]           # (N, H*(2Q+V), 3)

        qkv_g = qkv_g.view(qkv_g.shape[0], H, (2 * Q + V), 3)  # (N, H, 2Q+V, 3)
        q_g, k_g, v_g = torch.split(qkv_g, [Q, Q, V], dim=-2)  # (N,H,Q,3), (N,H,Q,3), (N,H,V,3)

        bias = self.bias(pair)  
        if bias.dim() < local.dim():
            extension = (local.dim() - bias.dim()) * [1]
            bias = bias.view(bias.shape[0], *extension, *bias.shape[1:])

        w_C = (2.0 / (9.0 * Q)) ** 0.5
        w_L = (1.0 / 3.0) ** 0.5

        dfactor = F.softplus(self.gamma.view(1, 1, H)) * w_C / 2.0  # (1,1,H)

        if neighbours is None: # (c) here full-attn can be implemented
            pair_mask = mask
            raise NotImplementedError(
                "neighbours=None branch is not supported in this port yet "
                "(the original code has it, but downstream einsums expect neighbour dimension)."
            )
        else:
            neighbours = neighbours.to(torch.long)
            pair_mask = mask * (neighbours != -1).to(mask.dtype)

        pair_mask_bool = pair_mask.to(torch.bool)

        k_n = k[neighbours]  # (N,K,H,C)
        v_n = v[neighbours]  # (N,K,H,C)

        k_gn = k_g[neighbours]  # (N,K,H,Q,3)
        v_gn = v_g[neighbours]  # (N,K,H,V,3)

        dist = dfactor * (q_g.unsqueeze(1) - k_gn).pow(2).sum(dim=(-1, -2))

        dot = (1.0 / C) ** 0.5 * (q.unsqueeze(1) * k_n).sum(dim=-1)

        attn_logits = w_L * (dot + bias - dist)

        neg_inf = torch.full_like(attn_logits, -1e9)
        attn_logits = torch.where(pair_mask_bool.unsqueeze(-1), attn_logits, neg_inf)

        attn = torch.softmax(attn_logits, dim=1)
        attn = torch.where(pair_mask_bool.unsqueeze(-1), attn, torch.zeros_like(attn))
        # (N,H,C_pair)
        out_pair = torch.einsum("nkh,nkc->nhc", attn, pair)

        # (N,H,C)
        out_scalar = torch.einsum("nkh,nkhc->nhc", attn, v_n)

        out_point = torch.einsum("nkh,nkhvd->nhvd", attn, v_gn)  # (N,H,V,3)

        out_point_v = Vec3Array.from_array(out_point.to(torch.float32))
        base_v = Vec3Array.from_array(base.to(torch.float32))[:, None, None]
        out_point_v = out_point_v - base_v

        out_norm = out_point_v.norm()  
        out_point_t = out_point_v.to_tensor() if hasattr(out_point_v, "to_tensor") else out_point_v.to_array()

        out = torch.cat(
            [
                out_pair.reshape(out_pair.shape[0], -1),
                out_scalar.reshape(out_scalar.shape[0], -1),
                out_point_t.reshape(out_point_t.shape[0], -1),
                out_norm.reshape(out_norm.shape[0], -1),
            ],
            dim=-1,
        )

        out = self.project_out(out)
        return out.to(dtype=local.dtype)
    
class SparseInvariantPointAttention(nn.Module):
    """Sparse IPA."""

    def __init__(
        self,
        size: int = 32,
        heads: int = 4,
        query_points: int = 8,
        value_points: int = 8,
        final_init: str = "zeros",
        normalize: bool = False
    ):
        super().__init__()
        self.size = int(size)
        self.heads = int(heads)
        self.query_points = int(query_points)
        self.value_points = int(value_points)
        self.final_init = final_init
        self.normalize = bool(normalize)

        self.ln_q = nn.LayerNorm(self.size, elementwise_affine=True)
        self.ln_k = nn.LayerNorm(self.size, elementwise_affine=True)

        self.qkv = Linear(self.heads * 3 * self.size, bias=False, name="qkv")

        self.qkv_global = LinearToPoints(
            self.heads * (2 * self.query_points + self.value_points)
        )

        self.bias = Linear(self.heads, bias=False, name="bias")

        gamma0 = torch.log(torch.exp(torch.tensor(1.0)) - torch.tensor(1.0))
        self.gamma = nn.Parameter(gamma0.repeat(self.heads))  # (heads,)

        self._ln_local: Optional[nn.LayerNorm] = None

        self._project_out: Optional[Linear] = None

    def forward(self, local, pair, frames, neighbours, mask):
        if self.normalize:
            if self._ln_local is None:
                self._ln_local = nn.LayerNorm(local.shape[-1], elementwise_affine=True).to(local.device)
            local = self._ln_local(local)

        frames = Rigid3Array.from_array(frames.to(dtype=torch.float32))

        qkv = self.qkv(local).view(*local.shape[:-1], self.heads, 3 * self.size)
        q, k, v = torch.split(qkv, self.size, dim=-1)

        q = self.ln_q(q)
        k = self.ln_k(k)

        qkv_g = self.qkv_global(local, frames)
        qkv_g = qkv_g.view(*qkv_g.shape[:-2], self.heads, -1, 3)

        q_g, k_g, v_g = torch.split(
            qkv_g,
            [self.query_points, self.query_points, self.value_points],
            dim=-2,
        )

        bias = self.bias(pair)
        if bias.dim() < local.dim():
            extension = (local.dim() - bias.dim()) * [1]
            bias = bias.view(bias.shape[0], *extension, *bias.shape[1:])

        w_C = (2.0 / (9.0 * self.query_points)) ** 0.5
        w_L = (1.0 / 3.0) ** 0.5

        dfactor = F.softplus(self.gamma.view(1, 1, self.heads)) * w_C / 2.0

        if neighbours is not None:
            neighbours = neighbours.to(dtype=torch.long)

        dist = dfactor * (q_g[:, None] - k_g[neighbours]).pow(2).sum(dim=(-1, -2))
        dot = (1.0 / self.size) ** 0.5 * (q.unsqueeze(1) * k[neighbours]).sum(dim=-1)

        attn_logits = w_L * (dot + bias - dist)

        if neighbours is None:
            pair_mask = mask
        else:
            pair_mask = mask * (neighbours != -1)

        attn_logits = torch.where(pair_mask[..., None].to(torch.bool), attn_logits, torch.full_like(attn_logits, -1e9))

        attn = torch.softmax(attn_logits, dim=1)
        attn = torch.where(pair_mask[..., None].to(torch.bool), attn, torch.zeros_like(attn))

        out_pair = torch.einsum("nkh,nkc->nhc", attn, pair)
        out_scalar = torch.einsum("nkh,nkhc->nhc", attn, v[neighbours])
        out_point = torch.einsum("nkh,nkhvd->nhvd", attn, v_g[neighbours])

        out_point = Vec3Array.from_array(out_point.to(dtype=torch.float32))
        out_point = frames[:, None, None].apply_inverse_to_point(out_point)

        out_norm = out_point.norm()
        out_point = out_point.to_tensor() if hasattr(out_point, "to_tensor") else out_point.to_array()

        concat = torch.cat(
            [
                out_pair.reshape(*out_pair.shape[:-2], -1),
                out_scalar.reshape(*out_scalar.shape[:-2], -1),
                out_point.reshape(*out_point.shape[:-3], -1),
                out_norm.reshape(*out_point.shape[:-3], -1),
            ],
            dim=-1,
        )

        if self._project_out is None:
            self._project_out = Linear(int(local.shape[-1]), initializer=self.final_init, name="project_out").to(local.device)

        out = self._project_out(concat)
        return out.to(dtype=local.dtype)