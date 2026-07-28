"""Tensor-native geometry kernels for the experimental Torch path.

This module intentionally avoids Vec3Array/Rigid3Array wrappers in hot-path
feature construction. Rotations are regular tensors with shape ``(..., 3, 3)``
and translations are tensors with shape ``(..., 3)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_nodes(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather rows from dim 0 for arbitrary index shape.

    ``-1`` keeps legacy semantics by wrapping to the last row; callers should
    mask invalid neighbours separately.
    """
    if index.dtype != torch.long:
        index = index.long()
    flat = index.reshape(-1).remainder(source.shape[0])
    gathered = torch.index_select(source, dim=0, index=flat)
    return gathered.reshape(*index.shape, *source.shape[1:])


def normalize_vector(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


def distance_rbf(distance, min_distance=0.0, max_distance=22.0, bins=64):
    step = (max_distance - min_distance) / bins
    centers = min_distance + (
        torch.arange(bins, device=distance.device, dtype=distance.dtype) * step + step / 2
    )
    return torch.exp(-((distance[..., None] - centers) ** 2) / (step ** 2))


def distance_one_hot(distance, min_distance=0.0, max_distance=22.0, bins=64):
    step = (max_distance - min_distance) / bins
    centers = min_distance + (
        torch.arange(bins, device=distance.device, dtype=distance.dtype) * step + step / 2
    )
    argmin = torch.argmin(torch.abs(distance[..., None] - centers), dim=-1)
    return F.one_hot(argmin, num_classes=bins).to(distance.dtype)


def compute_pseudo_cb(positions: torch.Tensor) -> torch.Tensor:
    n, ca, c = torch.moveaxis(positions[..., :3, :], -2, 0)
    b = ca - n
    c_vec = c - ca
    a = torch.cross(b, c_vec, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca


def positions_to_ncacocb(pos: torch.Tensor) -> torch.Tensor:
    cb = compute_pseudo_cb(pos)
    return torch.cat((pos[:, :4], cb[..., None, :]), dim=-2)


def unique_chain(chain: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    n = chain.shape[0]
    change = torch.ones(n, dtype=torch.bool, device=chain.device)
    change[1:] = (chain[1:] != chain[:-1]) | (batch[1:] != batch[:-1])
    return torch.cumsum(change.to(dtype=chain.dtype), dim=0) - 1


def index_sum(data: torch.Tensor, index: torch.Tensor, mask: torch.Tensor, apply_mask: bool = True) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()
    mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
    data_masked = torch.where(mask_bool, data, torch.zeros_like(data))
    result = torch.zeros_like(data)
    result.index_add_(0, index, data_masked)
    gathered = result.index_select(0, index)
    if not apply_mask:
        return gathered
    return torch.where(mask_bool, gathered, torch.zeros_like(gathered))


def index_mean(data, index, mask, weight=None, apply_mask=True):
    if index.dtype != torch.long:
        index = index.long()
    mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
    x = data * weight if weight is not None else data
    x = torch.where(mask_bool, x, torch.zeros_like(x))
    result = torch.zeros_like(x)
    result.index_add_(0, index, x)

    position_weight = mask_bool
    if weight is not None:
        position_weight = torch.where(mask_bool, weight, torch.zeros_like(weight))
    pw = position_weight + torch.zeros_like(x)
    denom = torch.zeros_like(x)
    denom.index_add_(0, index, pw)
    out = (result / denom.clamp_min(1e-6)).index_select(0, index)
    if not apply_mask:
        return out
    return torch.where(mask_bool, out, torch.zeros_like(out))


def index_count(index, mask, apply_mask=True):
    index = index.long()
    mask = mask.bool()
    result = torch.zeros_like(index)
    result.scatter_add_(0, index, mask.long())
    gathered = result.index_select(0, index)
    if not apply_mask:
        return gathered
    return torch.where(mask, gathered, torch.zeros_like(gathered))


def sequence_relative_position(
    count: int | None = 32,
    one_hot: bool = False,
    cyclic: bool = False,
    pseudo_chains: bool = False,
    identify_ends: bool = False,
):
    if count is None:
        count = 32

    def inner(resi, chain, batch, neighbours=None, cyclic_mask=None):
        if neighbours is None:
            same_chain = chain[:, None] == chain[None, :]
            same_batch = batch[:, None] == batch[None, :]
            dist = resi[:, None] - resi[None, :]
        else:
            neighbours_l = neighbours.long()
            chain_n = gather_nodes(chain, neighbours_l)
            batch_n = gather_nodes(batch, neighbours_l)
            resi_n = gather_nodes(resi, neighbours_l)
            same_chain = chain[:, None] == chain_n
            same_batch = batch[:, None] == batch_n
            dist = resi[:, None] - resi_n
        flat_resi = torch.arange(resi.shape[0], dtype=torch.int32, device=resi.device)
        if cyclic:
            lengths = index_count(chain, torch.ones_like(chain, dtype=torch.bool))
            wrap = abs(dist) > lengths[:, None] / 2
            if cyclic_mask is not None:
                wrap = wrap * cyclic_mask[:, None]
            dist = torch.where(
                wrap,
                torch.where(dist < 0, dist % lengths[:, None], dist % lengths[:, None] - lengths[:, None]),
                dist,
            )
        dist = torch.clamp(dist, -count, count) + count
        if identify_ends:
            count_total = 2 * count - 2
            dist = torch.where(dist == 0, 2 * count - 2, dist - 1)
            dist = torch.where(same_chain, dist, 2 * count - 2)
            dist = torch.where(same_batch, dist, 2 * count - 2)
        elif pseudo_chains:
            if neighbours is None:
                flat_dist = flat_resi[:, None] - flat_resi[None, :]
            else:
                flat_dist = flat_resi[:, None] - gather_nodes(flat_resi, neighbours.long())
            flat_dist = torch.where(flat_dist >= 0, 0, 2 * count - 1)
            count_total = 2 * count + 2
            dist = torch.where(same_chain, dist, flat_dist)
            dist = torch.where(same_batch, dist, 2 * count + 1)
        else:
            count_total = 2 * count + 2
            dist = torch.where(same_chain, dist, 2 * count + 1)
            dist = torch.where(same_batch, dist, 2 * count + 1)
        if one_hot:
            dist = F.one_hot(dist.long(), num_classes=int(count_total)).float()
        return dist

    return inner


def frames_from_ncac(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build residue frames from N/CA/C atoms.

    Args:
        pos: ``(..., A, 3)`` with N=0, CA=1, C=2.

    Returns:
        ``(rot, trans)`` where ``rot`` maps local coordinates to global
        coordinates and ``trans`` is the CA origin.
    """
    n = pos[..., 0, :]
    ca = pos[..., 1, :]
    c = pos[..., 2, :]
    e0 = normalize_vector(c - ca)
    e1_raw = n - ca
    e1 = normalize_vector(e1_raw - (e1_raw * e0).sum(dim=-1, keepdim=True) * e0)
    e2 = torch.cross(e0, e1, dim=-1)
    rot = torch.stack((e0, e1, e2), dim=-1)
    return rot, ca


def apply_frame(rot: torch.Tensor, trans: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply local-to-global transform to points."""
    return torch.einsum("...ij,...j->...i", rot, points) + trans


def frame_tensor_from_rot_trans(rot: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    frame = torch.zeros((*rot.shape[:-2], 4, 4), device=rot.device, dtype=rot.dtype)
    frame[..., :3, :3] = rot
    frame[..., :3, 3] = trans
    frame[..., 3, 3] = 1.0
    return frame


def rot_trans_from_frame_tensor(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if frames.shape[-2:] == (4, 4):
        return frames[..., :3, :3], frames[..., :3, 3]
    if frames.shape[-2:] == (3, 4):
        return frames[..., :3, :3], frames[..., :3, 3]
    raise ValueError(f"Expected frame tensor with trailing shape (4,4) or (3,4), got {tuple(frames.shape)}")


def invert_apply_frame(rot: torch.Tensor, trans: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply global-to-local transform to points."""
    shifted = points - trans
    return torch.einsum("...ji,...j->...i", rot, shifted)


def local_positions(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return frames and atom coordinates in their residue-local frames."""
    rot, trans = frames_from_ncac(pos)
    local = invert_apply_frame(rot[..., None, :, :], trans[..., None, :], pos)
    return rot, trans, local


def distance_features(
    pos: torch.Tensor,
    neighbours: torch.Tensor | None = None,
    d_min: float = 0.0,
    d_max: float = 22.0,
    num_rbf: int = 16,
) -> torch.Tensor:
    """Tensor-native equivalent of ``modules.geometric.distance_features``."""
    pos = pos[..., :5, :]
    if neighbours is None:
        neigh = pos[None, :, None, :, :]
        center = pos[:, None, :, None, :]
    else:
        neigh = gather_nodes(pos, neighbours)[:, :, None, :, :]
        center = pos[:, None, :, None, :]
    dist = torch.linalg.norm(center - neigh, dim=-1)
    dist = dist.reshape(*dist.shape[:-2], -1)
    return distance_rbf(dist, d_min, d_max, num_rbf).reshape(*dist.shape[:-1], -1)


def paired_distance_features(
    x: torch.Tensor,
    y: torch.Tensor,
    d_min: float = 0.0,
    d_max: float = 22.0,
    num_rbf: int = 16,
) -> torch.Tensor:
    """Tensor-native equivalent of ``modules.geometric.paired_distance_features``."""
    dist = torch.linalg.norm(x[..., :, None, :] - y[..., None, :, :], dim=-1)
    dist = dist.reshape(*dist.shape[:-2], -1)
    return distance_rbf(dist, d_min, d_max, num_rbf).reshape(*dist.shape[:-1], -1)


def direction_features(pos: torch.Tensor, neighbours: torch.Tensor | None = None) -> torch.Tensor:
    """Neighbour atom directions in each center residue frame."""
    rot, trans = frames_from_ncac(pos)
    if neighbours is None:
        neigh = pos[None, :, :, :]
    else:
        neigh = gather_nodes(pos, neighbours)
    local = invert_apply_frame(rot[:, None, None, :, :], trans[:, None, None, :], neigh)
    return normalize_vector(local).reshape(*local.shape[:2], -1)


def rotation_features_from_frames(rot: torch.Tensor, neighbours: torch.Tensor | None = None) -> torch.Tensor:
    """Relative rotations between center frames and neighbour frames."""
    if neighbours is None:
        neigh_rot = rot[None, :, :, :]
    else:
        neigh_rot = gather_nodes(rot, neighbours)
    rel = torch.matmul(rot[:, None].transpose(-1, -2), neigh_rot)
    return rel.reshape(*rel.shape[:-2], 9)


def position_rotation_features(pos: torch.Tensor, neighbours: torch.Tensor | None = None) -> torch.Tensor:
    rot, _ = frames_from_ncac(pos)
    return rotation_features_from_frames(rot, neighbours)


def paired_rotation_features(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Tensor-native equivalent of ``modules.geometric.paired_rotation_features``."""
    x_rot, _ = frames_from_ncac(x)
    y_rot, _ = frames_from_ncac(y)
    rel = torch.matmul(x_rot.transpose(-1, -2)[:, None], y_rot[None])
    return rel.reshape(*rel.shape[:-2], 9)


def pair_vector_features(
    pos: torch.Tensor,
    neighbours: torch.Tensor | None = None,
    scale: float = 0.1,
) -> torch.Tensor:
    """Tensor-native equivalent of ``modules.geometric.pair_vector_features``."""
    rot, trans = frames_from_ncac(pos)
    n = pos.shape[0]
    a = pos.shape[1]
    if neighbours is None:
        neighbours = torch.arange(n, device=pos.device, dtype=torch.long)[None, :].expand(n, n)
    else:
        neighbours = neighbours.long()

    self_local = invert_apply_frame(rot[:, None, None, :, :], trans[:, None, None, :], pos[:, None])
    self_local = self_local.expand(neighbours.shape[0], neighbours.shape[1], a, 3)
    neigh_local = invert_apply_frame(
        rot[:, None, None, :, :],
        trans[:, None, None, :],
        gather_nodes(pos, neighbours),
    )
    pair = torch.cat((self_local, neigh_local), dim=-2)
    direction = normalize_vector(pair).reshape(*pair.shape[:-2], -1)
    coords = (scale * pair).reshape(*pair.shape[:-2], -1)
    return torch.cat((direction, coords), dim=-1)


def local_frame_features(pos: torch.Tensor) -> torch.Tensor:
    """Flatten local atom coordinates for encoder/decoder local features."""
    _, _, local = local_positions(pos)
    return local.reshape(local.shape[0], -1)
