"""Compatibility geometry facade for legacy CAESAR modules.

The hot-path tensor kernels live in ``caesar.experimental.torch_native``.
This module keeps the historical import surface used by encoder/decoder/loss
code while avoiding duplicated JAX-style geometry implementations here.

OpenFold all-atom helpers are intentionally still used at the boundary where
legacy callers expect ``Vec3Array``/``Rigid3Array`` objects.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch

from caesar.aflib.common import residue_constants
from caesar.experimental.torch_native import tensor_geometry as _tg
from caesar.utils.all_atom_multimer import (
    frames_and_literature_positions_to_atom14_pos,
    make_transform_from_reference,
    torsion_angles_to_frames,
)
from caesar.utils.geometry import Rigid3Array, Vec3Array

Float = Union[float, torch.Tensor]


def _to_tensor(x: torch.Tensor | Vec3Array) -> torch.Tensor:
    return x.to_tensor() if hasattr(x, "to_tensor") else x


def _to_vec3(x: torch.Tensor | Vec3Array) -> Vec3Array:
    return x if isinstance(x, Vec3Array) else Vec3Array.from_array(x)


def _gather_rows(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return _tg.gather_nodes(source, index)


def _gather_cols_per_row(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()
    return torch.gather(source, dim=1, index=index.remainder(source.shape[1]))


def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...ij,...j->...i", r, t)


def make_backbone_affine(
    positions: Vec3Array | torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    atoms: Optional[Iterable[str]] = None,
    atom_order: Optional[Dict[str, int]] = None,
) -> Tuple[Rigid3Array, torch.Tensor]:
    """Build residue frames from three named atoms and return legacy wrappers."""
    if atom_order is None:
        atom_order = residue_constants.atom_order
    if atoms is None:
        atoms = ("N", "CA", "C")
    pos = _to_tensor(positions)
    pos_vec = _to_vec3(pos)
    if mask is None:
        mask = pos.new_ones(pos.shape[:-1])
    a, b, c = [atom_order[name] for name in atoms]
    rigid_mask = (mask[..., a] * mask[..., b] * mask[..., c]).float()
    rigid = make_transform_from_reference(
        a_xyz=pos_vec[..., a],
        b_xyz=pos_vec[..., b],
        c_xyz=pos_vec[..., c],
    )
    return rigid, rigid_mask


def extract_aa_frames(positions: Vec3Array | torch.Tensor) -> Tuple[Rigid3Array, Vec3Array]:
    """Extract protein residue frames and local atom positions."""
    pos = _to_tensor(positions)
    rot, trans, local = _tg.local_positions(pos)
    frames = Rigid3Array.from_array4x4(_tg.frame_tensor_from_rot_trans(rot, trans))
    return frames, Vec3Array.from_array(local)


def extract_na_frames(positions: Vec3Array | torch.Tensor):
    """Extract nucleic acid frames through the legacy OpenFold-compatible path."""
    rigids, _ = make_backbone_affine(positions, atoms=("O4", "C1", "C2"))
    local_positions = rigids[..., None].apply_inverse_to_point(_to_vec3(positions))
    return rigids, local_positions


def extract_aa_relmap(positions: Vec3Array | torch.Tensor, atom_mask: torch.Tensor):
    frames, _ = extract_aa_frames(positions)
    pos_vec = _to_vec3(positions)
    relmap = frames[:, None, None].apply_inverse_to_point(pos_vec[None])
    rel_mask = atom_mask[:, None, 1:2] * atom_mask[None, :]
    return relmap, rel_mask


def single_protein_sidechains(aatype: torch.Tensor, frames: Rigid3Array, angles: torch.Tensor):
    all_frames_to_global = torsion_angles_to_frames(aatype, frames, angles)
    pred_positions = frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global)
    return pred_positions, all_frames_to_global


def extract_neighbours(num_index=16, num_spatial=16, num_random=16):
    def inner(pos, resi, chain, item, mask):
        neighbours = get_index_neighbours(num_index)(resi, chain, item, mask)
        neighbours = get_spatial_neighbours(num_spatial)(_to_tensor(pos)[:, 1], item, mask, neighbours)
        neighbours = get_random_neighbours(num_random)(_to_tensor(pos)[:, 1], item, mask, neighbours)
        return neighbours

    return inner


def extract_neighbours_salad_compatible(num_index=16, num_spatial=16, num_random=16):
    """SALAD-compatible sequence/spatial/random neighbour extraction."""

    def inner(pos, resi, chain, item, mask, *, generator: Optional[torch.Generator] = None):
        pos_t = _to_tensor(pos)
        if pos_t.ndim == 3:
            ca = pos_t[:, 1]
        elif pos_t.ndim == 2:
            ca = pos_t
        else:
            raise ValueError(f"Unsupported pos shape for neighbours: {tuple(pos_t.shape)}")

        same_batch = item[:, None] == item[None, :]
        same_chain = chain[:, None] == chain[None, :]
        valid = same_batch & (mask[:, None].bool() & mask[None, :].bool())

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

        total = int(num_index) + int(num_spatial) + int(num_random)
        if int(num_random) <= 0:
            deterministic_distance = torch.where(
                within,
                torch.full_like(distance, -10_000.0),
                inf,
            )
            deterministic_distance = torch.where(valid, deterministic_distance, inf)
            return get_neighbours(total)(deterministic_distance, mask)

        random_distance = -3.0 * torch.log(torch.clamp(distance, min=1e-6))
        u = torch.rand(random_distance.shape, device=random_distance.device, dtype=random_distance.dtype, generator=generator)
        u = u * (1.0 - 2e-6) + 1e-6
        random_distance = -(random_distance - torch.log(-torch.log(u)))
        random_distance = torch.where(within, torch.full_like(random_distance, -10_000.0), random_distance)
        random_distance = torch.where(valid, random_distance, inf)
        return get_neighbours(total)(random_distance, mask)

    return inner


def get_index_neighbours(count: int):
    def inner(resi, chain, item, mask, neighbours=None):
        distance = (resi[:, None] - resi[None, :]).abs().float()
        valid = (item[:, None] == item[None, :]) & (chain[:, None] == chain[None, :])
        valid = valid & (mask[:, None].bool() & mask[None, :].bool())
        return get_neighbours(count)(distance, valid, neighbours)

    return inner


def get_spatial_neighbours(count: int):
    def inner(pos: Any, item, mask, neighbours=None):
        pos_t = _to_tensor(pos)
        distance = torch.linalg.norm(pos_t[:, None] - pos_t[None, :], dim=-1)
        valid = (item[:, None] == item[None, :]) & (mask[:, None].bool() & mask[None, :].bool())
        return get_neighbours(count)(distance, valid, neighbours)

    return inner


def get_neighbours(count: int):
    def inner(distance: torch.Tensor, mask: torch.Tensor, neighbours: Optional[torch.Tensor] = None):
        n = distance.shape[0]
        distance = torch.where(mask.bool(), distance.float(), torch.full_like(distance.float(), float("inf")))

        if neighbours is not None:
            neighbours = neighbours.long()
            gathered = _gather_cols_per_row(distance, neighbours)
            update = torch.where(neighbours != -1, torch.full_like(gathered, float("inf")), gathered)
            distance = distance.scatter(dim=1, index=neighbours.remainder(n), src=update)

        k = min(int(count), n)
        if k <= 0:
            knn = torch.empty((n, 0), device=distance.device, dtype=torch.long)
        else:
            col_index = torch.arange(n, device=distance.device, dtype=distance.dtype)[None, :]
            distance_for_sort = torch.where(torch.isfinite(distance), distance + col_index * 1e-6, distance)
            knn = torch.topk(distance_for_sort, k=k, dim=-1, largest=False, sorted=True).indices
            knn = torch.where(_gather_cols_per_row(distance, knn) < float("inf"), knn, -1)

        if neighbours is not None:
            knn = torch.cat((neighbours, knn), dim=-1)
        return knn

    return inner


def get_random_neighbours(count: int):
    def inner(pos: Any, item, mask, neighbours=None, generator=None):
        pos_t = _to_tensor(pos)
        if pos_t.ndim == 2 and pos_t.shape[0] == pos_t.shape[1]:
            distance = pos_t
        else:
            distance = torch.linalg.norm(pos_t[:, None] - pos_t[None, :], dim=-1)
        same_item = item[:, None] == item[None, :]
        weight = -3.0 * torch.log(torch.clamp(distance, min=1e-6))
        uniform = torch.rand(weight.shape, dtype=weight.dtype, device=weight.device, generator=generator)
        uniform = uniform * (1.0 - 2e-6) + 1e-6
        distance = -(weight - torch.log(-torch.log(uniform)))
        valid = same_item & (mask[:, None].bool() & mask[None, :].bool())
        return get_neighbours(count)(distance, valid, neighbours)

    return inner


def get_contact_neighbours(count):
    def inner(pair_condition, mask, neighbours):
        is_conditioned = pair_condition.any(dim=-1)
        distance = torch.where(
            is_conditioned,
            torch.zeros(is_conditioned.shape, dtype=torch.float32, device=is_conditioned.device),
            torch.full(is_conditioned.shape, float("inf"), dtype=torch.float32, device=is_conditioned.device),
        )
        return get_neighbours(count)(distance, mask, neighbours)

    return inner


def bond_angle(x, y, z):
    left = x - y
    right = z - y
    den = torch.linalg.norm(left, dim=-1) * torch.linalg.norm(right, dim=-1)
    cos_tau = (left * right).sum(dim=-1) / den.clamp_min(1e-6)
    return torch.arccos(torch.clamp(cos_tau, -1.0, 1.0)) / torch.pi * 180.0


def dihedral_angle(a, b, c, d):
    x = b - a
    y = c - b
    z = d - c
    y_norm = torch.linalg.norm(y, dim=-1)
    return torch.arctan2(
        y_norm * (x * torch.cross(y, z, dim=-1)).sum(dim=-1),
        (torch.cross(x, y, dim=-1) * torch.cross(y, z, dim=-1)).sum(dim=-1),
    ) / torch.pi * 180.0


def batch_pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a.unsqueeze(2) - b.unsqueeze(1)
    return torch.sqrt((diff**2).sum(dim=-1) + 1e-8)


def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    return torch.sqrt((diff**2).sum(dim=-1) + 1e-8)


def distance_rbf(distance, min_distance=0.0, max_distance=22.0, bins=64):
    return _tg.distance_rbf(distance, min_distance=min_distance, max_distance=max_distance, bins=bins)


def distance_one_hot(distance, min_distance=0.0, max_distance=22.0, bins=64):
    return _tg.distance_one_hot(distance, min_distance=min_distance, max_distance=max_distance, bins=bins)


def hl_gaussian(data, minimum=0.0, maximum=22.0, bins=64, sigma_ratio=1.0):
    step = (maximum - minimum) / bins
    sigma = torch.as_tensor(step * sigma_ratio, dtype=data.dtype, device=data.device)
    sqrt_two = torch.sqrt(torch.as_tensor(2.0, dtype=data.dtype, device=data.device))
    lower = torch.arange(bins, device=data.device, dtype=data.dtype) * step
    upper = lower + step
    values = torch.special.erf((upper - data[..., None]) / (sqrt_two * sigma))
    values = values - torch.special.erf((lower - data[..., None]) / (sqrt_two * sigma))
    denom = torch.special.erf((maximum - data[..., None]) / (sqrt_two * sigma))
    denom = denom - torch.special.erf((minimum - data[..., None]) / (sqrt_two * sigma))
    return values / denom.clamp_min(1e-6)


def compute_pseudo_cb(positions):
    return _tg.compute_pseudo_cb(positions)


def axis_index(data: torch.Tensor, dim=0):
    return torch.arange(data.shape[dim], dtype=torch.int32, device=data.device)


def index_sum(data: torch.Tensor, index: torch.Tensor, mask: torch.Tensor, apply_mask: bool = True) -> torch.Tensor:
    return _tg.index_sum(data, index, mask, apply_mask=apply_mask)


def index_max(data: torch.Tensor, index: torch.Tensor, mask: torch.Tensor, apply_mask: bool = True) -> torch.Tensor:
    n = data.shape[0]
    index = index.long()
    mask_bool = mask if mask.dtype == torch.bool else (mask > 0)
    mask_b = mask_bool.view(n, *([1] * (data.ndim - 1)))
    dmin = data.amin()
    data_masked = torch.where(mask_b, data, dmin)
    result = torch.full_like(data, dmin)
    idx = index.view(n, *([1] * (data.ndim - 1))).expand_as(data_masked)
    result.scatter_reduce_(0, idx, data_masked, reduce="amax", include_self=True)
    gathered = result.index_select(0, index)
    if not apply_mask:
        return gathered
    return torch.where(mask_b, gathered, torch.full_like(gathered, dmin))


def index_mean(data, index, mask, weight=None, apply_mask=True):
    return _tg.index_mean(data, index, mask, weight=weight, apply_mask=apply_mask)


def index_var(data: torch.Tensor, index: torch.Tensor, mask: torch.Tensor, apply_mask: bool = True):
    ex2 = index_mean(data**2, index, mask, apply_mask=apply_mask)
    e2x = index_mean(data, index, mask, apply_mask=apply_mask) ** 2
    return ex2 - e2x


def index_std(
    data: torch.Tensor,
    index: torch.Tensor,
    mask: torch.Tensor,
    apply_mask: bool = True,
    eps: Optional[float] = 1e-6,
):
    return torch.sqrt(index_var(data, index, mask, apply_mask=apply_mask) + eps)


def index_kabsch(x, y, index, mask, weight=None):
    device = x.device
    x_center = index_mean(x, index, mask[:, None], weight)
    y_center = index_mean(y, index, mask[:, None], weight)
    x0 = x - x_center
    y0 = y - y_center
    if weight is None:
        weight = torch.ones_like(index, dtype=x.dtype)
    cov = index_sum(weight[:, None, None] * x0[:, :, None] * y0[:, None, :], index, mask[:, None, None])
    u, _, v = torch.linalg.svd(cov.detach(), full_matrices=True)
    det = torch.linalg.det(u) * torch.linalg.det(v)
    flip = torch.ones((cov.shape[0], 3), device=device, dtype=x.dtype)
    flip[:, -1] = det
    rot = torch.einsum("...ak,...kb->...ba", u * flip[:, None, :], v)
    return rot, x_center, y_center


def index_align(x, y, index, mask, weight=None):
    return_vec3 = isinstance(x, Vec3Array) or isinstance(y, Vec3Array)
    x_t = _to_tensor(x)
    y_t = _to_tensor(y)
    rot, x_center, y_center = index_kabsch(x_t[:, 1], y_t[:, 1], index, mask, weight=weight)
    result = torch.einsum("...ak,...ik->...ia", rot, (x_t - x_center[:, None])) + y_center[:, None]
    return Vec3Array.from_array(result) if return_vec3 else result


def apply_alignment(x, kabsch_data):
    rot, x_center, y_center = kabsch_data
    x_t = _to_tensor(x)
    delta = torch.einsum("...ak,...k->...a", torch.swapaxes(rot, -1, -2), y_center) - x_center
    result = torch.einsum("...ak,...ik->...ia", rot, x_t + delta[:, None])
    return Vec3Array.from_array(result) if isinstance(x, Vec3Array) else result


def sequence_relative_position(
    count: Optional[int] = 32,
    one_hot=False,
    cyclic=False,
    identify_ends=False,
    pseudo_chains=False,
):
    return _tg.sequence_relative_position(
        count=count,
        one_hot=one_hot,
        cyclic=cyclic,
        identify_ends=identify_ends,
        pseudo_chains=pseudo_chains,
    )


def index_count(index, mask, apply_mask=True):
    return _tg.index_count(index, mask, apply_mask=apply_mask)


def unique_chain(chain, batch):
    return _tg.unique_chain(chain, batch)


def positions_to_ncacocb(pos: torch.Tensor):
    return _tg.positions_to_ncacocb(pos)


def replace_masked_with(pos: torch.Tensor, atom_mask: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
    return torch.where(atom_mask[..., None].bool(), pos, replacement)


def atom37_to_ncacocb(pos37: torch.Tensor) -> torch.Tensor:
    pseudo_cb = compute_pseudo_cb(pos37)
    return torch.cat((pos37[:, :3], pos37[:, 4:5], pseudo_cb[:, None, :]), dim=-2)


def atom37_local_feature_channels(
    local_pos37: torch.Tensor,
    atom_mask37: torch.Tensor,
    rbf_bins: int = 8,
) -> torch.Tensor:
    mask = atom_mask37.to(dtype=local_pos37.dtype)
    local_pos37 = torch.where(mask[..., None] > 0, local_pos37, torch.zeros_like(local_pos37))
    local_dist37 = torch.sqrt(torch.clamp((local_pos37**2).sum(dim=-1), min=1e-6))
    local_rbf37 = distance_rbf(local_dist37, 0.0, 22.0, rbf_bins)
    return torch.cat(
        [
            local_pos37.reshape(*local_pos37.shape[:-2], -1),
            local_dist37.reshape(*local_dist37.shape[:-1], -1),
            local_rbf37.reshape(*local_rbf37.shape[:-2], -1),
            mask.reshape(*mask.shape[:-1], -1),
        ],
        dim=-1,
    )


def mask_atom37_local_positions(local_pos37: torch.Tensor, atom_mask37: torch.Tensor) -> torch.Tensor:
    mask = atom_mask37.to(dtype=local_pos37.dtype)
    return torch.where(mask[..., None] > 0, local_pos37, torch.zeros_like(local_pos37))


__all__ = [
    "Float",
    "Rigid3Array",
    "Vec3Array",
    "apply_alignment",
    "atom37_local_feature_channels",
    "atom37_to_ncacocb",
    "axis_index",
    "batch_pairwise_dist",
    "bond_angle",
    "compute_pseudo_cb",
    "dihedral_angle",
    "distance_one_hot",
    "distance_rbf",
    "extract_aa_frames",
    "extract_aa_relmap",
    "extract_na_frames",
    "extract_neighbours",
    "extract_neighbours_salad_compatible",
    "get_contact_neighbours",
    "get_index_neighbours",
    "get_neighbours",
    "get_random_neighbours",
    "get_spatial_neighbours",
    "hl_gaussian",
    "index_align",
    "index_count",
    "index_kabsch",
    "index_max",
    "index_mean",
    "index_std",
    "index_sum",
    "index_var",
    "make_backbone_affine",
    "mask_atom37_local_positions",
    "pairwise_dist",
    "positions_to_ncacocb",
    "replace_masked_with",
    "rot_matmul",
    "rot_vec_mul",
    "sequence_relative_position",
    "single_protein_sidechains",
    "unique_chain",
]
