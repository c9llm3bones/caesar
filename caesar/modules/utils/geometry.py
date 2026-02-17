"""PyTorch geometry utilities for protein structures.

Adapted from OpenFold (https://github.com/aqlaboratory/openfold)
and SALAD (https://github.com/mjendrusch/salad)
"""
# IN PROGRESS
from __future__ import annotations

import dataclasses
from random import uniform
from typing import Any, Optional, Union, Tuple, Iterable, Dict, List

import torch
import torch.nn.functional as F
import numpy as np

import caesar.utils.geometry
from caesar.aflib.common import residue_constants
from caesar.utils import geometry
from caesar.utils.all_atom_multimer import (
    make_transform_from_reference, torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos)

from caesar.utils.geometry import Rigid3Array, Vec3Array

Float = Union[float, torch.Tensor]

def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiply two rotation matrices [*, 3, 3]."""
    def row_mul(i):
        return torch.stack(
            [
                a[..., i, 0] * b[..., 0, 0]
                + a[..., i, 1] * b[..., 1, 0]
                + a[..., i, 2] * b[..., 2, 0],
                a[..., i, 0] * b[..., 0, 1]
                + a[..., i, 1] * b[..., 1, 1]
                + a[..., i, 2] * b[..., 2, 1],
                a[..., i, 0] * b[..., 0, 2]
                + a[..., i, 1] * b[..., 1, 2]
                + a[..., i, 2] * b[..., 2, 2],
            ],
            dim=-1,
        )

    return torch.stack(
        [row_mul(0), row_mul(1), row_mul(2)],
        dim=-2
    )

def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rotation matrix r to vector t: r @ t."""
    x, y, z = torch.unbind(t, dim=-1)
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )

def make_backbone_affine(
    positions: Vec3Array,
    mask: torch.Tensor,
    atoms: Optional[Iterable[str]] = None,
    atom_order: Optional[Dict[str, int]] = None
    ) -> Tuple[Rigid3Array, torch.Tensor]:
    """Make backbone Rigid3Array and mask.
    
    Args:
        positions: atom positions of shape (N, 3+, 3).
        mask: atom mask.
        atoms: list of 3 atom names to use for frame construction. Default: N, CA, C
        atom_order: order of atom names in positions. Default: atom14.
    Returns:
        Rigid3Array of residue frames.
    """
    if atom_order is None:
        atom_order = residue_constants.atom_order
    if atoms is None:
        atoms = ('N', 'CA', 'C')
    a, b, c = [residue_constants.atom_order[name] for name in atoms]

    rigid_mask = (mask[..., a] * mask[..., b] * mask[..., c]).float()

    rigid = make_transform_from_reference(
        a_xyz=positions[..., a],
        b_xyz=positions[..., b],
        c_xyz=positions[..., c])

    return rigid, rigid_mask

def extract_aa_frames(positions: Vec3Array) -> Tuple[Rigid3Array, Vec3Array]:
    """Extract frames from protein backbone positions.
    
    Args:
        positions: Vec3Array of amino acid backbone atoms in atom14 format.
    Returns:
        Rigid3Array of residue frames and Vec3Array of local-frame
        side chain atom positions.
    """        
    rigids, _ = make_backbone_affine(positions, torch.ones((positions.shape[0], 14)), None)
    local_positions = rigids[..., None].apply_inverse_to_point(positions)
    return rigids, local_positions

def extract_na_frames(positions: Vec3Array):
    """Extract frames from nucleic acid backbone positions using O4, C1 and C2.

    Args:
        positions: Vec3Array of nucleic acid backbone atoms in atom14 format.
    Returns:
        Rigid3Array of residue frames and Vec3Array of local-frame
        side chain atom positions.
    """
    rigids, _ = make_backbone_affine(positions, atoms=('O4', 'C1', 'C2'))
    local_positions = rigids[..., None].apply_inverse_to_point(positions)
    return rigids, local_positions

def extract_aa_relmap(positions: Vec3Array, atom_mask: torch.Tensor):
    """Extract relative atom positions between residue pairs.
    
    Args:
        positions: Vec3Array of atom14 format atom positions.
        atom_mask: atom14 format atom mask.
    Returns:
        Relative position map of shape (N, N, 3+, 3) and corresponding
        mask of shape (N, N, 3+).
    """
    frames, _ = extract_aa_frames(positions)
    relmap = frames[:, None, None].apply_inverse_to_point(positions[None])
    rel_mask = atom_mask[:, None, 1:2] * atom_mask[None, :]
    return relmap, rel_mask

def single_protein_sidechains(aatype: torch.Tensor, frames: Rigid3Array, angles: torch.Tensor):
    """Compute side chain atom positions given backbone frames and dihedral angles.
    
    Args:
        aatype: integer amino acid type (0-19) of shape (N,).
        frames: amino acid backbone frames of shape (N,).
        angles: side chain dihedral angles of shape (N, 7, 2).
    Returns:
        atom14 format all-atom positions of shape (N, 14, 3).
    """
    # Map torsion angles to frames.
    # geometry.Rigid3Array with shape (N, 8)
    all_frames_to_global = torsion_angles_to_frames(
        aatype,
        frames,
        angles
    )

    # Use frames and literature positions to create the final atom coordinates.
    # geometry.Vec3Array with shape (N, 14)
    pred_positions = frames_and_literature_positions_to_atom14_pos(
        aatype, all_frames_to_global
    )

    return pred_positions, all_frames_to_global

def extract_neighbours(num_index=16, num_spatial=16, num_random=16):
    """Extracts the default set of nearest neighbours of each residue.
    
    Args:
        num_index: number of neighbours using residue index distance.
        num_spatial: number of nearest neighbours using euclidean distance d.
        num_random: number of neighbours sampled with probability 1 / d^3.

    Returns:
        A function extracting per-residue nearest neighbours given
        atom positions (N, 3+, 3), residue, chain and batch index
        and a residue mask.
    """
    def inner(pos, resi, chain, item, mask):
        neighbours = get_index_neighbours(num_index)(resi, chain, item, mask)
        neighbours = get_spatial_neighbours(num_spatial)(pos[:, 1], item, mask, neighbours)
        neighbours = get_random_neighbours(num_random)(pos[:, 1], item, mask, neighbours)
        return neighbours
    return inner


def extract_neighbours_salad_compatible(num_index=16, num_spatial=16, num_random=16):
    """Extract nearest neighbours of a residue based on sequence and euclidean distance."""
    def inner(pos, resi, chain, item, mask, *, generator: Optional[torch.Generator] = None):
        if isinstance(pos, Vec3Array):
            ca = pos[:, 1]
        else:
            if pos.ndim == 3:
                ca = Vec3Array.from_array(pos)[:, 1]
            elif pos.ndim == 2:
                ca = Vec3Array.from_array(pos)
            else:
                raise ValueError(f"Unsupported pos shape for neighbours: {tuple(pos.shape)}")

        same_batch = item[:, None] == item[None, :]
        same_chain = chain[:, None] == chain[None, :]
        valid = same_batch * (mask[:, None] * mask[None, :])

        within = (resi[:, None] - resi[None, :]).abs() < int(num_index)
        within = within * same_batch * same_chain

        distance = (ca[:, None] - ca[None, :]).norm()
        inf = torch.full_like(distance, float("inf"))
        distance = torch.where(within, inf, distance)
        distance = torch.where(valid.bool(), distance, inf)

        if int(num_spatial) > 0:
            sorted_distance = torch.sort(distance, dim=-1).values
            cutoff = sorted_distance[:, : int(num_spatial)][:, -1]
            # NOTE: Keep JAX/SALAD broadcasting semantics from
            # structure_autoencoder.extract_neighbours:
            # `(distance < cutoff)` where cutoff has shape (N,).
            within = within | (distance < cutoff)

        random_distance = -3.0 * torch.log(torch.clamp(distance, min=1e-6))
        u = torch.rand(
            random_distance.shape,
            device=random_distance.device,
            dtype=random_distance.dtype,
            generator=generator,
        )
        u = u * (1.0 - 2e-6) + 1e-6
        gumbel = torch.log(-torch.log(u))
        random_distance = -(random_distance - gumbel)

        random_distance = torch.where(
            within.bool(),
            torch.full_like(random_distance, -10_000.0),
            random_distance,
        )
        random_distance = torch.where(valid.bool(), random_distance, inf)

        total = int(num_index) + int(num_spatial) + int(num_random)
        return get_neighbours(total)(random_distance, mask)

    return inner

def get_index_neighbours(count: int):
    """Extracts the `count` nearest neighbours based on residue index."""
    def inner(resi, chain, item, mask, neighbours=None):
        distance = abs(resi[:, None] - resi[None, :])
        same_chain = chain[:, None] == chain[None, :]
        same_item = item[:, None] == item[None, :]
        mask = same_item * same_chain * (mask[:, None] * mask[None, :])
        return get_neighbours(count)(distance, mask, neighbours)
    return inner

def get_spatial_neighbours(count: int):
    """Extracts the `count` nearest neighbours based on euclidean distance."""
    def inner(pos: Vec3Array, item, mask, neighbours=None):
        distance = (pos[:, None] - pos[None, :]).norm()
        same_item = item[:, None] == item[None, :]
        distance = torch.where(same_item, distance, torch.full_like(distance, float('inf')))
        mask = (mask[:, None] * mask[None, :] * same_item)
        return get_neighbours(count)(distance, mask, neighbours)
    return inner

def bond_angle(x, y, z):
    """Compute the bond angle between three atoms x, y and z.
    
    Args:
        x, y, z: atom positions of shape (..., 3).
    Returns:
        Bond angle with y as the central atom.
    """
    left = x - y
    right = z - y
    den = torch.linalg.norm(left, dim=-1) * torch.linalg.norm(right, dim=-1)
    den = den.clamp_min(1e-6)
    cos_tau = (left * right).sum(dim=-1) / den

    return torch.arccos(cos_tau) / torch.pi * 180

def dihedral_angle(a, b, c, d):
    """Compute the dihedral angle for four atoms a, b, c, d.
    
    Args:
        a, b, c, d: atom positions of shape (..., 3).
    Returns:
        Dihedral angle along a, b, c and d.
    """
    x = b - a
    y = c - b
    z = d - c
    y_norm = torch.linalg.norm(y, dim=-1)
    result = torch.arctan2(y_norm * (x * torch.cross(y, z, dim=-1)).sum(dim=-1),
                         (torch.cross(x, y, dim=-1) * torch.cross(y, z, dim=-1)).sum(dim=-1))
    return result / torch.pi * 180

def batch_pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise distances between points in a and b.
    
    Args:
        a: (batch, n, 3)
        b: (batch, m, 3)
    Returns:
        (batch, n, m) pairwise distances
    """
    diff = a.unsqueeze(2) - b.unsqueeze(1)  # (batch, n, m, 3)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)

def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise distances (non-batched).
    
    Args:
        a: (n, 3)
        b: (m, 3)
    Returns:
        (n, m) distances
    """
    diff = a.unsqueeze(1) - b.unsqueeze(0)  # (n, m, 3)
    return torch.sqrt((diff ** 2).sum(-1) + 1e-8)

def get_neighbours(count: int):
    def inner(
        distance: torch.Tensor,              # (N, N), int or float
        mask: torch.Tensor,                  # (N, N), bool or {0,1}
        neighbours: Optional[torch.Tensor] = None,  # (N, K)
    ):
        N = distance.shape[0]
        device = distance.device

        index = torch.arange(N, device=device)

        distance = distance.float()

        distance = torch.where(
            mask.bool(),
            distance,
            torch.full_like(distance, float("inf")),
        )

        if neighbours is not None:
            idx = index[:, None]

            update = torch.where(
                neighbours != -1,
                torch.full_like(neighbours, float("inf"), dtype=distance.dtype),
                distance[idx, neighbours],
            )

            distance[idx, neighbours] = update

        # Canonical tie-break across frameworks: prefer lower column index
        # when distances are equal.
        col_index = torch.arange(N, device=device, dtype=distance.dtype)[None, :]
        distance_for_sort = torch.where(
            torch.isfinite(distance),
            distance + col_index * 1e-6,
            distance,
        )
        knn = torch.argsort(distance_for_sort, dim=-1, stable=True)[..., :count]

        knn = torch.where(
            distance[index[:, None], knn] < float("inf"),
            knn,
            -1,
        )

        if neighbours is not None:
            knn = torch.cat((neighbours, knn), dim=-1)

        return knn

    return inner

def get_random_neighbours(count: int):
    """Extracts `count` neighbours with probability 1 / d^3."""
    def inner(pos: Any, item, mask, neighbours=None, generator=None):
        distance = None
        if isinstance(pos, torch.Tensor):
            assert(pos.ndim == 2)
            if pos.shape[0] == pos.shape[1]:
                distance = pos
            else:
                pos = Vec3Array.from_array(pos)
        if distance is None:
            distance = (pos[:, None] - pos[None, :]).norm()
        same_item = item[:, None] == item[None, :]
        # apply gumbel topk trick to select random neighbours
        weight = -3 * torch.log(distance + 1e-6)
        uniform = torch.rand(weight.shape, dtype=weight.dtype, device=weight.device, generator=generator)
        gumbel = torch.log(-torch.log(uniform))
        weight = weight - gumbel
        distance = -weight
        distance = torch.where(same_item, distance, torch.inf)
        mask = (mask[:, None] * mask[None, :] * same_item)
        return get_neighbours(count)(distance, mask, neighbours)
    return inner

def get_contact_neighbours(count):
    """Extracts `count` neighbours with non-zero pair conditioning information."""
    def inner(pair_condition, mask, neighbours):
        # get pairs where at least one condition is True
        is_conditioned = pair_condition.any(dim=-1)
        # construct a distance matrix with random values for conditioned positions
        # and infinite distance everywhere else.
        distance = torch.where(is_conditioned,
                             torch.rand(is_conditioned.shape, dtype=torch.float32),
                              torch.full_like(is_conditioned, float('inf')))
        # get new neighbours using that distance matrix.
        # infinite distance results in a pair being masked (set to -1 in neighbours).
        # this way effectively /no additional neighbours are used when not conditioning/
        return get_neighbours(count)(distance, mask, neighbours)
    return inner

def distance_rbf(distance, min_distance=0.0, max_distance=22.0, bins=64):
    step = (max_distance - min_distance) / bins
    centers = min_distance + (
        torch.arange(bins, device=distance.device, dtype=distance.dtype) * step + step / 2
    )
    return torch.exp(-((distance[..., None] - centers) ** 2) / (step ** 2))

def distance_one_hot(distance, min_distance=0.0, max_distance=22.0, bins=64):
    """Computes one-hot encoding of continuous inputs.

    Args:
        distance: array of distances to embed.
        min_distance: minimum distance for input binning.
        max_distance: maximum distance for input binning.
        bins: number of bins.

    Returns:
        One-hot encoding of `distance` with `bins` bins.
    """
    step = (max_distance - min_distance) / bins
    centers = (min_distance
               + torch.arange(bins, device=distance.device, dtype=distance.dtype) * step
               + step / 2)
    argmin = torch.argmin(torch.abs(distance[..., None] - centers), dim=-1)
    oh = F.one_hot(argmin, num_classes=bins)
    return oh.to(distance.dtype)

def hl_gaussian(data, minimum=0.0, maximum=22.0, bins=64, sigma_ratio=1.0):
    """Computes HL-Gauss multihot encoding of continuous inputs.
    
    HL-Gauss embedding proposed by Farebrother et al. 2024 (arxiv.org/abs/2403.03950v1).
    Convolves a Gaussian with the input and bins the resulting probability distribution.
    
    Args:
        distance: array of distances to embed.
        min_distance: minimum distance for input binning.
        max_distance: maximum distance for input binning.
        bins: number of bins.
        sigma_ratio: scaling factor broadening or sharpening
            the gaussian convolved with the data.

    Returns:
        HL-Gauss encoding of `distance` with `bins` bins.
    """
    step = (maximum - minimum) / bins
    sigma = step * sigma_ratio
    def erf_aux(x, mu):
        return torch.scipy.special.erf((x - mu) / (torch.sqrt(2) * sigma))
    def erfinv_aux(x, mu):
        return torch.scipy.special.erfinv(x) * (torch.sqrt(2) * sigma) + mu
    # set an upper and lower bound for the input data
    # to stop the output from becoming NaN
    lower_bound = erfinv_aux(-0.999, minimum)
    upper_bound = erfinv_aux(0.999, maximum)
    data = torch.clip(data, lower_bound, upper_bound)
    lower = torch.arange(bins, device=data.device, dtype=data.dtype) * step
    upper = lower + step
    value = erf_aux(upper, data[..., None]) - erf_aux(lower, data[..., None])
    value /= erf_aux(maximum, data[..., None]) - erf_aux(minimum, data[..., None])
    return value

def compute_pseudo_cb(positions):
    """Compute idealized CB atom positions.
    
    Args:
        positions: array of atom positions in atom14 order
            containing at least N, CA and C of shape (N, 3+, 3).

    Returns:
        Array of idealized CB atom positions of shape (N, 3).
    """
    n, ca, c = torch.moveaxis(positions[..., :3, :], -2, 0)
    b = ca - n
    c = c - ca
    a = torch.cross(b, c, dim=-1)
    const = [-0.58273431, 0.56802827, -0.54067466]
    return const[0] * a + const[1] * b + const[2] * c + ca

def axis_index(data: torch.Tensor, dim=0):
    """Index along an dim of `data`.
    
    Args:
        data: input data array of shape (..., N, ...).
        dim: dim along which to construct an index.
    Returns:
        Index array containing values (0, ..., N-1).
    """
    return torch.arange(data.shape[dim], dtype=torch.int32, device=data.device)

def index_sum(data: torch.Tensor,
              index: torch.Tensor,
              mask: torch.Tensor,
              apply_mask: bool = True) -> torch.Tensor:
    """Sum array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Sum of array entries with the same index value, broadcasted
        to all entries with that index value.
        E.g. for values [1, 2, 3, 4, 5] and index [0, 0, 0, 1, 1]
        the result would be [6, 6, 6, 9, 9].
    """
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

def index_max(data: torch.Tensor,
              index: torch.Tensor,
              mask: torch.Tensor,
              apply_mask: bool = True) -> torch.Tensor:
    """Maximum of array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Maximum of array entries with the same index value, broadcasted
        to all entries with that index value.
        E.g. for values [1, 2, 3, 4, 5] and index [0, 0, 0, 1, 1]
        the result would be [3, 3, 3, 5, 5].
    """
    N = data.shape[0]
    index = index.long()
    mask_bool = mask if mask.dtype == torch.bool else (mask > 0)
    mask_b = mask_bool.view(N, *([1] * (data.ndim - 1)))
    dmin = data.amin()  
    data_masked = torch.where(mask_b, data, dmin)
    result = torch.full_like(data, dmin)
    idx = index.view(N, *([1] * (data.ndim - 1))).expand_as(data_masked)
    if hasattr(result, "scatter_reduce_"):
        result.scatter_reduce_(0, idx, data_masked, reduce="amax", include_self=True)
    elif hasattr(torch, "scatter_reduce"):
        result = torch.scatter_reduce(result, 0, idx, data_masked, reduce="amax", include_self=True)
    else:
        raise RuntimeError(
            "index_max requires scatter_reduce_ (PyTorch >= 1.12/2.0). "
            "Upgrade PyTorch to avoid slow CPU fallback."
        )
    gathered = result.index_select(0, index)  

    if not apply_mask:
        return gathered

    return torch.where(mask_b, gathered, torch.full_like(gathered, dmin))

def index_mean(data, index, mask, weight=None, apply_mask=True):
    """Mean of array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Mean of array entries with the same index value, broadcasted
        to all entries with that index value.
        E.g. for values [1, 2, 3, 4, 5] and index [0, 0, 0, 1, 1]
        the result would be [2, 2, 2, 4.5, 4.5].
    """
    if index.dtype != torch.long:
        index = index.long()

    mask_bool = mask if mask.dtype == torch.bool else (mask != 0)

    x = data
    if weight is not None:
        x = x * weight

    x = torch.where(mask_bool, x, torch.zeros_like(x))

    result = torch.zeros_like(x)
    result.index_add_(0, index, x)

    position_weight = mask_bool
    if weight is not None:
        position_weight = torch.where(mask_bool, weight, torch.zeros_like(weight))

    pw = position_weight + torch.zeros_like(x)

    denom = torch.zeros_like(x)
    denom.index_add_(0, index, pw)
    denom = denom.clamp_min(1e-6)

    result = result / denom
    out = result.index_select(0, index)

    if not apply_mask:
        return out

    return torch.where(mask_bool, out, torch.zeros_like(out))

def index_var(data: torch.Tensor,
              index: torch.Tensor,
              mask: torch.Tensor,
              apply_mask: bool = True):
    """Variance of array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Variance of array entries with the same index value, broadcasted
        to all entries with that index value.
    """
    ex2 = index_mean(data ** 2, index, mask, apply_mask=apply_mask)
    e2x = index_mean(data, index, mask, apply_mask=apply_mask) ** 2
    return ex2 - e2x

def index_std(data: torch.Tensor,
              index: torch.Tensor,
              mask: torch.Tensor,
              apply_mask: bool = True,
              eps: Optional[float] = 1e-6):
    """Standard deviation of array entries with the same index value.

    Args:
        data: data array of shape (N, ...).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Standard deviation of array entries with the same index value, broadcasted
        to all entries with that index value.
    """
    return torch.sqrt(index_var(data, index, mask, apply_mask) + eps)


def index_kabsch(x, y, index, mask, weight=None):
    device = x.device

    x_center = index_mean(x, index, mask[:, None], weight)
    y_center = index_mean(y, index, mask[:, None], weight)

    x0 = x - x_center
    y0 = y - y_center

    if weight is None:
        weight = torch.ones_like(index, dtype=x.dtype)

    cov = index_sum(
        weight[:, None, None] * x0[:, :, None] * y0[:, None, :],
        index, mask[:, None, None]
    ) 

    u, _, v = torch.linalg.svd(cov.detach(), full_matrices=True)
    det = torch.linalg.det(u) * torch.linalg.det(v)
    flip = torch.ones((cov.shape[0], 3), device=device)
    flip[:, -1] = det
    rot = torch.einsum("...ak,...kb->...ba", u * flip[:, None, :], v)

    return rot, x_center, y_center

def sequence_relative_position(count: Optional[int] = 32,
                               one_hot=False,
                               cyclic=False,
                               identify_ends=False,
                               pseudo_chains=False):
    """Compute sequence relative positions features for a protein chain.
    
    Args:
        count: returns separate features for signed distances from -count to +count.
        one_hot: return one-hot encoded features. Default: False.
        cyclic: cyclise one or more chains. Default: False.
        identify_ends: use the same representation for +count and -count. Default: False.
        pseudo_chains: represent distances across chains by +count or -count
            instead of a separate label. Default: False.
    Returns:
        A function computing relative position features given
        residue, chain and batch indices (N,), as well a neighbour array (N, K)
        and optionally a cyclic_mask (N,) which specifies which chains should
        be cyclised.
    """
    def inner(resi, chain, batch, neighbours=None, cyclic_mask=None):
        compare_index = (None, slice(None))
        if neighbours is not None:
            compare_index = neighbours
        same_chain = chain[:, None] == chain[compare_index]
        same_batch = batch[:, None] == batch[compare_index]
        dist = resi[:, None] - resi[compare_index]
        flat_resi = torch.arange(resi.shape[0], dtype=torch.int32, device=resi.device)
        if cyclic:
            lengths = index_count(chain, torch.ones_like(chain, dtype=torch.bool))
            wrap = abs(dist) > lengths[:, None] / 2
            # control cyclic wrapping per residue/chain
            if cyclic_mask is not None:
                wrap = wrap * cyclic_mask[:, None]
            dist = torch.where(
                wrap,
                torch.where(dist < 0,
                         dist % lengths[:, None],
                         dist % lengths[:, None] - lengths[:, None]),
                dist)
        dist = torch.clamp(dist, -count, count) + count
        if identify_ends:
            count_total = 2 * count - 2
            dist = torch.where(dist == 0, 2 * count - 2, dist - 1)
            dist = torch.where(same_chain, dist, 2 * count - 2)
            dist = torch.where(same_batch, dist, 2 * count - 2)
        elif pseudo_chains:
            flat_dist = flat_resi[:, None] - flat_resi[compare_index]
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

def index_count(index, mask, apply_mask=True):
    """Count the number of entries with the same index value.
    
    Args:
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        apply_mask: restrict the output to entries where mask is True. Default: True.
    Returns:
        Count of index entries with the same value, broadcasted
        to all entries with that index value.
        E.g. for index [0, 0, 0, 1, 1] the result would be [3, 3, 3, 2, 2]
    """
    index = index.long()
    mask = mask.bool()
    result = torch.zeros_like(index)
    result.scatter_add_(0, index, mask.long())
    if not apply_mask:
        return result[index]
    return torch.where(mask, result[index], torch.zeros_like(result[index]))

def index_align(x, y, index, mask, weight=None):
    """Rigid align two structures x and y.
    
    Args:
        x, y: atom positions of shape (N, ..., 3).
        index: integer index of shape (N,) with values between 0 and N-1.
        mask: boolean entry mask of shape (N,).
        weight: optional array of importance weights for biasing alignment. Default: None.
    Returns:
        x aligned to y.
    """
    return_vec3 = False
    if isinstance(x, Vec3Array):
        x = x.to_tensor()
        return_vec3 = True
    if isinstance(y, Vec3Array):
        y = y.to_tensor()
        return_vec3 = True
    rot, x_center, y_center = index_kabsch(
        x[:, 1], y[:, 1], index, mask, weight=weight)
    result = torch.einsum(
        "...ak,...ik->...ia", rot, (x - x_center[:, None])) + y_center[:, None]
    if return_vec3:
        result = Vec3Array.from_array(result)
    return result

def apply_alignment(x, kabsch_data):
    """Apply alignment parameters to a structure.
    
    Args:
        x: atom positions of shape (N, ..., 3).
        kabsch_data: output of index_kabsch.
    Returns:
        x transformed according to kabsch_data.
    """
    rot, x_center, y_center = kabsch_data
    delta = torch.einsum(
         "...ak,...k->...a", torch.swapaxes(rot, -1, -2), y_center) - x_center
    return_vec3 = False
    if isinstance(x, Vec3Array):
        x = x.to_tensor()
        return_vec3 = True
    result = torch.einsum(
        "...ak,...ik->...ia", rot, x + delta[:, None])
    if return_vec3:
        result = Vec3Array.from_array(result)
    return result

def unique_chain(chain, batch):
    """
    Compute a unique chain index given a batch of chains.

    Args:
        chain: [N] chain index
        batch: [N] batch index
    Returns:
        [N] unique chain index
    """
    N = chain.shape[0]

    change = torch.ones(N, dtype=torch.bool, device=chain.device)
    change[1:] = (chain[1:] != chain[:-1]) | (batch[1:] != batch[:-1])

    out = torch.cumsum(change.to(dtype=chain.dtype), dim=0) - 1
    return out

def positions_to_ncacocb(pos: torch.ndarray):
    """Compute N, CA, C, O, CB positions for atom14 positions.
    
    Args:
        pos: atom positions in atom14 format containing at least N, CA, C and O.
    Returns:
        Atom positions for N, CA, C, O and idealised CB.
    """
    cb = compute_pseudo_cb(pos)
    return torch.concatenate((pos[:, :4], cb[..., None, :]), dim=-2)

def replace_masked_with(pos: torch.ndarray, # (..., N, 3)
                        atom_mask: torch.ndarray, # (.... N)
                        replacement: torch.ndarray # (..., 3)
                       ) -> torch.ndarray: # (..., N, 3)
    """Replace masked atom positions with replacement positions."""
    return torch.where(atom_mask[..., None], pos, replacement)
