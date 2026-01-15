"""PyTorch geometry utilities for protein structures.

Adapted from OpenFold (https://github.com/aqlaboratory/openfold)
and SALAD (https://github.com/mjendrusch/salad)
Provides Vec3Array and rotation utilities needed for structure manipulation.
"""
# IN PROGRESS
from __future__ import annotations

import dataclasses
from random import uniform
from typing import Any, Optional, Union, Tuple, Iterable, Dict, List

import torch
import numpy as np

import caesar.utils.geometry
from caesar.aflib.common import residue_constants
from caesar.utils import geometry
from caesar.utils.all_atom_multimer import (
    make_transform_from_reference, torsion_angles_to_frames,
    frames_and_literature_positions_to_atom14_pos)

Float = Union[float, torch.Tensor]

@dataclasses.dataclass(frozen=True)
class Vec3Array:
    x: torch.Tensor = dataclasses.field(metadata={'dtype': torch.float32})
    y: torch.Tensor
    z: torch.Tensor

    def __post_init__(self):
        if hasattr(self.x, 'dtype'):
            assert self.x.dtype == self.y.dtype
            assert self.x.dtype == self.z.dtype
            assert all([x == y for x, y in zip(self.x.shape, self.y.shape)])
            assert all([x == z for x, z in zip(self.x.shape, self.z.shape)])

    def __add__(self, other: Vec3Array) -> Vec3Array:
        return Vec3Array(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Vec3Array) -> Vec3Array:
        return Vec3Array(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, other: Float) -> Vec3Array:
        return Vec3Array(
            self.x * other,
            self.y * other,
            self.z * other,
        )

    def __rmul__(self, other: Float) -> Vec3Array:
        return self * other

    def __truediv__(self, other: Float) -> Vec3Array:
        return Vec3Array(
            self.x / other,
            self.y / other,
            self.z / other,
        )

    def __neg__(self) -> Vec3Array:
        return self * -1 

    def __pos__(self) -> Vec3Array:
        return self * 1

    def __getitem__(self, index) -> Vec3Array:
        return Vec3Array(
            self.x[index],
            self.y[index],
            self.z[index],
        )

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    @property
    def shape(self):
        return self.x.shape

    def map_tensor_fn(self, fn) -> Vec3Array:
        return Vec3Array(
            fn(self.x),
            fn(self.y),
            fn(self.z),
        )
        
    def cross(self, other: Vec3Array) -> Vec3Array:
        """Compute cross product between 'self' and 'other'."""
        new_x = self.y * other.z - self.z * other.y
        new_y = self.z * other.x - self.x * other.z
        new_z = self.x * other.y - self.y * other.x
        return Vec3Array(new_x, new_y, new_z)

    def dot(self, other: Vec3Array) -> Float:
        """Compute dot product between 'self' and 'other'."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self, epsilon: float = 1e-6) -> Float:
        """Compute Norm of Vec3Array, clipped to epsilon."""
        # To avoid NaN on the backward pass, we must use maximum before the sqrt
        norm2 = self.dot(self)
        if epsilon:
            norm2 = torch.clamp(norm2, min=epsilon**2)
        return torch.sqrt(norm2)

    def norm2(self):
        return self.dot(self)

    def normalized(self, epsilon: float = 1e-6) -> Vec3Array:
        """Return unit vector with optional clipping."""
        return self / self.norm(epsilon)

    def clone(self) -> Vec3Array:
        return Vec3Array(
            self.x.clone(),
            self.y.clone(),
            self.z.clone(),
        )

    def reshape(self, new_shape) -> Vec3Array:
        x = self.x.reshape(new_shape)
        y = self.y.reshape(new_shape)
        z = self.z.reshape(new_shape)

        return Vec3Array(x, y, z)

    def sum(self, dim: int) -> Vec3Array:
        return Vec3Array(
            torch.sum(self.x, dim=dim),
            torch.sum(self.y, dim=dim),
            torch.sum(self.z, dim=dim),
        )

    def unsqueeze(self, dim: int):
        return Vec3Array(
            self.x.unsqueeze(dim),
            self.y.unsqueeze(dim),
            self.z.unsqueeze(dim),
        )

    @classmethod
    def zeros(cls, shape, device="cpu"):
        """Return Vec3Array corresponding to zeros of given shape."""
        return cls(
            torch.zeros(shape, dtype=torch.float32, device=device), 
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(shape, dtype=torch.float32, device=device)
        )

    def to_tensor(self) -> torch.Tensor:
        return torch.stack([self.x, self.y, self.z], dim=-1)

    @classmethod
    def from_array(cls, tensor):
        return cls(*torch.unbind(tensor, dim=-1))

    @classmethod
    def cat(cls, vecs: List[Vec3Array], dim: int) -> Vec3Array:
        return cls(
            torch.cat([v.x for v in vecs], dim=dim),
            torch.cat([v.y for v in vecs], dim=dim),
            torch.cat([v.z for v in vecs], dim=dim),
        )


def square_euclidean_distance(
    vec1: Vec3Array,
    vec2: Vec3Array,
    epsilon: float = 1e-6
) -> Float:
    """Computes square of euclidean distance between 'vec1' and 'vec2'.

    Args:
        vec1: Vec3Array to compute    distance to
        vec2: Vec3Array to compute    distance from, should be
                    broadcast compatible with 'vec1'
        epsilon: distance is clipped from below to be at least epsilon

    Returns:
        Array of square euclidean distances;
        shape will be result of broadcasting 'vec1' and 'vec2'
    """
    difference = vec1 - vec2
    distance = difference.dot(difference)
    if epsilon:
        distance = torch.clamp(distance, min=epsilon)
    return distance


def dot(vector1: Vec3Array, vector2: Vec3Array) -> Float:
    return vector1.dot(vector2)


def cross(vector1: Vec3Array, vector2: Vec3Array) -> Float:
    return vector1.cross(vector2)


def norm(vector: Vec3Array, epsilon: float = 1e-6) -> Float:
    return vector.norm(epsilon)


def normalized(vector: Vec3Array, epsilon: float = 1e-6) -> Vec3Array:
    return vector.normalized(epsilon)


def euclidean_distance(
    vec1: Vec3Array,
    vec2: Vec3Array,
    epsilon: float = 1e-6
) -> Float:
    """Computes euclidean distance between 'vec1' and 'vec2'.

    Args:
        vec1: Vec3Array to compute euclidean distance to
        vec2: Vec3Array to compute euclidean distance from, should be
                    broadcast compatible with 'vec1'
        epsilon: distance is clipped from below to be at least epsilon

    Returns:
        Array of euclidean distances;
        shape will be result of broadcasting 'vec1' and 'vec2'
    """
    distance_sq = square_euclidean_distance(vec1, vec2, epsilon**2)
    distance = torch.sqrt(distance_sq)
    return distance


def dihedral_angle(a: Vec3Array, b: Vec3Array, c: Vec3Array,
                                     d: Vec3Array) -> Float:
    """Computes torsion angle for a quadruple of points.

    For points (a, b, c, d), this is the angle between the planes defined by
    points (a, b, c) and (b, c, d). It is also known as the dihedral angle.

    Arguments:
        a: A Vec3Array of coordinates.
        b: A Vec3Array of coordinates.
        c: A Vec3Array of coordinates.
        d: A Vec3Array of coordinates.

    Returns:
        A tensor of angles in radians: [-pi, pi].
    """
    v1 = a - b
    v2 = b - c
    v3 = d - c

    c1 = v1.cross(v2)
    c2 = v3.cross(v2)
    c3 = c2.cross(c1)

    v2_mag = v2.norm()
    return torch.atan2(c3.dot(v2), v2_mag * c1.dot(c2))

# code above is exactly openfold/utils/geometry/vector.py

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

    rigid_mask = (mask[..., a] * mask[..., b] * mask[..., c]).astype(
        torch.float32)

    rigid = make_transform_from_reference(
        a_xyz=positions[..., a],
        b_xyz=positions[..., b],
        c_xyz=positions[..., c])

    return rigid, rigid_mask

def make_transform_from_reference(
    a_xyz: geometry.Vec3Array,
    b_xyz: geometry.Vec3Array,
    c_xyz: geometry.Vec3Array) -> geometry.Rigid3Array:
  """Returns rotation and translation matrices to convert from reference.

  Note that this method does not take care of symmetries. If you provide the
  coordinates in the non-standard way, the A atom will end up in the negative
  y-axis rather than in the positive y-axis. You need to take care of such
  cases in your code.

  Args:
    a_xyz: A Vec3Array.
    b_xyz: A Vec3Array.
    c_xyz: A Vec3Array.

  Returns:
    A Rigid3Array which, when applied to coordinates in a canonicalized
    reference frame, will give coordinates approximately equal
    the original coordinates (in the global frame).
  """
  rotation = geometry.Rot3Array.from_two_vectors(c_xyz - b_xyz,
                                                 a_xyz - b_xyz)
  return geometry.Rigid3Array(rotation, b_xyz)

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
        flat_resi = torch.arange(resi.shape[0], dtype=torch.int32)
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
            dist = torch.nn.functional.one_hot(dist, count_total, axis=-1)
        return dist
    return inner

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

def get_random_neighbours(count: int):
    """Extracts `count` neighbours with probability 1 / d^3."""
    def inner(pos: Any, item, mask, neighbours=None):
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
        uniform = torch.empty_like(weight).uniform_(1e-6, 1.0 - 1e-6) #(c) avoid log(0)
        gumbel = torch.log(-torch.log(uniform))
        weight = weight - gumbel
        distance = -weight
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
    cos_tau = (left * right).sum(dim=-1) / torch.maximum(torch.linalg.norm(left, dim=-1) * torch.linalg.norm(right, dim=-1), 1e-6)
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
    result = torch.arctan2(y_norm * (x * torch.cross(y, z)).sum(dim=-1),
                         (torch.cross(x, y) * torch.cross(y, z)).sum(dim=-1))
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

        knn = torch.argsort(distance, dim=-1, stable=True)[..., :count]

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
    def inner(pos: Any, item, mask, neighbours=None):
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
        uniform = torch.rand(weight.shape, dtype=weight.dtype, device=weight.device)
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
    """Computes Gaussian RBF features of continuous inputs.
    
    Args:
        distance: array of distances to embed.
        min_distance: minimum distance to place RBFs.
        max_distance: maximum distance to place RBFs.
        bins: number of radial basis functions.

    Returns:
        Gaussian RBF embedding of `distance` with `bins` centers.
    """
    step = (max_distance - min_distance) / bins
    centers = min_distance + torch.arange(bins) * step + step / 2
    rbf = torch.exp(-(distance[..., None] - centers) ** 2 / step ** 2)
    return rbf

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
    centers = min_distance + torch.arange(bins) * step + step / 2
    argmin = torch.argmin(abs(distance[..., None] - centers), dim=-1)
    return torch.nn.functional.one_hot(argmin, bins, dim=-1)


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
    lower = torch.arange(bins) * step
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
    a = torch.cross(b, c)
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
    return torch.arange(data.shape[dim], dtype=torch.int32)

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
    data = torch.where(mask, data, 0)
    result = torch.zeros_like(data).at[index].add(data)
    if not apply_mask:
        return result[index]
    return torch.where(mask, result[index], 0)

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
    dmin = data.min()
    data = torch.where(mask, data, dmin)
    result = torch.full_like(data, dmin).at[index].max(data)
    if not apply_mask:
        return result[index]
    return torch.where(mask, result[index], dmin)

# Helper functions to broadcast mask and weight to data shape
def _broadcast_mask_to_data(mask: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    Make mask broadcastable to data by adding singleton dims.
    jax allows mask (N,) or (N,1) etc; we emulate that.
    """
    # ensure mask starts with N dimension
    if mask.dim() == 1 and data.dim() > 1:
        mask = mask.view(mask.shape[0], *([1] * (data.dim() - 1)))
    elif mask.dim() < data.dim():
        mask = mask.view(*mask.shape, *([1] * (data.dim() - mask.dim())))
    return mask

def _broadcast_weight_to_data(weight: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """
    Broadcast weight to match data shape (same logic as mask).
    """
    if weight.dim() == 1 and data.dim() > 1:
        weight = weight.view(weight.shape[0], *([1] * (data.dim() - 1)))
    elif weight.dim() < data.dim():
        weight = weight.view(*weight.shape, *([1] * (data.dim() - weight.dim())))
    return weight

def index_mean(data, index, mask, weight=None, apply_mask=True):
    if index.dtype != torch.long:
        index = index.long()

    # mask -> bool
    mask_bool = (mask != 0) if mask.dtype != torch.bool else mask

    # broadcast mask для torch.where
    mask_b = _broadcast_mask_to_data(mask_bool, data)

    x = data
    if weight is not None:
        w = weight if torch.is_tensor(weight) else torch.as_tensor(weight, device=data.device)
        w = w.to(device=data.device, dtype=data.dtype)
        w = _broadcast_weight_to_data(w, data)              # -> (N,1) or (N,...) broadcastable
        w = w.expand_as(data)                               
        x = x * w

    x = torch.where(mask_b, x, torch.zeros_like(x))

    result = torch.zeros_like(x)
    result.index_add_(0, index, x)

    if weight is None:
        position_weight = mask_bool.to(dtype=data.dtype, device=data.device)
        position_weight = _broadcast_weight_to_data(position_weight, data) 
        position_weight = position_weight.expand_as(data)                   
    else:
        w = weight.to(device=data.device, dtype=data.dtype) if torch.is_tensor(weight) else torch.as_tensor(weight, device=data.device, dtype=data.dtype)
        w = _broadcast_weight_to_data(w, data)
        w = w.expand_as(data)                                               # !!! (N,3)
        position_weight = torch.where(mask_b, w, torch.zeros_like(w))

    denom = torch.zeros_like(result)
    denom.index_add_(0, index, position_weight)                            
    denom = torch.clamp(denom, min=1e-6)

    result = result / denom
    out = result[index]

    if not apply_mask:
        return out

    out = torch.where(mask_b, out, torch.zeros_like(out))
    return out

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
    result = torch.zeros_like(index).at[index].add(mask.astype(index.dtype))
    if not apply_mask:
        return result[index]
    return torch.where(mask, result[index], 0)

def index_kabsch(x, y, index, mask, weight=None):
    device = x.device

    x_center = index_mean(x, index, mask[:, None], weight)
    y_center = index_mean(y, index, mask[:, None], weight)

    x0 = x - x_center[index]
    y0 = y - y_center[index]

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

# meow meow meow 


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
        x = x.to_array()
        return_vec3 = True
    if isinstance(y, Vec3Array):
        y = y.to_array()
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
        x = x.to_array()
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
    device = chain.device

    out = torch.zeros(N, dtype=chain.dtype, device=device)

    prev_chain = torch.tensor(-1, dtype=chain.dtype, device=device)
    prev_batch = torch.tensor(-1, dtype=batch.dtype, device=device)
    current = torch.tensor(-1, dtype=chain.dtype, device=device)

    for i in range(N):
        if (chain[i] != prev_chain) or (batch[i] != prev_batch):
            current += 1
        out[i] = current
        prev_chain = chain[i]
        prev_batch = batch[i]

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

def assign_sse(pos, batch, mask):
    """Assign secondary structure using P-SEA.
    
    Implements the secondary structure assignment from Labesse et al. 1997
    (pubmed.ncbi.nlm.nih.gov/9183534/)

    Args:
        pos: atom positions in atom14 format, containing at least CA atoms.
        batch: batch index.
        mask: residue mask.
    Returns:
        3-state secondary structure assignment (0: loop, 1: helix, 2: strand);
        secondary structure blocks; block adjacency matrix.
    """
    device = pos.device
    pos = pos[:, 1]  # CA
    N = pos.shape[0]

    z1 = torch.zeros(1, device=device)
    z2 = torch.zeros(2, device=device)
    z3 = torch.zeros(3, device=device)

    d2 = torch.cat((z1, torch.linalg.norm(pos[2:] - pos[:-2], dim=-1), z1))
    d3 = torch.cat((z1, torch.linalg.norm(pos[3:] - pos[:-3], dim=-1), z2))
    d4 = torch.cat((z1, torch.linalg.norm(pos[4:] - pos[:-4], dim=-1), z3))

    tau = torch.cat((z1, bond_angle(pos[:-2], pos[1:-1], pos[2:]), z1))
    alpha = torch.cat((z1, dihedral_angle(pos[:-3], pos[1:-2], pos[2:-1], pos[3:]), z2))

    helix_tau = (77 <= tau) & (tau <= 101)
    helix_alpha = (30 <= alpha) & (alpha <= 70)
    helix_d3 = (4.8 <= d3) & (d3 <= 5.8)
    helix_d4 = (5.8 <= d4) & (d4 <= 7.0)

    sheet_tau = (110 <= tau) & (tau <= 138)
    sheet_alpha = (-215 <= alpha) & (alpha <= -125)
    sheet_d2 = (6.1 <= d2) & (d2 <= 7.3)
    sheet_d3 = (9.0 <= d3) & (d3 <= 10.8)
    sheet_d4 = (11.3 <= d4) & (d4 <= 13.5)

    helix_init = (helix_tau & helix_alpha) | (helix_d3 & helix_d4)
    helix_extend = helix_init | helix_tau | helix_d3

    sheet_init = (sheet_tau & sheet_alpha) | (sheet_d2 & sheet_d3 & sheet_d4)
    sheet_extend = sheet_init | sheet_d3

    index = torch.zeros(N, dtype=torch.long, device=device)
    carry = 0

    for i in range(N):
        if helix_init[i] or (helix_extend[i] and carry == 1):
            index[i] = 1
        elif sheet_init[i] or (sheet_extend[i] and carry == 2):
            index[i] = 2
        carry = index[i].item()

    blocks = torch.zeros(N, dtype=torch.long, device=device)
    bid = 0
    for i in range(1, N):
        if index[i] != index[i - 1]:
            bid += 1
        blocks[i] = bid

    loop = index == 0
    pair_mask = (batch[:, None] == batch[None, :]) & mask[:, None] & mask[None, :]

    dist = torch.linalg.norm(pos[:, None] - pos[None, :], dim=-1)
    block_dist = torch.full((bid + 1, bid + 1), 1e6, device=device)

    for i in range(N):
        for j in range(N):
            bi, bj = blocks[i], blocks[j]
            block_dist[bi, bj] = min(block_dist[bi, bj], dist[i, j])

    block_adjacency = block_dist[blocks[:, None], blocks[None, :]] <= 8
    block_adjacency &= ~(blocks[:, None] == blocks[None, :])
    block_adjacency &= ~loop[:, None] & ~loop[None, :]
    block_adjacency &= pair_mask

    return index, blocks, block_adjacency

POLAR_THRESHOLD = 3.0
CONTACT_THRESHOLD = 6.0

def unit_sphere(n):
    """Generates n points on the surface of the unit sphere."""
    dl = np.pi * (3 - 5 ** 0.5)
    dz = 2.0 / n

    indices = np.arange(n)
    z = 1 - dz / 2 - indices * dz
    longitude = indices * dl
    r = (1 - z ** 2) ** 0.5
    coords = np.stack((
        np.cos(longitude) * r,
        np.sin(longitude) * r,
        z), axis=-1)
    return coords

# Only define ATOM14_RADIUS if residue_constants is available
ATOM14_RADIUS=np.array([
    [
        residue_constants.van_der_waals_radius[c[0]] + 1.4
        if c else 0.0
        for c in residue_constants.restype_name_to_atom14_names[res]
    ]
    for res in residue_constants.restype_name_to_atom14_names
])

def fast_sasa(pos, atom_mask, aatype, batch, atom_radius=ATOM14_RADIUS, n=20, neighbours=20):
    """Quick and dirty SASA estimate.
    
    Args:
        pos: atom positions in atom14 format.
        atom_mask: atom mask.
        aatype: amino acid identity for each residue.
        batch: batch_index.
        atom_radius: dictionary of atom radii. Default: ATOM14_RADIUS.
        n: number of points on each unit sphere.
        neighbours: number of neighbour amino acids for SASA computation.
    Returns:
        Per residue SASA estimate.
    """
    mask = atom_mask.any(axis=1)
    valid = (batch[:, None] == batch[None, :]) * (mask[:, None] * mask[None, :])
    radius = atom_radius[aatype] * atom_mask
    spheres = pos[:, :, None, :] + radius[:, :, None, None] * unit_sphere(n)[None, None, :, :]
    aa_dist = torch.linalg.norm(pos[:, None, 1] - pos[None, :, 1], axis=-1)
    aa_dist = torch.where(valid, aa_dist, torch.inf)
    neighbours = torch.argsort(aa_dist, axis=1)[:, :neighbours]
    index = axis_index(neighbours, 0)
    neighbours = torch.where(valid[index[:, None], neighbours], neighbours, -1)
    distances = torch.linalg.norm(spheres[:, None, :, None, :, :] - pos[neighbours, None, :, None, :], axis=-1)
    drop = (distances < radius[neighbours, None, :, None])
    index = torch.arange(atom_radius.shape[0])
    drop = drop.at[:, 0, index, index].set(0)
    drop = drop.any(axis=(1, 3))
    count = n * atom_mask - (atom_mask[..., None] * drop).sum(axis=2)
    bare_radius = 4 * torch.pi * radius ** 2
    surf = bare_radius * count / n
    surf = surf.sum(axis=1) / atom_mask.sum(axis=1)
    return surf
