from typing import Optional, Union, Any

import torch
import numpy as np

from caesar.modules.basic import Linear, MLP, init_glorot, init_linear
from caesar.modules.transformer import Transition
from caesar.geometry import (
    Vec3Array,
    distance_rbf, extract_aa_frames, extract_neighbours,
    sequence_relative_position, get_neighbours,
    index_mean, axis_index
)

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
        dirs = local_pos.normalized().to_array()
    else:
        local_pos = frames[:, None, None].apply_inverse_to_point(pos[neighbours])
        dirs = local_pos.normalized().to_array()
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
        type_dir = type_pos.normalized().to_array().reshape(local.shape[0], size * 3)
        type_dist = distance_rbf(type_pos.norm(), 0.0, 22.0, 10).reshape(type_pos.shape[0], -1)
        type_pos = type_pos.to_array().reshape(type_pos.shape[0], size * 3) / scale
        return torch.concatenate((type_dir, type_dist, type_pos), axis=-1)
    # compute type weight
    base_type_weight = Linear(size, bias=False, initializer="linear")(local)
    base_type_weight = torch.nn.functional.gelu(base_type_weight)
    # local type positions
    type_weight = base_type_weight[neighbours]
    type_weight = torch.where((neighbours != -1)[..., None], type_weight, 0)
    pos = pos.to_array()
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
    rot = rot.to_array().reshape(*rot.shape, -1)
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
    rot = rot.to_array().reshape(*rot.shape, -1)
    return rot

def paired_rotation_features(x, y):
    """Compute distance features for a pair of structures x and y.
    
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
    return rot.to_array().reshape(*rot.shape, -1)

def pair_vector_features(pos, neighbours=None, scale=0.1):
    """Compute local coordinate features.
    
    Args:
        pos: residue atom positions of shape (N, atoms, 3).
        neighbours (Optional[array]): neighbours of each residue
        scale: position scale factor. Default: 0.1.
    Returns:
        Coordinate features of atoms of a residue and its neighbours
        in the residue's local frame.
    """
    if neighbours is None:
        neighbours = jnp.broadcast_to(
            jnp.arange(pos.shape[0], dtype=jnp.int32)[None, :],
            (pos.shape[0], pos.shape[0]))
    frames, _ = extract_aa_frames(pos)
    pair_vectors = jnp.concatenate((
        jnp.broadcast_to(
            frames[:, None, None].apply_inverse_to_point(
                pos[:, None]).to_array(),
            (neighbours.shape[0], neighbours.shape[1], pos.shape[1], 3)),
        frames[:, None, None].apply_inverse_to_point(pos[neighbours]).to_array(),
    ), axis=-2)
    pair_vectors = Vec3Array.from_array(pair_vectors)
    direction = pair_vectors.normalized().to_array().reshape(
        *pair_vectors.shape[:-1], -1)
    result = jnp.concatenate((
        direction,
        scale * pair_vectors.to_array().reshape(
            *pair_vectors.shape[:-1], -1)
    ), axis=-1)
    return result

# TODO: Implement SparseStructureAttention in PyTorch
# (Currently JAX-only code, not yet ported)
# class SparseStructureAttention(hk.Module):
#     """Sparse attention wrapper."""
#     def __init__(self, config, normalize=True,
#                  name: Optional[str] = "sparse_structure_attn"):
#         super().__init__(name)
#         self.normalize = normalize
#         self.config = config
# 
#     def __call__(self, local, pos, pair, pair_mask, neighbours,
#                  resi, chain, batch, mask):
#         c = self.config
#         final_init = c.update_init if c.update_init else "zeros"
#         frames, _ = extract_aa_frames(Vec3Array.from_array(pos))
#         if c.multi_query:
#             local_update = SparseInvariantMultiQueryAttention(
#                 heads=c.heads, size=c.key_size,
#                 final_init=final_init, normalize=self.normalize)(
#                 local, pair, frames.to_array(),
#                 neighbours, pair_mask)
#         else:
#             local_update = SparseInvariantPointAttention(
#                 heads=c.heads, size=c.key_size,
#                 final_init=final_init, normalize=self.normalize)(
#                 local, pair, frames.to_array(),
#                 neighbours, pair_mask)
#         return local_update
