"""Compatibility facade for tensor-native geometric modules.

The original port kept JAX-style object wrappers in this file. The actual
implementations now live in ``caesar.experimental.torch_native`` and operate on
plain tensors. This module preserves the legacy import surface used by
``encoder.py``/``decoder.py`` while moving hot-path math to tensor-native code.
"""

from __future__ import annotations

import torch

from caesar.experimental.torch_native import tensor_geometry as _tg
from caesar.experimental.torch_native.native_geometric import (
    DenseNonEquivariantPointAttentionNative,
    LinearToPointsNative,
    SemiEquivariantSparseStructureAttentionNative,
    SparseAttentionNative,
    SparseInvariantMultiQueryAttentionNative,
    SparseInvariantPointAttentionNative,
    SparseSemiEquivariantPointAttentionNative,
    SparseStructureAttentionNative,
    SparseStructureMessageNative,
)
from caesar.utils.geometry import Rigid3Array, Vec3Array


def _to_tensor(x):
    return x.to_tensor() if hasattr(x, "to_tensor") else x


def _frame_to_tensor(frames):
    return frames.to_tensor() if hasattr(frames, "to_tensor") else frames


def extract_aa_frames(positions: Vec3Array | torch.Tensor) -> tuple[Rigid3Array, Vec3Array]:
    """Extract residue frames and local atom positions.

    Returns OpenFold-style wrappers for legacy callers, but computes the frame
    math through tensor-native kernels.
    """
    pos = _to_tensor(positions)
    rot, trans, local = _tg.local_positions(pos)
    frames = Rigid3Array.from_array4x4(_tg.frame_tensor_from_rot_trans(rot, trans))
    return frames, Vec3Array.from_array(local)


def sequence_relative_position(
    count: int | None = 32,
    one_hot: bool = False,
    cyclic: bool = False,
    identify_ends: bool = False,
    pseudo_chains: bool = False,
):
    return _tg.sequence_relative_position(
        count=count,
        one_hot=one_hot,
        cyclic=cyclic,
        identify_ends=identify_ends,
        pseudo_chains=pseudo_chains,
    )


def distance_features(pos, neighbours=None, d_min=0.0, d_max=22.0, num_rbf=16):
    return _tg.distance_features(_to_tensor(pos), neighbours, d_min=d_min, d_max=d_max, num_rbf=num_rbf)


def direction_features(pos, neighbours=None, d_min=0.0, d_max=22.0, num_rbf=16):
    del d_min, d_max, num_rbf
    return _tg.direction_features(_to_tensor(pos), neighbours)


def paired_distance_features(x, y, d_min=0.0, d_max=22.0, num_rbf=16):
    return _tg.paired_distance_features(_to_tensor(x), _to_tensor(y), d_min=d_min, d_max=d_max, num_rbf=num_rbf)


def rotation_features(frames, neighbours=None):
    if hasattr(frames, "rotation"):
        rot = frames.rotation.to_tensor()
    else:
        rot, _ = _tg.rot_trans_from_frame_tensor(frames)
    return _tg.rotation_features_from_frames(rot, neighbours)


def position_rotation_features(pos, neighbours=None):
    return _tg.position_rotation_features(_to_tensor(pos), neighbours)


def paired_rotation_features(x, y):
    return _tg.paired_rotation_features(_to_tensor(x), _to_tensor(y))


def pair_vector_features(pos, neighbours=None, scale: float = 0.1):
    return _tg.pair_vector_features(_to_tensor(pos), neighbours, scale=scale)


class LinearToPoints(LinearToPointsNative):
    def forward(self, data: torch.Tensor, frames):
        return super().forward(data, _frame_to_tensor(frames))


class SparseInvariantPointAttention(SparseInvariantPointAttentionNative):
    def forward(self, local, pair, frames, neighbours, mask):
        return super().forward(local, pair, _frame_to_tensor(frames), neighbours, mask)


class SparseInvariantMultiQueryAttention(SparseInvariantMultiQueryAttentionNative):
    def forward(self, local, pair, frames, neighbours, mask):
        return super().forward(local, pair, _frame_to_tensor(frames), neighbours, mask)


SparseStructureMessage = SparseStructureMessageNative
SparseStructureAttention = SparseStructureAttentionNative
SemiEquivariantSparseStructureAttention = SemiEquivariantSparseStructureAttentionNative
SparseAttention = SparseAttentionNative
DenseNonEquivariantPointAttention = DenseNonEquivariantPointAttentionNative
SparseSemiEquivariantPointAttention = SparseSemiEquivariantPointAttentionNative


__all__ = [
    "DenseNonEquivariantPointAttention",
    "LinearToPoints",
    "SemiEquivariantSparseStructureAttention",
    "SparseAttention",
    "SparseInvariantMultiQueryAttention",
    "SparseInvariantPointAttention",
    "SparseSemiEquivariantPointAttention",
    "SparseStructureAttention",
    "SparseStructureMessage",
    "direction_features",
    "distance_features",
    "extract_aa_frames",
    "paired_distance_features",
    "paired_rotation_features",
    "pair_vector_features",
    "position_rotation_features",
    "rotation_features",
    "sequence_relative_position",
]
