"""Stable tensor contracts for the experimental Torch-native path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


def _as_tensor(value: Any, *, device: torch.device | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value if device is None else value.to(device=device)
    return torch.as_tensor(value, device=device)


@dataclass(frozen=True)
class RawBatch:
    all_atom_positions: torch.Tensor
    all_atom_mask: torch.Tensor
    aa_gt: torch.Tensor
    residue_index: torch.Tensor
    chain_index: torch.Tensor
    batch_index: torch.Tensor
    seq_mask: torch.Tensor
    residue_mask: torch.Tensor

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        device: torch.device | None = None,
    ) -> "RawBatch":
        return cls(
            all_atom_positions=_as_tensor(data["all_atom_positions"], device=device),
            all_atom_mask=_as_tensor(data["all_atom_mask"], device=device),
            aa_gt=_as_tensor(data["aa_gt"], device=device).long(),
            residue_index=_as_tensor(data["residue_index"], device=device).long(),
            chain_index=_as_tensor(data["chain_index"], device=device).long(),
            batch_index=_as_tensor(data["batch_index"], device=device).long(),
            seq_mask=_as_tensor(data["seq_mask"], device=device),
            residue_mask=_as_tensor(data["residue_mask"], device=device),
        )

    def to_legacy_dict(self) -> dict[str, torch.Tensor]:
        return {
            "all_atom_positions": self.all_atom_positions,
            "all_atom_mask": self.all_atom_mask,
            "aa_gt": self.aa_gt,
            "residue_index": self.residue_index,
            "chain_index": self.chain_index,
            "batch_index": self.batch_index,
            "seq_mask": self.seq_mask,
            "residue_mask": self.residue_mask,
        }


@dataclass(frozen=True)
class PreparedBatch:
    raw: RawBatch
    pos_gt: torch.Tensor
    pos_input: torch.Tensor
    atom_pos: torch.Tensor
    atom_mask: torch.Tensor
    mask: torch.Tensor
    mask_bool: torch.Tensor
    chain_index: torch.Tensor
    dmap_mask: torch.Tensor
    centered_all_atom_positions: torch.Tensor
    centered_all_atom_mask: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.pos_gt.device

    @property
    def dtype(self) -> torch.dtype:
        return self.pos_gt.dtype

    @property
    def num_residues(self) -> int:
        return int(self.pos_gt.shape[0])

    def to_legacy_base_dict(self) -> dict[str, torch.Tensor]:
        data = self.raw.to_legacy_dict()
        data.update(
            {
                "pos_gt": self.pos_gt,
                "pos_input": self.pos_input,
                "chain_index": self.chain_index,
                "mask_bool": self.mask_bool,
                "mask": self.mask,
                "atom_pos": self.atom_pos,
                "atom_mask": self.atom_mask,
                "all_atom_positions": self.centered_all_atom_positions,
                "all_atom_mask": self.centered_all_atom_mask,
                "dmap_mask": self.dmap_mask,
            }
        )
        return data


@dataclass(frozen=True)
class TrainingInputs:
    pos_init: torch.Tensor
    dmap: torch.Tensor
    recycle_count: int = 0


@dataclass(frozen=True)
class NeighbourSet:
    fape: torch.Tensor
    local_atom: torch.Tensor


@dataclass(frozen=True)
class ModelOutput:
    latent: torch.Tensor
    trajectory: torch.Tensor
    sup_neighbours: torch.Tensor
    sup_distogram: torch.Tensor
    local: torch.Tensor
    pos: torch.Tensor
    aa: torch.Tensor
    aa_features: torch.Tensor
    corrupt_aa: torch.Tensor
    aa_gt: torch.Tensor
    raw_angles: torch.Tensor
    angles: torch.Tensor
    atom_pos: torch.Tensor

    def to_legacy_result_dict(self) -> dict[str, torch.Tensor]:
        return {
            "latent": self.latent,
            "trajectory": self.trajectory,
            "sup_neighbours": self.sup_neighbours,
            "sup_distogram": self.sup_distogram,
            "local": self.local,
            "pos": self.pos,
            "aa": self.aa,
            "aa_features": self.aa_features,
            "corrupt_aa": self.corrupt_aa,
            "aa_gt": self.aa_gt,
            "raw_angles": self.raw_angles,
            "angles": self.angles,
            "atom_pos": self.atom_pos,
        }

