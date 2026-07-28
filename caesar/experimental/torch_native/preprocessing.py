"""Preprocessing split for the experimental Torch-native backbone path."""

from __future__ import annotations

import torch

from caesar.experimental.torch_native.types import PreparedBatch, RawBatch, TrainingInputs
from caesar.experimental.torch_native.tensor_geometry import (
    compute_pseudo_cb,
    positions_to_ncacocb,
    unique_chain,
)
from caesar.utils.all_atom_multimer import atom37_to_atom14


def prepare_deterministic(raw: RawBatch) -> PreparedBatch:
    """Prepare deterministic atom14/backbone tensors without sampling noise."""
    pos = raw.all_atom_positions
    atom_mask = raw.all_atom_mask
    if not pos.dtype.is_floating_point:
        pos = pos.float()
    dtype = pos.dtype
    atom_mask = atom_mask.to(device=pos.device, dtype=dtype)

    if pos.shape[-2] == 37:
        pos, atom_mask = atom37_to_atom14(raw.aa_gt.long(), pos, atom_mask)
    else:
        pos = pos[:, :14]
        atom_mask = atom_mask[:, :14]

    batch = raw.batch_index.long()
    chain = unique_chain(raw.chain_index.long(), batch)

    seq_mask = raw.seq_mask.to(device=pos.device, dtype=dtype)
    residue_mask = raw.residue_mask.to(device=pos.device, dtype=dtype)
    atom_mask_bool = atom_mask.bool()
    mask_bool = seq_mask.bool() & residue_mask.bool() & atom_mask_bool[:, :3].all(dim=-1)
    mask = mask_bool.to(dtype)

    center = _index_mean(pos[:, 1], batch, atom_mask[:, 1, None])
    pos = pos - center[:, None, :]

    pseudo_cb = compute_pseudo_cb(pos)
    pos = torch.where(atom_mask_bool[..., None], pos, pseudo_cb[:, None, :].expand_as(pos))

    pos_gt = positions_to_ncacocb(pos)
    dmap_mask = batch[:, None] == batch[None, :]

    return PreparedBatch(
        raw=raw,
        pos_gt=pos_gt,
        pos_input=pos_gt,
        atom_pos=pos,
        atom_mask=atom_mask_bool.to(dtype),
        mask=mask,
        mask_bool=mask_bool,
        chain_index=chain,
        dmap_mask=dmap_mask,
        centered_all_atom_positions=pos,
        centered_all_atom_mask=atom_mask_bool.to(dtype),
    )


def prepare_training_inputs(
    prepared: PreparedBatch,
    *,
    generator: torch.Generator | None = None,
    no_random: bool = False,
    dmap_noise_scale: float = 0.3,
    recycle_count: int = 0,
) -> TrainingInputs:
    """Prepare stochastic training tensors outside model forward."""
    if no_random:
        pos_init = prepared.pos_gt
        cb = prepared.pos_gt[:, -1]
    else:
        pos_init = torch.randn(
            prepared.pos_gt.shape,
            device=prepared.device,
            dtype=prepared.dtype,
            generator=generator,
        )
        cb = prepared.pos_gt[:, -1] + float(dmap_noise_scale) * torch.randn(
            (prepared.num_residues, 3),
            device=prepared.device,
            dtype=prepared.dtype,
            generator=generator,
        )
    dmap = torch.linalg.norm(cb[:, None] - cb[None, :], dim=-1)
    return TrainingInputs(pos_init=pos_init, dmap=dmap, recycle_count=int(recycle_count))


def _index_mean(data: torch.Tensor, index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if index.dtype != torch.long:
        index = index.long()
    count = int(index.max().item()) + 1 if index.numel() else 0
    if count == 0:
        return torch.empty_like(data)
    weight = mask.to(dtype=data.dtype)
    weighted = data * weight
    out = torch.zeros((count, *data.shape[1:]), device=data.device, dtype=data.dtype)
    den = torch.zeros((count, *weight.shape[1:]), device=data.device, dtype=data.dtype)
    out.index_add_(0, index, weighted)
    den.index_add_(0, index, weight)
    mean = out / torch.clamp(den, min=1.0)
    return mean.index_select(0, index)
