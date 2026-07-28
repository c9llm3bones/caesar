"""Native loss modules for the experimental backbone-only path."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from caesar.experimental.torch_native.types import ModelOutput, NeighbourSet, PreparedBatch, TrainingInputs
from caesar.experimental.torch_native.tensor_geometry import (
    distance_one_hot,
    gather_nodes,
    index_sum,
    invert_apply_frame,
    local_positions,
)


class AALoss(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = float(weight)

    def forward(self, prepared: PreparedBatch, output: ModelOutput) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aa_gt = prepared.raw.aa_gt.long()
        mask = prepared.mask
        aa_predict_mask = mask * (aa_gt != 20)
        aa_one_hot = F.one_hot(aa_gt.clamp(0, 19), num_classes=20).to(dtype=output.aa.dtype)
        valid = ((aa_gt >= 0) & (aa_gt < 20)).unsqueeze(-1)
        aa_one_hot = aa_one_hot * valid.to(dtype=output.aa.dtype)
        aa_nll = -(output.aa * aa_one_hot).sum(dim=-1)
        aa_nll = torch.where(aa_predict_mask.bool(), aa_nll, torch.zeros_like(aa_nll))
        aa_nll = aa_nll.sum() / torch.clamp(aa_predict_mask.to(dtype=aa_nll.dtype).sum(), min=1.0)
        weighted = self.weight * aa_nll
        return weighted, {"aa": aa_nll, "weighted_aa": weighted}


class FAPELoss(nn.Module):
    def __init__(self, *, weight: float, trajectory_weight: float, clip_fape: float):
        super().__init__()
        self.weight = float(weight)
        self.trajectory_weight = float(trajectory_weight)
        self.clip_fape = float(clip_fape)

    def forward(
        self,
        prepared: PreparedBatch,
        output: ModelOutput,
        neighbours: NeighbourSet,
        base_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = prepared.mask
        pos_gt = torch.where(mask[:, None, None].bool(), prepared.pos_gt, torch.zeros_like(prepared.pos_gt))
        rot_gt, trans_gt, _ = local_positions(pos_gt.detach())
        gathered_mask = gather_nodes(mask, neighbours.fape)
        mask_neighbours = (neighbours.fape != -1) * mask[:, None] * gathered_mask
        pos_gt_local = invert_apply_frame(
            rot_gt[:, None, None],
            trans_gt[:, None, None],
            gather_nodes(pos_gt.detach(), neighbours.fape),
        )

        traj = output.trajectory
        rot, trans, _ = local_positions(traj)
        safe = neighbours.fape.long().remainder(traj.shape[1])
        traj_neighbours = torch.index_select(traj, dim=1, index=safe.reshape(-1)).reshape(
            traj.shape[0], neighbours.fape.shape[0], neighbours.fape.shape[1], traj.shape[2], traj.shape[3]
        )
        traj_local = invert_apply_frame(rot[:, :, None, None], trans[:, :, None, None], traj_neighbours)
        fape_base = ((traj_local - pos_gt_local) ** 2).sum(dim=-1)
        fape = torch.clamp(fape_base, 0.0, self.clip_fape)
        fape_traj = fape.mean(dim=-1)
        fape_traj = torch.where(mask_neighbours[None].bool(), fape_traj, torch.zeros_like(fape_traj))
        fape_traj = fape_traj.sum(dim=-1) / torch.clamp(mask_neighbours.sum(dim=1)[None], min=1.0)
        fape_traj = (fape_traj * base_weight).sum(dim=-1)
        final = fape_traj[-1] / 3.0
        trajectory = fape_traj.mean() / 3.0
        weighted = self.weight * final + self.trajectory_weight * trajectory
        return weighted, {"fape": final, "fape_trajectory": trajectory, "weighted_fape": weighted}


class LocalAtomLoss(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = float(weight)

    def forward(
        self,
        prepared: PreparedBatch,
        output: ModelOutput,
        neighbours: NeighbourSet,
        base_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = prepared.mask
        mask_gt = gather_nodes(prepared.atom_mask, neighbours.local_atom)

        rot_gt, trans_gt, _ = local_positions(prepared.atom_pos)
        local_gt = invert_apply_frame(
            rot_gt[:, None, None],
            trans_gt[:, None, None],
            gather_nodes(prepared.atom_pos, neighbours.local_atom),
        )
        rot, trans, _ = local_positions(output.atom_pos)
        local_pred = invert_apply_frame(
            rot[:, None, None],
            trans[:, None, None],
            gather_nodes(output.atom_pos, neighbours.local_atom),
        )

        sq = ((local_pred - local_gt) ** 2).sum(dim=-1)
        local_loss = torch.where(mask_gt.bool(), sq, torch.zeros_like(sq)).sum(dim=(1, 2))
        local_loss = local_loss / torch.clamp(mask_gt.sum(dim=(1, 2)), min=1)
        local_loss = (torch.where(mask.bool(), local_loss, torch.zeros_like(local_loss)) * base_weight).sum() / 3.0
        weighted = self.weight * local_loss
        return weighted, {"local": local_loss, "weighted_local": weighted}


class DistogramLoss(nn.Module):
    def forward(
        self,
        prepared: PreparedBatch,
        training: TrainingInputs,
        output: ModelOutput,
        base_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = prepared.mask
        cb_gt = prepared.pos_gt[:, -1]
        gathered_mask = gather_nodes(mask, output.sup_neighbours)
        sup_mask = (output.sup_neighbours != -1) * mask[:, None] * gathered_mask
        dist_gt = torch.linalg.norm(cb_gt[:, None] - gather_nodes(cb_gt, output.sup_neighbours), dim=-1)
        dist_one_hot = distance_one_hot(dist_gt, 0.0, 22.0, 16)
        nll = -(output.sup_distogram * dist_one_hot[None]).sum(dim=-1)
        nll = torch.where(sup_mask.bool(), nll, torch.zeros_like(nll)).sum(dim=-1)
        nll = nll / torch.clamp(sup_mask.sum(dim=1), min=1.0)
        nll = (nll * base_weight).sum(dim=1)
        final = nll[-1]
        trajectory = nll.mean()
        weighted = 10.0 * final + 5.0 * trajectory
        return weighted, {"distogram": final, "distogram_trajectory": trajectory, "weighted_distogram": weighted}


class BackboneLoss(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.aa = AALoss(float(config.aa_weight))
        self.fape = FAPELoss(
            weight=float(config.fape_weight),
            trajectory_weight=float(config.fape_trajectory_weight),
            clip_fape=float(config.clip_fape),
        )
        self.local = LocalAtomLoss(float(config.local_weight))
        if config.distogram_block != "inner":
            raise ValueError("BackboneLoss v1 expects distogram_block='inner'")
        self.distogram = DistogramLoss()

    def forward(
        self,
        prepared: PreparedBatch,
        training: TrainingInputs,
        neighbours: NeighbourSet,
        output: ModelOutput,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask = prepared.mask
        batch = prepared.raw.batch_index.long()
        mask_bool = mask.bool()
        base_weight = mask / torch.clamp(index_sum(mask, batch, mask_bool), min=1.0) / (batch.max() + 1)

        total, losses = self.aa(prepared, output)
        part, extra = self.fape(prepared, output, neighbours, base_weight)
        total = total + part
        losses.update(extra)
        part, extra = self.local(prepared, output, neighbours, base_weight)
        total = total + part
        losses.update(extra)
        part, extra = self.distogram(prepared, training, output, base_weight)
        total = total + part
        losses.update(extra)
        losses["total"] = total
        return total, losses
