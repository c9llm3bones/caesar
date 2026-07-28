"""Backbone-only Torch-native experimental autoencoder."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from caesar.experimental.torch_native.losses import BackboneLoss
from caesar.experimental.torch_native.native_decoder import (
    DecoderNative,
    prepare_decoder_features_native,
)
from caesar.experimental.torch_native.native_encoder import EncoderNative
from caesar.experimental.torch_native.neighbours import prepare_neighbours
from caesar.experimental.torch_native.preprocessing import prepare_deterministic, prepare_training_inputs
from caesar.experimental.torch_native.types import ModelOutput, NeighbourSet, PreparedBatch, RawBatch, TrainingInputs


class BackboneAutoencoderNative(nn.Module):
    """Experimental native backbone path with explicit batch contracts.

    This v1 intentionally supports the small backbone autoencoder slice only.
    The legacy `StructureAutoencoder` remains the reference implementation for
    full SALAD-compatible behaviour.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = _native_backbone_config(config)
        self.predict_angles_from_logits = bool(getattr(self.config, "eval", False))
        self.select_angle_aatype = self._select_predicted_aatype if self.predict_angles_from_logits else self._select_ground_truth_aatype
        self.encoder = EncoderNative(self.config)
        self.decoder = DecoderNative(self.config)
        self.loss_fn = BackboneLoss(self.config)

    def prepare(
        self,
        batch: RawBatch | dict[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
        recycle_count: int = 0,
        no_random: bool = False,
    ) -> tuple[PreparedBatch, TrainingInputs, NeighbourSet]:
        raw = batch if isinstance(batch, RawBatch) else RawBatch.from_dict(batch)
        prepared = prepare_deterministic(raw)
        training = prepare_training_inputs(
            prepared,
            generator=generator,
            recycle_count=recycle_count,
            no_random=no_random,
        )
        neighbours = prepare_neighbours(
            prepared,
            training.dmap,
            fape_count=int(self.config.fape_neighbours),
            local_count=int(self.config.local_neighbours),
        )
        return prepared, training, neighbours

    def forward(
        self,
        batch: RawBatch | PreparedBatch | dict[str, torch.Tensor],
        training: TrainingInputs | None = None,
        neighbours: NeighbourSet | None = None,
        *,
        generator: torch.Generator | None = None,
        recycle_count: int | None = None,
        return_trace: bool = False,
    ):
        if isinstance(batch, PreparedBatch):
            prepared = batch
            if training is None:
                training = prepare_training_inputs(
                    prepared,
                    generator=generator,
                    recycle_count=0 if recycle_count is None else int(recycle_count),
                )
            if neighbours is None:
                neighbours = prepare_neighbours(
                    prepared,
                    training.dmap,
                    fape_count=int(self.config.fape_neighbours),
                    local_count=int(self.config.local_neighbours),
                )
        else:
            prepared, training, neighbours = self.prepare(
                batch,
                generator=generator,
                recycle_count=0 if recycle_count is None else int(recycle_count),
            )

        return self.forward_prepared(prepared, training, neighbours, return_trace=return_trace)

    def forward_prepared(
        self,
        prepared: PreparedBatch,
        training: TrainingInputs,
        neighbours: NeighbourSet,
        *,
        return_trace: bool = False,
    ):
        if int(training.recycle_count) != 0:
            raise ValueError("BackboneAutoencoderNative hot path supports recycle_count=0")

        data = _legacy_data(prepared, training)
        latent = self.encoder(data, generator=None)
        data["latent"] = latent

        prev = self.decoder.init_prev(data)
        output = self._decode_once(data, prev, neighbours)
        total, losses = self.loss_fn(prepared, training, neighbours, output)
        out = {"results": output.to_legacy_result_dict(), "losses": losses}
        if return_trace:
            trace = {
                "prep/pos": training.pos_init.detach(),
                "prep/pos_gt": prepared.pos_gt.detach(),
                "prep/mask": prepared.mask.detach(),
                "prep/dmap": training.dmap.detach(),
                "enc/latent": latent.detach(),
                "dec/pos": output.pos.detach(),
                "dec/local": output.local.detach(),
                "dec/atom_pos": output.atom_pos.detach(),
                "dec/aa": output.aa.detach(),
            }
            return total, out, trace
        return total, out

    def _decode_once(self, data: dict[str, torch.Tensor], prev: dict[str, torch.Tensor], neighbours: NeighbourSet) -> ModelOutput:
        local, pos, resi, chain, batch, mask = prepare_decoder_features_native(self.decoder, data, prev)
        local, pos, trajectory, sup_distogram = self.decoder.decoder_stack(
            local,
            pos,
            resi,
            chain,
            batch,
            mask,
            neighbours.fape,
            generator=None,
        )
        aa_logits, decoder_features, corrupt_aa = self.decoder.aa_decoder.decode_train(
            data["aa_gt"], local, pos, resi, chain, batch, mask
        )
        aatype = self.select_angle_aatype(data["aa_gt"], aa_logits)
        raw_angles, angles, atom_pos = self.decoder.angle_pos(aatype, local, pos)
        return ModelOutput(
            latent=data["latent"],
            trajectory=trajectory,
            sup_neighbours=neighbours.fape,
            sup_distogram=sup_distogram,
            local=local,
            pos=pos,
            aa=aa_logits,
            aa_features=decoder_features,
            corrupt_aa=corrupt_aa * data["mask"] * (data["aa_gt"] != 20),
            aa_gt=data["aa_gt"],
            raw_angles=raw_angles,
            angles=angles,
            atom_pos=atom_pos,
        )

    @staticmethod
    def _select_ground_truth_aatype(aa_gt: torch.Tensor, aa_logits: torch.Tensor) -> torch.Tensor:
        del aa_logits
        return aa_gt

    @staticmethod
    def _select_predicted_aatype(aa_gt: torch.Tensor, aa_logits: torch.Tensor) -> torch.Tensor:
        del aa_gt
        return aa_logits.argmax(dim=-1)


def _legacy_data(prepared: PreparedBatch, training: TrainingInputs) -> dict[str, torch.Tensor]:
    data = prepared.to_legacy_base_dict()
    data.update({"pos": training.pos_init, "dmap": training.dmap})
    return data


def _native_backbone_config(config: Any) -> Any:
    c = deepcopy(config)
    unsupported = {
        "atom37_parallel_mode": getattr(c, "atom37_parallel_mode", "none") != "none",
        "atom37_main_branch": bool(getattr(c, "atom37_main_branch", False)),
        "codebook_size": bool(getattr(c, "codebook_size", 0)),
        "fsq": bool(getattr(c, "fsq", False)),
        "input_diffusion": bool(getattr(c, "input_diffusion", False)),
        "latent_diffusion": bool(getattr(c, "latent_diffusion", False)),
        "noembed": bool(getattr(c, "noembed", False)),
        "time_embedding": bool(getattr(c, "time_embedding", False)),
        "noise_encoder": bool(getattr(c, "noise_encoder", 0)),
        "symm": getattr(c, "symm", None) is not None,
        "num_recycle": int(getattr(c, "num_recycle", 0)) != 0,
        "multi_query": bool(getattr(c, "multi_query", False)),
        "teacher_forcing_style": bool(getattr(c, "teacher_forcing_style", False)),
        "distogram_block": getattr(c, "distogram_block", None) != "inner",
        "no_fape2": bool(getattr(c, "no_fape2", False)),
    }
    enabled = [name for name, is_enabled in unsupported.items() if is_enabled]
    if enabled:
        raise ValueError(f"BackboneAutoencoderNative v1 does not support: {', '.join(enabled)}")
    c.atom37_parallel_mode = "none"
    c.atom37_main_branch = False
    c.num_random_neighbours = 0
    c.num_recycle = 0
    c.input_diffusion = False
    c.latent_diffusion = False
    c.time_embedding = False
    c.noembed = False
    c.noise_encoder = 0
    c.multi_query = False
    c.distogram_block = "inner"
    c.teacher_forcing_style = False
    c.symm = None
    c.fsq = False
    c.codebook_size = 0
    return c
