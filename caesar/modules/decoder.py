"""Decoder for protein structures in PyTorch.

Adapted from SALAD structure_autoencoder.
"""
# IN PROGRESS
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict

from caesar.modules.utils.geometry import distance_one_hot, get_spatial_neighbours, index_align
from caesar.modules.basic import Linear, MLP, init_zeros
from caesar.config import DecoderConfig

from caesar.modules.basic import Linear, MLP, block_stack, init_linear
from caesar.geometry import (
    Vec3Array, 
    distance_rbf, 
    index_mean,
    get_neighbours,
    get_random_neighbours,
    extract_aa_frames,    
    index_sum,
    single_protein_sidechains,
)
from caesar.config import EncoderConfig

class DecoderBlock(nn.Module):
    """Single decoder block."""
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        c = config
        
        # Feature update
        self.feature_update = MLP(
            hidden=c.local_size * 2,
            out_size=c.local_size,
            depth=2,
            init="gelu",
            final_init="zeros"
        )
        
        # Position update
        self.pos_update = Linear(
            in_features=c.local_size,
            out_features=12,  # 4 atoms * 3 coords
            bias=False,
            initializer="zeros"
        )
        
        self.ln_features = nn.LayerNorm(c.local_size)
    
    def forward(self,
                features: torch.Tensor,  # (batch, seq_len, local_size)
                pos: torch.Tensor,        # (batch, seq_len, 4, 3) positions
                ) -> tuple:
        """Forward pass of decoder block.
        
        Returns:
            features: updated features
            pos: updated positions
        """
        # Normalize and update features
        features_norm = self.ln_features(features)
        features = features + self.feature_update(features_norm)
        
        # Update positions (simplified)
        pos_delta = self.pos_update(features_norm)  # (batch, seq, 12)
        pos_delta = pos_delta.reshape(*pos_delta.shape[:-1], 4, 3)
        pos = pos + pos_delta
        
        return features, pos

class Decoder(nn.Module):
    """Protein structure decoder (Haiku → PyTorch)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder_block = DecoderBlock

    def forward(self, data, prev):
        c = self.config

        # select decoder type
        decoder_module = DecoderStack
        if c.equivariance == "nonequivariant":
            decoder_module = NonEquivariantDecoderStack
            self.decoder_block = NonEquivariantDecoderBlock
        elif c.equivariance == "semiequivariant":
            decoder_module = SemiEquivariantDecoderStack
            self.decoder_block = SemiEquivariantDecoderBlock

        decoder_stack = decoder_module(c, self.decoder_block)
        aa_decoder = AADecoder(c)

        # prepare features
        local, pos, resi, chain, batch, mask = self.prepare_features(data, prev)

        sup_neighbours = get_random_neighbours(c.fape_neighbours)(
            Vec3Array.from_array(data["pos_gt"][:, -1]), batch, mask
        )

        local, pos, trajectory, sup_distogram = decoder_stack(
            local, pos, resi, chain, batch, mask, sup_neighbours
        )

        result = {
            "latent": data["latent"],
            "trajectory": trajectory,
            "sup_neighbours": sup_neighbours,
            "sup_distogram": sup_distogram,
            "local": local,
            "pos": pos,
        }

        # latent diffusion
        if c.input_diffusion or c.latent_diffusion:
            latent_decoder = MLP(
                c.local_size * 2,
                c.local_size if c.noembed else c.latent_size,
                bias=False,
                activation=F.gelu,
                final_init=init_zeros()
            )

            latent_update = latent_decoder(torch.cat([
                nn.LayerNorm(local.shape[-1]).to(local.device)(local),
                data["latent"]
            ], dim=-1))

            time = data["time"]
            if c.vp_diffusion:
                predicted_latent = data["latent"] + latent_update
            else:
                predicted_latent = (
                    data["skip_latent"]
                    + latent_update * time[:, None] / torch.sqrt(1 + time[:, None] ** 2)
                )

            result["predicted_latent"] = predicted_latent

        # AA prediction
        aa_logits, decoder_features, corrupt_aa = aa_decoder.train(
            data["aa_gt"], local, pos, resi, chain, batch, mask
        )

        result.update({
            "aa": aa_logits,
            "aa_features": decoder_features,
            "corrupt_aa": corrupt_aa * data["mask"] * (data["aa_gt"] != 20),
            "aa_gt": data["aa_gt"],
        })

        # all-atom reconstruction
        aatype = data["aa_gt"]
        if c.eval:
            aatype = aa_logits.argmax(dim=-1)

        raw_angles, angles, atom_pos = get_angle_positions(
            aatype, local, pos
        )

        result.update({
            "raw_angles": raw_angles,
            "angles": angles,
            "atom_pos": atom_pos,
        })

        return result

    def init_prev(self, data):
        c = self.config
        return {
            "pos": data["pos"],
            "local": torch.zeros(
                (data["pos"].shape[0], c.local_size),
                device=data["pos"].device
            )
        }

    def prepare_features(self, data, prev):
        c = self.config

        pos = prev["pos"]
        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        latent = data["latent"]
        mask = data["mask"]

        _, local_pos = extract_aa_frames(Vec3Array.from_array(pos))

        local_features = [
            local_pos.to_array().reshape(local_pos.shape[0], -1),
            distance_rbf(local_pos.norm(), 0.0, 22.0).reshape(local_pos.shape[0], -1),
            latent
        ]

        if c.time_embedding and c.latent_diffusion and "time" in data:
            time = distance_rbf(data["time"], 0, 80.0, bins=200)
            local_features.append(time)

        local_features.append(
            nn.LayerNorm(prev["local"].shape[-1]).to(prev["local"].device)(prev["local"])
        )

        local_features = torch.cat(local_features, dim=-1)

        local = MLP(
            4 * c.local_size,
            c.local_size,
            activation=F.gelu,
            bias=False,
            final_init=init_linear()
        )(local_features)

        local = nn.LayerNorm(local.shape[-1]).to(local.device)(local)

        return local, pos, resi, chain, batch, mask

    def loss(self, data, result):
        c = self.config
        mask = data["mask"]
        batch = data["batch_index"]

        losses = {}
        total = 0.0

        # AA NLL
        aa_mask = mask * (data["aa_gt"] != 20)
        aa_nll = -(result["aa"] * F.one_hot(data["aa_gt"], 20)).sum(dim=-1)
        aa_nll = torch.where(aa_mask, aa_nll, torch.zeros_like(aa_nll))
        aa_nll = aa_nll.sum() / torch.clamp(aa_mask.sum(), min=1)

        losses["aa"] = aa_nll
        total += c.aa_weight * aa_nll

        # position losses
        base_weight = mask / torch.maximum(index_sum(mask.astype(torch.float32), batch, mask), 1) / (batch.max() + 1)

        # sparse neighbour FAPE ** 2
        pair_mask = batch[:, None] == batch[None, :]
        pair_mask *= mask[:, None] * mask[None, :]
        pos_gt = data["pos_gt"]
        pos_gt = torch.where(mask[:, None, None], pos_gt, 0)
        pos_gt = Vec3Array.from_array(pos_gt)
        frames_gt, _ = extract_aa_frames(torch.lax.stop_gradient(pos_gt))
        # CB distance
        distance = data["dmap"]
        distance = torch.where(pair_mask, distance, torch.inf)
        # get random neighbours to compute sparse FAPE on
        neighbours = get_random_neighbours(c.fape_neighbours)(distance, batch, mask)
        mask_neighbours = (neighbours != -1) * mask[:, None] * mask[neighbours]
        pos_gt_local = frames_gt[:, None, None].apply_inverse_to_point(pos_gt[neighbours])
        traj = Vec3Array.from_array(result["trajectory"])
        frames, _ = torch.vmap(extract_aa_frames)(traj)
        traj_local = frames[:, :, None, None].apply_inverse_to_point(traj[:, neighbours])
        fape_base = (traj_local - pos_gt_local).norm2()
        fape_clipped = torch.clip(fape_base, 0.0, c.clip_fape)
        if c.unclipped_weight:
            fape_clipped = fape_base * c.unclipped_weight + fape_clipped
        if c.no_fape2:
            # use FAPE instead of FAPE ** 2
            fape_clipped = torch.sqrt(torch.maximum(fape_clipped, 1e-6))
        fape_traj = fape_clipped.mean(axis=-1)
        fape_traj = torch.where(mask_neighbours[None], fape_traj, 0)
        fape_traj = fape_traj.sum(axis=-1) / torch.maximum(mask_neighbours.sum(axis=1)[None], 1)
        fape_traj = (fape_traj * base_weight).sum(axis=-1)
        losses["fape"] = fape_traj[-1] / 3
        losses["fape_trajectory"] = fape_traj.mean() / 3
        fape_loss = (c.fape_weight * fape_traj[-1] + c.fape_trajectory_weight * fape_traj.mean()) / 3
        total += fape_loss

        # sup distogram loss
        if c.distogram_block != "none":
            cb_gt = pos_gt[:, -1]
            sup_neighbours = result["sup_neighbours"]
            sup_mask = (sup_neighbours != -1) * mask[:, None] * mask[sup_neighbours]
            dist_gt = (cb_gt[:, None] - cb_gt[sup_neighbours]).norm()
            dist_one_hot = distance_one_hot(dist_gt, 0, 22.0, 16)
            distogram_nll = -(result["sup_distogram"] * dist_one_hot[None]).sum(axis=-1)
            distogram_nll = torch.where(sup_mask, distogram_nll, 0).sum(axis=-1)
            distogram_nll /= torch.maximum(sup_mask.sum(axis=1), 1)
            distogram_nll = (distogram_nll * base_weight).sum(axis=1)
            losses["distogram"] = distogram_nll[-1]
            losses["distogram_trajectory"] = distogram_nll.mean()
            total += 10.0 * distogram_nll[-1] + 5.0 * distogram_nll.mean()

        # Kabsch RMSD loss
        if c.kabsch_rmsd:
            last = traj[-1]
            pos_gt = torch.lax.stop_gradient(index_align(pos_gt, last, batch, mask))
            pos_loss_unclipped = (pos_gt[None] - traj).norm2()
            pos_loss_clipped = torch.clip(pos_loss_unclipped, 0, 100.0)
            pos_loss = pos_loss_clipped
            if c.unclipped_weight:
                pos_loss = pos_loss_unclipped * c.unclipped_weight + pos_loss
            pos_loss = pos_loss.mean(axis=-1)
            pos_loss *= base_weight[None]
            pos_loss = pos_loss.sum(axis=-1)
            losses["kabsch_rmsd"] = pos_loss[-1] / 3
            losses["kabsch_rmsd_trajectory"] = pos_loss.mean() / 3
            pos_loss = (c.fape_weight * pos_loss[-1] + c.fape_trajectory_weight * pos_loss.mean()) / 3
            total += pos_loss

        # local loss
        atom_pos = Vec3Array.from_array(result["atom_pos"])
        atom_pos_gt = Vec3Array.from_array(data["atom_pos"])
        local_neighbours = get_spatial_neighbours(
            count=c.local_neighbours)(
                pos_gt[:, -1], batch, mask)
        mask_gt = data["atom_mask"][local_neighbours]
        frames_gt, _ = extract_aa_frames(atom_pos_gt)
        atom_pos_gt = frames_gt[:, None, None].apply_inverse_to_point(atom_pos_gt[local_neighbours])
        frames, _ = extract_aa_frames(atom_pos)
        atom_pos = frames[:, None, None].apply_inverse_to_point(atom_pos[local_neighbours])
        local_loss = torch.where(mask_gt, (atom_pos - atom_pos_gt).norm2(), 0).sum(axis=(1, 2))
        local_loss /= torch.maximum(mask_gt.sum(axis=(1, 2)), 1)
        local_loss = (torch.where(mask, local_loss, 0) * base_weight).sum() / 3
        losses["local"] = local_loss
        total += c.local_weight * local_loss

        # VQ losses
        if c.codebook_size and not c.is_decoder:
            cl = result["codebook_losses"]
            losses.update(cl)
            if not c.state:
                total += c.codebook_loss_scale * (cl["codebook"] + cl["unassigned"])
            total += c.codebook_loss_scale * c.codebook_b * cl["commitment"]

        # additional denoising losses (not used in manuscript)
        if (c.input_diffusion or c.latent_diffusion) and c.latent_loss_scale:
            raw_loss = ((data["clean_latent"] - result["predicted_latent"]) ** 2).mean(axis=-1)
            if c.vp_diffusion:
                weighted_loss = (torch.where(mask, raw_loss, 0) * base_weight).sum()
            else:
                time = torch.maximum(data["time"], 1e-2)
                raw_loss = raw_loss * (1 + time ** 2) / torch.maximum(time ** 2, 1e-6)
                weighted_loss = (torch.where(mask, raw_loss, 0) * base_weight).sum()
            losses["latent"] = weighted_loss
            total += c.latent_loss_scale * weighted_loss
        # AlphaFold violation loss. (not used in manuscript)
        if c.violation_scale:
            res_mask = data["mask"]
            pred_mask = get_atom14_mask(data["aa_gt"]) * res_mask[:, None]
            violation, _ = violation_loss(data["aa_gt"],
                                          data["residue_index"],
                                          result["atom_pos"],
                                          pred_mask,
                                          res_mask,
                                          clash_overlap_tolerance=1.5,
                                          violation_tolerance_factor=2.0,
                                          chain_index=data["chain_index"],
                                          batch_index=data["batch_index"],
                                          per_residue=False)
            losses["violation"] = violation.mean()
            total += c.violation_scale * violation.mean()

        return total, losses

def get_angle_positions(aa_gt, local, pos):
    """Construct side chain atom positions from amino acid sequence and features."""
    frames, local_positions = extract_aa_frames(Vec3Array.from_array(pos))
    features = [
        local,
        local_positions.to_array().reshape(local_positions.shape[0], -1),
        distance_rbf(local_positions.norm(),
                     0.0, 10.0, 16).reshape(local_positions.shape[0], -1),
        torch.nn.one_hot(aa_gt, 21, axis=-1)
    ]
    raw_angles = MLP(
        local.shape[-1] * 2, 7 * 2, bias=False,
        activation=torch.nn.gelu, final_init="linear")(
            torch.cat(features, axis=-1))

    raw_angles = raw_angles.reshape(-1, 7, 2)
    angles = raw_angles / torch.sqrt(torch.maximum(
        (raw_angles ** 2).sum(axis=-1, keepdims=True), 1e-6))
    angle_pos, _ = single_protein_sidechains(
        aa_gt, frames, angles)
    angle_pos = angle_pos.to_array().reshape(-1, 14, 3)
    angle_pos = torch.cat((
        pos[..., :4, :],
        angle_pos[..., 4:, :]
    ), axis=-2)
    return raw_angles, angles, angle_pos

class AADecoder(nn.Module):
    """Amino acid sequence decoder module."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.local_size)
        self.proj = nn.Linear(config.local_size, 20, bias=False)
        nn.init.zeros_(self.proj.weight)

        self.stack = AADecoderStack(config, depth=config.aa_decoder_depth)

    def forward(self, aa, local, pos, resi, chain, batch, mask):
        neighbours = get_spatial_neighbours(32)(
            Vec3Array.from_array(pos)[:, -1], batch, mask
        )

        local = self.stack(
            aa, local, pos, neighbours,
            resi, chain, batch, mask
        )

        local = self.norm(local)
        logits = self.proj(local)
        return logits, local

    def train(self, aa, local, pos, resi, chain, batch, mask):
        aa = torch.full_like(aa, 20)
        logits, features = self.forward(
            aa, local, pos, resi, chain, batch, mask
        )
        logits = F.log_softmax(logits, dim=-1)
        return logits, features, torch.ones_like(mask)
    
class DecoderStack(nn.Module):
    def __init__(self, config, block_cls):
        super().__init__()
        self.blocks = nn.ModuleList([
            block_cls(config) for _ in range(config.depth)
        ])

    def forward(self, local, pos, resi, chain, batch, mask, sup_neighbours):
        trajectory = []
        sup_distograms = []

        for block in self.blocks:
            local, pos, sup_dist = block(
                local, pos, resi, chain, batch, mask, sup_neighbours
            )
            trajectory.append(pos)
            sup_distograms.append(sup_dist)

        trajectory = torch.stack(trajectory, dim=0)
        sup_distograms = torch.stack(sup_distograms, dim=0)

        return local, pos, trajectory, sup_distograms
    
class SemiEquivariantDecoderStack(nn.Module):
    """Stack of partly non-equivariant Decoder blocks using data augmentation."""

    def __init__(self, config, block_cls):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([
            block_cls(config) for _ in range(config.depth)
        ])

    def forward(self, local, pos,
                resi, chain, batch, mask,
                sup_neighbours):
        c = self.config

        # data augmentation before decoding
        pos = structure_augmentation(
            pos / c.sigma_data, batch, mask
        ) * c.sigma_data

        trajectory = []
        sup_distograms = []

        for block in self.blocks:
            local, pos, sup_dist = block(
                local, pos,
                resi, chain, batch, mask,
                sup_neighbours
            )
            trajectory.append(pos)
            sup_distograms.append(sup_dist)

        trajectory = torch.stack(trajectory, dim=0)
        sup_distograms = torch.stack(sup_distograms, dim=0)

        return local, pos, trajectory, sup_distograms

class NonEquivariantDecoderStack(nn.Module):
    """Stack of fully non-equivariant decoder blocks."""

    def __init__(self, config, block_cls):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList([
            block_cls(config) for _ in range(config.depth)
        ])

        # project (local + augmented pos) -> local_size
        self.input_proj = nn.Linear(
            config.local_size + 5 * 3,
            config.local_size,
            bias=False
        )

        # project back to positions
        self.pos_proj = nn.Linear(5 * 3, 5 * 3, bias=False)
        self.norm = nn.LayerNorm(5 * 3)

    def forward(self, local, pos,
                resi, chain, batch, mask,
                sup_neighbours):
        c = self.config

        # structure augmentation
        aug_pos = structure_augmentation(
            pos / c.sigma_data, batch, mask
        )

        # embed positions into local space
        localpos = torch.cat(
            [local, aug_pos.reshape(pos.shape[0], -1)],
            dim=-1
        )
        localpos = self.input_proj(localpos)

        trajectory = []
        sup_distograms = []

        for block in self.blocks:
            localpos, sup_dist = block(
                localpos,
                resi, chain, batch, mask,
                sup_neighbours
            )
            trajectory.append(localpos)
            sup_distograms.append(sup_dist)

        trajectory = torch.stack(trajectory, dim=0)
        sup_distograms = torch.stack(sup_distograms, dim=0)

        # project back to positions
        traj = self.norm(trajectory)
        traj = self.pos_proj(traj)
        traj = traj.view(
            traj.shape[0], traj.shape[1], 5, 3
        )
        traj = traj * c.sigma_data

        pos = traj[-1]
        return local, pos, traj, sup_distograms

def structure_augmentation_params(pos, batch, mask):
    """Sample random rotation and translation parameters for data augmentation."""
    # center positions
    center = index_mean(pos[:, 1], batch, mask[:, None])
    # centering + random translation
    translation = torch.rnd.normal(torch.next_rng_key(), pos[:, 1].shape)[batch]
    # random rotation
    rotation = random_rotation(batch)
    return center, translation, rotation

def apply_structure_augmentation(pos, center, translation, rotation):
    """Apply a rotation and translation to an array of atom positions."""
    # center positions
    pos -= center[:, None]
    # apply transformation
    pos = rotation[:, None].apply_to_point(Vec3Array.from_array(pos)).to_array()
    pos += translation[:, None]
    return pos

def apply_inverse_structure_augmentation(pos, center, translation, rotation):
    """Apply the inverse of a rotation and translation to an array of atom positions."""
    # invert translation
    pos -= translation[:, None]
    # invert rotation
    pos = rotation[:, None].apply_inverse_to_point(Vec3Array.from_array(pos)).to_array()
    # move to center
    pos += center[:, None]
    return pos

def structure_augmentation(pos, batch, mask):
    """Randomly rotate and translate an array of atom positions."""
    # get random augmentation parameters
    center, translation, rotation = structure_augmentation_params(pos, batch, mask)
    # apply random augmentation
    pos = apply_structure_augmentation(pos, center, translation, rotation)
    return pos

def random_rotation(batch):
    """Sample a random rotation."""
    x = torch.randn(batch.shape[0], 3)[batch]
    y = torch.randn(batch.shape[0], 3)[batch]
    result = Rot3Array.from_two_vectors(
        Vec3Array.from_array(x),
        Vec3Array.from_array(y))
    return result