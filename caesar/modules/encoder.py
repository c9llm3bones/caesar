"""Encoder for protein structures in PyTorch.

Adapted from SALAD structure_autoencoder.
"""
# IN PROGRESS

import torch
import torch.nn as nn
import torch.nn as nn
from typing import Optional, Dict, Any

from caesar.utils.geometry import Vec3Array, Rot3Array

from caesar.modules.basic import Linear, MLP, block_stack, init_linear
from caesar.modules.utils import (
    distance_rbf, 
    index_mean,
    get_neighbours,
    extract_neighbours,
)
from caesar.modules.geometric import (
    extract_aa_frames, 
    rotation_features,
    distance_features,
    direction_features,
    sequence_relative_position,
    position_rotation_features,
    direction_features,
    pair_vector_features
)
from caesar.config import EncoderConfig

class EncoderBlock(nn.Module):
    """Encoder block."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.pair_mlp = MLP(
            2 * c.pair_size,
            c.pair_size,
            activation=torch.nn.gelu,
            final_init=init_linear()
        )
        # no attn for now. will be added later
        self.attn = SparseStructureAttention(c)
        self.update = EncoderUpdate(c)

        self.ln_attn = nn.LayerNorm(c.local_size)
        self.ln_update = nn.LayerNorm(c.local_size)

    def forward(self, features, pos, resi, chain, batch, mask):
        c = self.config

        neighbours = extract_neighbours(16, 16, 32)(
            Vec3Array.from_array(pos),
            resi, chain, batch, mask
        )

        pair, pair_mask = aa_decoder_pair_features(c)(
            Vec3Array.from_array(pos),
            neighbours, resi, chain, batch, mask
        )

        pair = self.pair_mlp(pair)

        features = features + self.attn(
            self.ln_attn(features),
            pos, pair, pair_mask,
            neighbours, resi, chain, batch, mask
        )

        features = features + self.update(
            self.ln_update(features),
            pos, chain, batch, mask
        )

        return features

class EncoderStack(nn.Module):
    """Encoder stack."""

    def __init__(self, config, depth=None):
        super().__init__()
        self.config = config
        self.depth = depth or 3

        self.blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(self.depth)]
        )

        self.ln_final = nn.LayerNorm(config.local_size)

    def forward(self, local, pos, resi, chain, batch, mask):
        for block in self.blocks:
            local = block(
                local, pos,
                resi, chain, batch, mask
            )

        local = self.ln_final(local)
        return local

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.config = c

        self.init_local = InitLocalFeatures(c)

        self.local_mlp = MLP(
            size=4 * c.local_size,
            out_size=c.local_size,
            activation=F.gelu,
            bias=False,
            final_init=init_linear(),   
        )

        self.local_ln = nn.LayerNorm(c.local_size)

        if not c.noembed:
            self.ln_final = nn.LayerNorm(c.local_size)
            self.to_latent = Linear(c.latent_size, bias=False, initializer=init_linear()) 


    def prepare_features(self, data):
        """Construct encoder input features from a batch of data."""
        c = self.config

        pos = data["pos_input"]

        if c.noise_encoder and self.training:
            pos = pos + c.noise_encoder * torch.randn_like(pos)

        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data["mask"]

        pos = Vec3Array.from_array(pos)

        neighbours = extract_neighbours(5, 5, 0)(
            pos, resi, chain, batch, mask
        )

        local_features = self.init_local(pos, neighbours, resi, chain, batch, mask)
        local = self.local_mlp(local_features)

        if c.time_embedding and c.input_diffusion and "time" in data:
            time = data["time"]
            time = distance_rbf(time, 0, 1, bins=100)
            local = local + Linear(
                local.shape[-1],
                bias=False,
                initializer="linear"
            )(time)

        local = self.local_ln(local)

        return local, pos.to_array(), resi, chain, batch, mask

    def forward(self, data):
        c = self.config

        local, pos, resi, chain, batch, mask = self.prepare_features(data)

        local = EncoderStack(c, depth=c.encoder_depth)(
            local, pos, resi, chain, batch, mask
        )

        if c.noembed:
            return local

        local = self.ln_final(local)
        return self.to_latent(local)

def aa_decoder_pair_features(c):
    """Pair features for the AADecoder module."""
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(32, one_hot=True, pseudo_chains=True)(
                resi, chain, batch, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            direction_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            position_rotation_features(pos, neighbours))
        pair += Linear(c.pair_size, bias=False, initializer="linear")(
            pair_vector_features(pos, neighbours))
        pair = nn.LayerNorm([-1], True, True)(pair)
        return pair, pair_mask
    return inner

class InitLocalFeatures(nn.Module):
    """Torch port of init_local_features(c), with parameters registered."""
    def __init__(self, c):
        super().__init__()
        self.c = c

        self.relpos = sequence_relative_position(8, one_hot=True, pseudo_chains=True)

        self.p_relpos = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dist   = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dir    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_rot    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_vec    = Linear(c.pair_size, bias=False, initializer="linear")

        # hk.LayerNorm([-1], True, True) equivalent
        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)

        self.mlp = MLP(size=c.pair_size * 2, out_size=c.pair_size, activation=F.gelu)

        self.to_local = Linear(c.local_size, bias=False)

    def forward(self, pos, neighbours, resi, chain, batch, mask):
        neighbours = neighbours.to(torch.long)

        valid = (neighbours != -1)
        neigh = neighbours.clamp_min(0)

        pair_mask = (mask[:, None] * mask[neigh]).to(torch.bool) & valid  # bool mask

        pair  = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        pair += self.p_dist(distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += self.p_dir(direction_features(pos, neighbours))
        pair += self.p_rot(position_rotation_features(pos, neighbours))
        pair += self.p_vec(pair_vector_features(pos, neighbours))

        pair = self.ln(pair)
        pair = self.mlp(pair)

        # jnp.where(pair_mask[...,None], pair, 0)
        pair = pair * pair_mask.unsqueeze(-1).to(pair.dtype)

        denom = pair_mask.sum(dim=-1, keepdim=True).to(pair.dtype).clamp_min(1.0)
        local = pair.sum(dim=1) / denom

        local = self.to_local(local)
        return local
