"""Encoder for protein structures in PyTorch.

Adapted from SALAD structure_autoencoder.
"""
# IN PROGRESS

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from caesar.modules.basic import Linear, MLP, block_stack, init_linear
from caesar.geometry import (
    Vec3Array, 
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
    """Equivariant protein structure encoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        
        if not c.noembed:
            self.ln_final = nn.LayerNorm(c.local_size)
            self.to_latent = Linear(
                in_features=c.local_size,
                out_features=c.latent_size,
                bias=False,
                initializer=init_linear()
            )

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

        local_features = init_local_features(c)(
            pos, neighbours, resi, chain, batch, mask
        )

        local = MLP(
            4 * c.local_size,
            c.local_size,
            activation=torch.nn.gelu,
            bias=False,
            final_init=init_linear()
        )(local_features)

        if c.time_embedding and c.input_diffusion and "time" in data:
            time = data["time"]
            time = distance_rbf(time, 0, 1, bins=100)
            local = local + Linear(
                local.shape[-1],
                bias=False,
                initializer="linear"
            )(time)

        local = nn.LayerNorm(local.shape[-1])(local)

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

def init_local_features(c):
    """Initial structure embedding for the Encoder."""
    def inner(pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask *= neighbours != -1
        pair = Linear(c.pair_size, bias=False, initializer="linear")(
            sequence_relative_position(8, one_hot=True, pseudo_chains=True)(
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
        pair = MLP(c.pair_size * 2, c.pair_size, activation=torch.nn.gelu)(pair)
        pair = torch.where(pair_mask[..., None], pair, 0)
        local = pair.sum(axis=1) / torch.maximum(pair_mask.sum(axis=-1)[..., None], 1)
        local = Linear(c.local_size, bias=False)(local)
        return local
    return inner