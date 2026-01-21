"""Encoder for protein structures in PyTorch.

Adapted from SALAD structure_autoencoder.
"""
# IN PROGRESS

import torch
import torch.nn as nn
import torch.nn.functional as F

from caesar.utils.geometry import Vec3Array

from caesar.modules.basic import Linear, MLP, block_stack, init_linear, init_relu, init_zeros
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
        
        self.pair_features = AADecoderPairFeatures(c)
        
        self.pair_mlp = MLP(
            2 * c.pair_size,
            c.pair_size,
            activation=F.gelu,
            final_init=init_linear()
        )
        # no attn for now. will be added later
        # self.attn = SparseStructureAttention(c)
        self.update = EncoderUpdate(c)

        self.ln_attn = nn.LayerNorm(c.local_size)
        self.ln_update = nn.LayerNorm(c.local_size)

    def forward(self, features, pos, resi, chain, batch, mask):
        c = self.config

        neighbours = extract_neighbours(16, 16, 32)(
            Vec3Array.from_array(pos),
            resi, chain, batch, mask
        )

        pair, pair_mask = self.pair_features(
            Vec3Array.from_array(pos),
            neighbours, resi, chain, batch, mask
        )

        pair = self.pair_mlp(pair)

        # features = features + self.attn(
        #     self.ln_attn(features),
        #     pos, pair, pair_mask,
        #     neighbours, resi, chain, batch, mask
        # )

        features = features + self.update(
            self.ln_update(features),
            pos, chain, batch, mask
        )

        return features

class EncoderUpdate(nn.Module):
    """GeGLU update for the encoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        # salad: MLP(local_dim*2 -> local_dim, gelu, final_init=zeros)(local_pos_flat)
        self.localpos_mlp = MLP(
            size=c.local_size * 2,
            out_size=c.local_size,
            activation=F.gelu,    
            final_init=init_zeros()
        )

        hidden = c.local_size * c.factor
        self.update_proj = Linear(hidden, initializer=init_linear(), bias=False)
        self.gate_proj = Linear(hidden, initializer=init_relu(), bias=False)

        self.out_proj = Linear(c.local_size, initializer=init_zeros(), bias=True)

    def forward(self, local, pos, chain=None, batch=None, mask=None):
        """
        Args:
            local: Tensor (N, local_size)
            pos: Tensor (N, A, 3) or Vec3Array (N, A)

        Returns:
            Tensor (N, local_size): update delta to be added to local/features.
        """
        pos_v = pos if isinstance(pos, Vec3Array) else Vec3Array.from_array(pos)

        _, local_pos = extract_aa_frames(pos_v) # local_pos: Vec3Array (N, A)
        local_pos_flat = local_pos.to_tensor().reshape(local_pos.shape[0], -1)  # (N, A*3)

        local = local + self.localpos_mlp(local_pos_flat)

        upd = self.update_proj(local)
        gate = F.gelu(self.gate_proj(local))
        local_local = gate * upd

        return self.out_proj(local_local)
    
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
        self.time_proj = Linear(c.local_size, bias=False, initializer="linear") \
            if (c.time_embedding and c.input_diffusion) else None

        self.stack = EncoderStack(c, depth=c.encoder_depth)
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
            local = local + self.time_proj(time)

        local = self.local_ln(local)

        return local, pos.to_tensor(), resi, chain, batch, mask

    def forward(self, data):
        c = self.config

        local, pos, resi, chain, batch, mask = self.prepare_features(data)
        local = self.stack(local, pos, resi, chain, batch, mask)
        
        if c.noembed:
            return local

        local = self.ln_final(local)
        return self.to_latent(local)

class AADecoderPairFeatures(nn.Module):
    """Pair features for the AADecoder module."""
    def __init__(self, c):
        super().__init__()
        self.c = c

        self.relpos = sequence_relative_position(32, one_hot=True, pseudo_chains=True)

        self.p_relpos = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dist   = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dir    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_rot    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_vec    = Linear(c.pair_size, bias=False, initializer="linear")

        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)

    def forward(self, pos, neighbours, resi, chain, batch, mask):
        neighbours = neighbours.to(torch.long)

        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)

        pair  = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        pair += self.p_dist(distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += self.p_dir(direction_features(pos, neighbours))
        pair += self.p_rot(position_rotation_features(pos, neighbours))
        pair += self.p_vec(pair_vector_features(pos, neighbours))

        pair = self.ln(pair)
        return pair, pair_mask

class InitLocalFeatures(nn.Module):
    """port of init_local_features(c), with parameters registered."""
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

        ### safe gather for neighbours (handle -1)
        # valid = (neighbours != -1)
        # neigh = neighbours.clamp_min(0)
        # pair_mask = (mask[:, None] * mask[neigh]).to(torch.bool) & valid  # bool mask
  
        pair_mask = mask[:, None] * mask[neighbours]
        pair_mask = pair_mask * (neighbours != -1).to(pair_mask.dtype)
        pair_mask = pair_mask.to(torch.bool)

        pair  = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        pair += self.p_dist(distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += self.p_dir(direction_features(pos, neighbours))
        pair += self.p_rot(position_rotation_features(pos, neighbours))
        pair += self.p_vec(pair_vector_features(pos, neighbours))

        pair = self.ln(pair)
        pair = self.mlp(pair)

        pair = pair * pair_mask.unsqueeze(-1).to(pair.dtype)

        denom = pair_mask.sum(dim=-1, keepdim=True).to(pair.dtype).clamp_min(1.0)
        local = pair.sum(dim=1) / denom

        local = self.to_local(local)
        return local
