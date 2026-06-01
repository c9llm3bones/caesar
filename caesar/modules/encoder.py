"""Encoder for protein structures in PyTorch.

Adapted from SALAD structure_autoencoder.
"""
# IN PROGRESS

import torch
import torch.nn as nn
import torch.nn.functional as F

from caesar.utils.geometry import Vec3Array

from caesar.modules.basic import Linear, MLP, gelu_salad, init_linear, init_relu, init_zeros
from caesar.modules.utils import (
    distance_rbf, 
    index_mean,
    get_neighbours,
)
from caesar.modules.utils.geometry import (
    extract_neighbours_salad_compatible as extract_neighbours,
    atom37_to_ncacocb,
    atom37_local_feature_channels,
    mask_atom37_local_positions,
)
from caesar.modules.geometric import (
    SparseStructureAttention,
    extract_aa_frames, 
    rotation_features,
    distance_features,
    direction_features,
    sequence_relative_position,
    position_rotation_features,
    pair_vector_features
)
from caesar.config import EncoderConfig

def _gather_rows(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Gather along dim 0 for arbitrary index shape with legacy -1 semantics."""
    if index.dtype != torch.long:
        index = index.long()
    flat = index.reshape(-1).remainder(source.shape[0])
    gathered = torch.index_select(source, dim=0, index=flat)
    return gathered.reshape(*index.shape, *source.shape[1:])


def _gather_vec3(source: Vec3Array, index: torch.Tensor) -> Vec3Array:
    """Gather Vec3Array along dim 0 for arbitrary index shape."""
    return Vec3Array(
        _gather_rows(source.x, index),
        _gather_rows(source.y, index),
        _gather_rows(source.z, index),
    )

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
            activation=gelu_salad,
            final_init=init_linear()
        )
        self.attn = SparseStructureAttention(c)
        self.update = EncoderUpdate(c)

        self.ln_attn = nn.LayerNorm(c.local_size)
        self.ln_update = nn.LayerNorm(c.local_size)

    def forward(self, features, pos, resi, chain, batch, mask, *, generator: torch.Generator | None = None):
        c = self.config
        nr = int(getattr(c, "num_random_neighbours", 32))
        pos_v = Vec3Array.from_array(pos)
        neighbours = extract_neighbours(16, 16, nr)(
            pos_v,
            resi, chain, batch, mask,
            generator=generator
        )

        pair, pair_mask = self.pair_features(
            pos_v,
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

class EncoderUpdate(nn.Module):
    """GeGLU update for the encoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.localpos_mlp = MLP(
            size=c.local_size * 2,
            out_size=c.local_size,
            activation=gelu_salad,    
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
        gate = gelu_salad(self.gate_proj(local))
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

    def forward(self, local, pos, resi, chain, batch, mask, *, generator: torch.Generator | None = None):
        for block in self.blocks:
            local = block(
                local, pos,
                resi, chain, batch, mask,
                generator=generator
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
            if (getattr(c, 'time_embedding', False) and getattr(c, 'input_diffusion', False)) else None

        self.stack = EncoderStack(c, depth=c.encoder_depth)
        self.local_mlp = MLP(
            size=4 * c.local_size,
            out_size=c.local_size,
            activation=gelu_salad,
            bias=False,
            final_init=init_linear(),
        )

        self.local_ln = nn.LayerNorm(c.local_size)

        if not getattr(c, 'noembed', False):
            self.ln_final = nn.LayerNorm(c.local_size)
            self.to_latent = Linear(c.latent_size, bias=False, initializer=init_linear())

        # atom37 main branch encoder
        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )
        if atom37_main:
            self.init_local_atom37 = InitLocalFeaturesAtom37(c)
            self.local_mlp_atom37 = MLP(
                size=4 * c.local_size, out_size=c.local_size,
                activation=gelu_salad, bias=False, final_init=init_linear(),
            )
            self.local_ln_atom37 = nn.LayerNorm(c.local_size)
            self.stack_atom37 = EncoderStackAtom37(c, depth=c.encoder_depth)
            if not getattr(c, 'noembed', False):
                self.ln_final_atom37 = nn.LayerNorm(c.local_size)
                self.to_latent_atom37 = Linear(c.latent_size, bias=False, initializer=init_linear())


    def prepare_features(
        self,
        data,
        *,
        generator: torch.Generator | None = None,
        trace: dict | None = None,
    ):
        """Construct encoder input features from a batch of data."""
        c = self.config

        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )

        if atom37_main:
            pos37 = data['pos37_input']
            if getattr(c, 'noise_encoder', 0) and self.training:
                pos37 = pos37 + c.noise_encoder * torch.randn_like(pos37)
            resi = data['residue_index']
            chain = data['chain_index']
            batch = data['batch_index']
            mask = data.get('mask_bool', data['mask'])
            if mask.dtype != torch.bool:
                mask = mask != 0
            atom_mask37 = data['all_atom_mask_37'].to(dtype=pos37.dtype)
            pos37_v = Vec3Array.from_array(pos37)
            neighbours37 = extract_neighbours(5, 5, 0)(
                pos37_v, resi, chain, batch, mask, generator=generator)
            local_features37 = self.init_local_atom37(
                pos37_v, atom_mask37, neighbours37, resi, chain, batch, mask)
            local37 = self.local_mlp_atom37(local_features37)
            local37 = self.local_ln_atom37(local37)
            return local37, pos37, atom_mask37, resi, chain, batch, mask

        pos = data["pos_input"]

        if getattr(c, 'noise_encoder', 0) and self.training:
            pos = pos + c.noise_encoder * torch.randn_like(pos)

        resi = data["residue_index"]
        chain = data["chain_index"]
        batch = data["batch_index"]
        mask = data.get("mask_bool", data["mask"])
        if mask.dtype != torch.bool:
            mask = mask != 0

        pos = Vec3Array.from_array(pos)

        neighbours = extract_neighbours(5, 5, 0)(
            pos, resi, chain, batch, mask,
            generator=generator
        )

        local_features = self.init_local(pos, neighbours, resi, chain, batch, mask)
        local = self.local_mlp(local_features)

        if getattr(c, 'time_embedding', False) and getattr(c, 'input_diffusion', False) and "time" in data:
            time = data["time"]
            time = distance_rbf(time, 0, 1, bins=100)
            local = local + self.time_proj(time)

        local = self.local_ln(local)

        if trace is not None:
            trace["enc/neighbours0"] = neighbours.detach()
            trace["enc/local_features0"] = local_features.detach()
            trace["enc/local0"] = local.detach()

        return local, pos.to_tensor(), resi, chain, batch, mask

    def forward(
        self,
        data,
        *,
        generator: torch.Generator | None = None,
        trace: dict | None = None,
    ):
        c = self.config

        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )

        if atom37_main:
            local37, pos37, atom_mask37, resi, chain, batch, mask = self.prepare_features(
                data, generator=generator, trace=trace)
            local37 = self.stack_atom37(local37, pos37, atom_mask37,
                                        resi, chain, batch, mask, generator=generator)
            if getattr(c, 'noembed', False):
                return local37
            local37 = self.ln_final_atom37(local37)
            return self.to_latent_atom37(local37)

        local, pos, resi, chain, batch, mask = self.prepare_features(
            data, generator=generator, trace=trace
        )
        local = self.stack(local, pos, resi, chain, batch, mask, generator=generator)

        if getattr(c, 'noembed', False):
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
        if neighbours.dtype != torch.long:
            neighbours = neighbours.long()

        mask_bool = mask
        if mask_bool.dtype != torch.bool:
            mask_bool = (mask_bool != 0)
        neigh_mask = _gather_rows(mask_bool, neighbours)
        pair_mask = (mask_bool[:, None] & neigh_mask & (neighbours != -1))
        
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

        self.mlp = MLP(size=c.pair_size * 2, out_size=c.pair_size, activation=gelu_salad)

        self.to_local = Linear(c.local_size, bias=False)

    def forward(self, pos, neighbours, resi, chain, batch, mask):
        if neighbours.dtype != torch.long:
            neighbours = neighbours.long()

        mask_bool = mask
        if mask_bool.dtype != torch.bool:
            mask_bool = (mask_bool != 0)

        neigh_mask = _gather_rows(mask_bool, neighbours)
        pair_mask_bool = (mask_bool[:, None] & neigh_mask & (neighbours != -1))
        pair  = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        pair += self.p_dist(distance_features(pos, neighbours, d_min=0.0, d_max=22.0))
        pair += self.p_dir(direction_features(pos, neighbours))
        pair += self.p_rot(position_rotation_features(pos, neighbours))
        pair += self.p_vec(pair_vector_features(pos, neighbours))

        pair = self.ln(pair)
        pair = self.mlp(pair)

        pair = torch.where(
            pair_mask_bool.unsqueeze(-1),
            pair,
            torch.zeros_like(pair),
        )

        denom = pair_mask_bool.sum(dim=-1, keepdim=True).to(dtype=pair.dtype).clamp_min(1.0)
        local = pair.sum(dim=1) / denom
        local = self.to_local(local)
        return local


class InitLocalFeaturesAtom37(nn.Module):
    """Initial structure embedding for the atom37 encoder branch."""
    def __init__(self, c):
        super().__init__()
        self.c = c

        self.relpos = sequence_relative_position(8, one_hot=True, pseudo_chains=True)

        self.p_relpos = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dist   = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dir    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_rot    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_vec    = Linear(c.pair_size, bias=False, initializer="linear")
        # atom37 local features
        # channels per residue: 37*3 + 37 + 37*8 + 37 = 481
        self.p_local_self = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_local_self_j = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_local_neigh = Linear(c.pair_size, bias=False, initializer="linear")

        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)
        self.mlp = MLP(size=c.pair_size * 2, out_size=c.pair_size, activation=gelu_salad)
        self.to_local = Linear(c.local_size, bias=False)

    def forward(self, pos37_v: Vec3Array, atom_mask37: torch.Tensor,
                neighbours: torch.Tensor, resi, chain, batch, mask):
        if neighbours.dtype != torch.long:
            neighbours = neighbours.long()

        mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
        neigh_mask = _gather_rows(mask_bool, neighbours)
        if neigh_mask.dtype != torch.bool:
            neigh_mask = neigh_mask.bool()
        pair_mask_bool = mask_bool[:, None] & neigh_mask & (neighbours != -1)

        # use NCACOCB view for the standard geometric features
        pos_view = Vec3Array.from_array(atom37_to_ncacocb(pos37_v.to_tensor()))

        pair  = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        pair += self.p_dist(distance_features(pos_view, neighbours, d_min=0.0, d_max=22.0))
        pair += self.p_dir(direction_features(pos_view, neighbours))
        pair += self.p_rot(position_rotation_features(pos_view, neighbours))
        pair += self.p_vec(pair_vector_features(pos_view, neighbours))

        # atom37 local geometry features
        frames37, local_pos37 = extract_aa_frames(pos37_v)
        local_pos37_arr = local_pos37.to_tensor()  # (N, 37, 3)
        local_self_feat = atom37_local_feature_channels(local_pos37_arr, atom_mask37)

        local_neigh_pos = frames37[:, None, None].apply_inverse_to_point(
            _gather_vec3(pos37_v, neighbours)
        ).to_tensor()  # (N, K, 37, 3)
        neigh_mask37 = _gather_rows(atom_mask37, neighbours)  # (N, K, 37)
        local_neigh_feat = atom37_local_feature_channels(local_neigh_pos, neigh_mask37)

        pair += self.p_local_self(local_self_feat)[:, None]
        pair += self.p_local_self_j(_gather_rows(local_self_feat, neighbours))
        pair += self.p_local_neigh(local_neigh_feat)

        pair = self.ln(pair)
        pair = self.mlp(pair)

        pair = torch.where(pair_mask_bool.unsqueeze(-1), pair, torch.zeros_like(pair))

        denom = pair_mask_bool.sum(dim=-1, keepdim=True).to(dtype=pair.dtype).clamp_min(1.0)
        local = pair.sum(dim=1) / denom
        local = self.to_local(local)
        return local


class EncoderUpdateAtom37(nn.Module):
    """GeGLU update for the atom37 encoder branch."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.localpos_mlp = MLP(
            size=c.local_size * 2,
            out_size=c.local_size,
            activation=gelu_salad,
            final_init=init_zeros()
        )
        hidden = c.local_size * c.factor
        self.update_proj = Linear(hidden, initializer=init_linear(), bias=False)
        self.gate_proj = Linear(hidden, initializer=init_relu(), bias=False)
        self.out_proj = Linear(c.local_size, initializer=init_zeros(), bias=True)

    def forward(self, local, pos37, chain=None, batch=None, mask=None):
        pos_v = pos37 if isinstance(pos37, Vec3Array) else Vec3Array.from_array(pos37)
        _, local_pos37 = extract_aa_frames(pos_v)
        local_pos37_flat = local_pos37.to_tensor().reshape(local_pos37.shape[0], -1)
        local = local + self.localpos_mlp(local_pos37_flat)
        upd = self.update_proj(local)
        gate = gelu_salad(self.gate_proj(local))
        return self.out_proj(gate * upd)


class AADecoderPairFeaturesAtom37(nn.Module):
    """Pair features for the atom37 encoder/decoder branch."""

    def __init__(self, c):
        super().__init__()
        self.c = c

        self.relpos = sequence_relative_position(32, one_hot=True, pseudo_chains=True)
        self.p_relpos = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dist   = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_dir    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_rot    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_vec    = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_local_self = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_local_self_j = Linear(c.pair_size, bias=False, initializer="linear")
        self.p_local_neigh = Linear(c.pair_size, bias=False, initializer="linear")

        self.ln = nn.LayerNorm(c.pair_size, elementwise_affine=True)

    def forward(self, pos37_v: Vec3Array, atom_mask37: torch.Tensor,
                neighbours: torch.Tensor, resi, chain, batch, mask):
        if neighbours.dtype != torch.long:
            neighbours = neighbours.long()

        mask_bool = mask if mask.dtype == torch.bool else (mask != 0)
        neigh_mask_b = _gather_rows(mask_bool, neighbours)
        if neigh_mask_b.dtype != torch.bool:
            neigh_mask_b = neigh_mask_b.bool()
        pair_mask = (mask_bool[:, None] & neigh_mask_b & (neighbours != -1)).to(dtype=pos37_v.x.dtype)

        pos_view = Vec3Array.from_array(atom37_to_ncacocb(pos37_v.to_tensor()))
        pair  = self.p_relpos(self.relpos(resi, chain, batch, neighbours))
        pair += self.p_dist(distance_features(pos_view, neighbours, d_min=0.0, d_max=22.0))
        pair += self.p_dir(direction_features(pos_view, neighbours))
        pair += self.p_rot(position_rotation_features(pos_view, neighbours))
        pair += self.p_vec(pair_vector_features(pos_view, neighbours))

        frames37, local_pos37 = extract_aa_frames(pos37_v)
        local_pos37_arr = local_pos37.to_tensor()
        local_self_feat = atom37_local_feature_channels(local_pos37_arr, atom_mask37)

        local_neigh_pos = frames37[:, None, None].apply_inverse_to_point(
            _gather_vec3(pos37_v, neighbours)
        ).to_tensor()
        neigh_mask37 = _gather_rows(atom_mask37, neighbours)
        local_neigh_feat = atom37_local_feature_channels(local_neigh_pos, neigh_mask37)

        pair += self.p_local_self(local_self_feat)[:, None]
        pair += self.p_local_self_j(_gather_rows(local_self_feat, neighbours))
        pair += self.p_local_neigh(local_neigh_feat)

        pair = self.ln(pair)
        return pair, pair_mask


class EncoderBlockAtom37(nn.Module):
    """Encoder block for the atom37 branch."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config

        self.pair_features = AADecoderPairFeaturesAtom37(c)
        self.pair_mlp = MLP(2 * c.pair_size, c.pair_size, activation=gelu_salad, final_init=init_linear())
        self.attn = SparseStructureAttention(c)
        self.update = EncoderUpdateAtom37(c)

        self.ln_attn = nn.LayerNorm(c.local_size)
        self.ln_update = nn.LayerNorm(c.local_size)

    def forward(self, features, pos37, atom_mask37,
                resi, chain, batch, mask, *, generator=None):
        c = self.config
        nr = int(getattr(c, "num_random_neighbours", 32))
        pos37_v = Vec3Array.from_array(pos37) if isinstance(pos37, torch.Tensor) else pos37
        neighbours = extract_neighbours(16, 16, nr)(
            pos37_v, resi, chain, batch, mask, generator=generator)

        pair, pair_mask = self.pair_features(pos37_v, atom_mask37, neighbours, resi, chain, batch, mask)
        pair = self.pair_mlp(pair)

        pos37_t = pos37 if isinstance(pos37, torch.Tensor) else pos37.to_tensor()
        features = features + self.attn(
            self.ln_attn(features),
            pos37_t, pair, pair_mask,
            neighbours, resi, chain, batch, mask)
        features = features + self.update(
            self.ln_update(features), pos37_t, chain, batch, mask)
        return features


class EncoderStackAtom37(nn.Module):
    """Encoder stack for the atom37 branch."""

    def __init__(self, config, depth=None):
        super().__init__()
        self.config = config
        self.depth = depth or 3
        self.blocks = nn.ModuleList([EncoderBlockAtom37(config) for _ in range(self.depth)])
        self.ln_final = nn.LayerNorm(config.local_size)

    def forward(self, local, pos37, atom_mask37,
                resi, chain, batch, mask, *, generator=None):
        pos37_t = pos37 if isinstance(pos37, torch.Tensor) else pos37.to_tensor()
        for block in self.blocks:
            local = block(local, pos37_t, atom_mask37,
                          resi, chain, batch, mask, generator=generator)
        local = self.ln_final(local)
        return local
