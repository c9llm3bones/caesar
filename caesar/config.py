from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class NeighbourConfig:
    k_local: int
    k_nonlocal: int
    k_random: int = 0

    def __post_init__(self):
        if self.k_local < 0 or self.k_nonlocal < 0 or self.k_random < 0:
            raise ValueError("NeighbourConfig: k_local/k_nonlocal/k_random must be >= 0")

@dataclass
class AutoencoderConfig:
    local_size: int = 256
    pair_size: int = 64
    latent_size: int = 32

    encoder_depth: int = 3
    decoder_depth: int = 4

    noembed: bool = False
    noise_encoder: float = 0.0
    time_embedding: bool = False
    input_diffusion: bool = False
    eval: bool = False
    factor=4

    # ВАЖНО: default_factory, иначе ValueError про mutable default [[4]] [[7]]
    neighbours_init: NeighbourConfig = field(default_factory=lambda: NeighbourConfig(5, 5, 0))
    neighbours_block: NeighbourConfig = field(default_factory=lambda: NeighbourConfig(16, 16, 32))

    sigma_data: float = 10.0

    def __post_init__(self):
        if self.local_size <= 0 or self.pair_size <= 0 or self.latent_size <= 0:
            raise ValueError("local_size/pair_size/latent_size must be > 0")
        if self.encoder_depth <= 0:
            raise ValueError("encoder_depth must be > 0")
        if self.decoder_depth <= 0:
            raise ValueError("decoder_depth must be > 0")
        if self.noise_encoder < 0:
            raise ValueError("noise_encoder must be >= 0")
        if self.sigma_data <= 0:
            raise ValueError("sigma_data must be > 0")

@dataclass
class EncoderConfig(AutoencoderConfig):
    pass

@dataclass
class DecoderConfig(AutoencoderConfig):
    num_recycle: int = 1
    num_recycle_eval: int = 4
    clip_fape: float = 10.0
    mlp_factor: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.num_recycle < 0 or self.num_recycle_eval < 0:
            raise ValueError("num_recycle/num_recycle_eval must be >= 0")
        if self.clip_fape <= 0:
            raise ValueError("clip_fape must be > 0")
        if self.mlp_factor <= 0:
            raise ValueError("mlp_factor must be > 0")

def create_encoder_config(**kwargs) -> EncoderConfig:
    return EncoderConfig(**kwargs)


def create_decoder_config(**kwargs) -> DecoderConfig:
    return DecoderConfig(**kwargs)


def create_autoencoder_config(**kwargs) -> AutoencoderConfig:
    return AutoencoderConfig(**kwargs)
