# tests/configs.py
from caesar.modules.utils.collections import dotdict, deepcopy
import numpy as np

default = dotdict(
    local_size=128,
    pair_size=64,
    latent_size=20,

    relative_position_encoding_max=32,
    factor=4,
    heads=8,
    key_size=32,
    multi_query=False,
    sigma_data=10.0,

    depth=6,
    block_size=1,
    num_recycle=0,

    aa_decoder_depth=3,
    encoder_depth=3,

    p_clip=1.0,
    clip_fape=100,
    time_embedding=False,

    local_neighbours=16,
    fape_neighbours=64,

    local_weight=1.0,
    aa_weight=10.0,
    fape_weight=1,
    fape_trajectory_weight=0.5,

    min_size=50,
    max_size=None,
)

small = deepcopy(default)
small.aa_decoder_depth = 1
small.encoder_depth = 1
small.distogram_block = "mlp"

small_inner = deepcopy(small)
small_inner.distogram_block = "inner"

test_deterministic = deepcopy(small_inner)
test_deterministic.noise_encoder = 0.0
test_deterministic.input_diffusion = False
test_deterministic.latent_diffusion = False
test_deterministic.eval = True  

