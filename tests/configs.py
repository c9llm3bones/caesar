# tests/configs.py
from caesar.modules.utils.collections import dotdict, deepcopy
import numpy as np


default = dotdict(
    ## feature sizes
    local_size=128,
    pair_size=64,
    latent_size=20,
    ## module parameters
    relative_position_encoding_max=32,
    factor=4,
    heads=8,
    key_size=32,
    multi_query=False,
    sigma_data=10.0,
    ## diffusion stack parameters
    depth=6,
    block_size=1,
    num_recycle=0,
    ## aa diffusion parameters
    # no sequence diffusion
    aa_decoder_depth=3,
    encoder_depth=3,
    ## loss parameters
    # loss clipping
    p_clip=1.0,
    clip_fape=100,
    # time embedding
    time_embedding=False,
    # local loss parameters
    local_neighbours=16,
    fape_neighbours=64,
    # loss weights
    local_weight=1.0,
    full_atom_weight=0.0,
    aa_weight=10.0,
    fape_weight=1,
    fape_trajectory_weight=0.5,
    sidechain_decoder="angles",
    atom14_masked_input="local",
    atom14_masked_backbone_source="predicted",
    full_atom_loss_mode="local",
    atom37_encoder_mode="none",
    # dataset constraints
    min_size=50,
    max_size=None
)

small = deepcopy(default)
small.aa_decoder_depth = 1
small.encoder_depth = 1
small.distogram_block = "mlp"

small_inner = deepcopy(small)
small_inner.distogram_block = "inner"

test_deterministic = deepcopy(small_inner)
test_deterministic.num_recycle = 0
test_deterministic.noise_encoder = 0.0
test_deterministic.input_diffusion = False
test_deterministic.num_random_neighbours = 0
test_deterministic.latent_diffusion = False
test_deterministic.eval = True  
# test_deterministic.fape_neighbours = 0
