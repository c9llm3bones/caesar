import numpy as np
import pytest
import torch

import jax
import jax.numpy as jnp
import haiku as hk

from tests.utils import to_jax, to_torch, assert_allclose, assert_array_equal


def test_encoder_prepare_features_preparams_matches_jax(
    protein_data,
    atol,
    rtol,
    seed,
    jax_keys,
    cfg):
    """
      - pos_input (without noise_encoder for deterministic features)
      - neighbours = extract_neighbours(5,5,0)
      - raw pair feature components:
          sequence_relative_position(8, one_hot=True, pseudo_chains=True)
          distance_features / direction_features / position_rotation_features / pair_vector_features
    """

    from salad.modules.structure_autoencoder import StructureAutoencoder as JaxAE
    from salad.aflib.model.geometry import Vec3Array as JaxVec3
    from salad.modules.utils.geometry import extract_neighbours as jax_extract_neighbours
    from salad.modules.utils.geometry import sequence_relative_position as jax_sequence_relative_position
    from salad.modules.geometric import (
        distance_features as jax_distance_features,
        direction_features as jax_direction_features,
        position_rotation_features as jax_position_rotation_features,
        pair_vector_features as jax_pair_vector_features,
    )

    from caesar.modules.autoencoder import StructureAutoencoder as TorchAE
    from caesar.modules.autoencoder import prepare_data as torch_prepare_data
    from caesar.utils.geometry import Vec3Array as TorchVec3
    from caesar.modules.utils import extract_neighbours as torch_extract_neighbours
    from caesar.modules.utils.geometry import sequence_relative_position as torch_sequence_relative_position
    from caesar.modules.geometric import (
        distance_features as torch_distance_features,
        direction_features as torch_direction_features,
        position_rotation_features as torch_position_rotation_features,
        pair_vector_features as torch_pair_vector_features,
    )
    c = cfg 

    data_np = dict(protein_data)
    data_jax = {k: to_jax(v) for k, v in data_np.items()}
    data_torch = {k: to_torch(v) for k, v in data_np.items()}

    def jax_prepare_data_fn(d):
        m = JaxAE(config=c)
        return m.prepare_data(d)

    jax_pd = hk.transform(jax_prepare_data_fn)
    
    key_init, key_apply = jax_keys
    params = jax_pd.init(key_init, data_jax)
    out_jax_pd = jax_pd.apply(params, key_apply, data_jax)

    enc_in_jax = dict(out_jax_pd)
    enc_in_jax["residue_index"] = data_jax["residue_index"]
    enc_in_jax["batch_index"] = data_jax["batch_index"]

    torch_m = TorchAE(config=c)
    torch_m.eval()
    out_t_pd = torch_prepare_data(data_torch)

    enc_in_torch = dict(out_t_pd)
    enc_in_torch["residue_index"] = data_torch["residue_index"]
    enc_in_torch["batch_index"] = data_torch["batch_index"]

    # sanity: inputs used by pre-param computation 
    assert_allclose("pos_input", enc_in_torch["pos_input"], enc_in_jax["pos_input"], atol, rtol)
    assert_allclose("mask", enc_in_torch["mask"], enc_in_jax["mask"], atol, rtol)
    assert_array_equal("batch_index", enc_in_torch["batch_index"].cpu(), enc_in_jax["batch_index"])

    # JAX: compute neighbours + raw components
    
    def jax_raw_components_fn(d):
        pos = d["pos_input"]
        resi = d["residue_index"]
        chain = d["chain_index"]
        batch = d["batch_index"]
        mask = d["mask"]

        pos_v = JaxVec3.from_array(pos)
        neigh = jax_extract_neighbours(5, 5, 0)(pos_v, resi, chain, batch, mask)

        relpos = jax_sequence_relative_position(8, one_hot=True, pseudo_chains=True)(
            resi, chain, batch, neigh
        )
        dist = jax_distance_features(pos_v, neigh, d_min=0.0, d_max=22.0)
        dire = jax_direction_features(pos_v, neigh)
        rot  = jax_position_rotation_features(pos_v, neigh)
        vec  = jax_pair_vector_features(pos_v, neigh)

        return neigh, relpos, dist, dire, rot, vec, pos_v.to_array()

    jax_raw = hk.transform(jax_raw_components_fn)

    key_raw = jax.random.PRNGKey(seed)
    key_raw_init, key_raw_apply = jax.random.split(key_raw, 2)

    params_raw = jax_raw.init(key_raw_init, enc_in_jax)
    neigh_jax, relpos_jax, dist_jax, dire_jax, rot_jax, vec_jax, pos_jax = jax_raw.apply(
        params_raw, key_raw_apply, enc_in_jax
    )
    
    pos_t = enc_in_torch["pos_input"]
    resi_t = enc_in_torch["residue_index"].long()
    chain_t = enc_in_torch["chain_index"].long()
    batch_t = enc_in_torch["batch_index"].long()
    mask_t = enc_in_torch["mask"]

    pos_v_t = TorchVec3.from_array(pos_t)
    neigh_t = torch_extract_neighbours(5, 5, 0)(pos_v_t, resi_t, chain_t, batch_t, mask_t)

    relpos_t = torch_sequence_relative_position(8, one_hot=True, pseudo_chains=True)(
        resi_t, chain_t, batch_t, neigh_t
    )
    dist_t = torch_distance_features(pos_v_t, neigh_t, d_min=0.0, d_max=22.0)
    dire_t = torch_direction_features(pos_v_t, neigh_t)
    rot_t = torch_position_rotation_features(pos_v_t, neigh_t)
    vec_t = torch_pair_vector_features(pos_v_t, neigh_t)

    assert_allclose("pos_preparams", pos_v_t.to_tensor(), pos_jax, atol, rtol)
    assert_array_equal("neighbours", neigh_t, neigh_jax)

    assert_array_equal("relpos(onehot) exact", relpos_t, relpos_jax) 
    assert_allclose("distance_features", dist_t, dist_jax, atol, rtol)
    assert_allclose("direction_features", dire_t, dire_jax, atol, rtol)
    assert_allclose("position_rotation_features", rot_t, rot_jax, atol, rtol)
    assert_allclose("pair_vector_features", vec_t, vec_jax, atol, rtol)
