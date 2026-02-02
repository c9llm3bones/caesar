import numpy as np
import pytest
import torch

from caesar.utils.geometry import Vec3Array as TVec3
from caesar.modules.utils.geometry import extract_neighbours as t_extract_neighbours
from caesar.modules.geometric import (
    direction_features as t_direction_features,
    position_rotation_features as t_position_rotation_features,
    pair_vector_features as t_pair_vector_features,
)


def _to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.array(x)


@pytest.mark.slow
def test_init_local_geofeatures_match_jax(jax_keys, protein_data, torch_device):
    try:
        import jax.numpy as jnp
        import haiku as hk
        from salad.aflib.model.geometry import Vec3Array as JVec3
        from salad.modules.utils.geometry import extract_neighbours as j_extract_neighbours
        from salad.modules.geometric import (
            direction_features as j_direction_features,
            position_rotation_features as j_position_rotation_features,
            pair_vector_features as j_pair_vector_features,
        )
    except Exception as e:
        pytest.skip(f"JAX/salad not available or missing funcs: {e}")

    _key_init, key_apply = jax_keys

    pos_gt = protein_data["pos_gt"].astype(np.float32)

    pos_t = TVec3.from_array(torch.tensor(pos_gt, device=torch_device))
    resi_t = torch.tensor(protein_data["residue_index"], device=torch_device).long()
    chain_t = torch.tensor(protein_data["chain_index"], device=torch_device).long()
    batch_t = torch.tensor(protein_data["batch_index"], device=torch_device).long()
    mask_t = torch.tensor(protein_data["mask"], device=torch_device).bool()

    neigh_t = t_extract_neighbours(5, 5, 0)(pos_t, resi_t, chain_t, batch_t, mask_t)  # (N,K)

    pos_j = JVec3.from_array(jnp.array(pos_gt, dtype=jnp.float32))
    resi_j = jnp.array(protein_data["residue_index"], dtype=jnp.int32)
    chain_j = jnp.array(protein_data["chain_index"], dtype=jnp.int32)
    batch_j = jnp.array(protein_data["batch_index"], dtype=jnp.int32)
    mask_j = jnp.array(protein_data["mask"]).astype(jnp.bool_)

    neigh_np = _to_np(neigh_t).astype(np.int32)
    neigh_j = jnp.array(neigh_np, dtype=jnp.int32)

    def f_neigh(pos, resi, chain, batch, mask):
        return j_extract_neighbours(5, 5, 0)(pos, resi, chain, batch, mask)
    tf_neigh = hk.transform(f_neigh)
    params0 = tf_neigh.init(key_apply, pos_j, resi_j, chain_j, batch_j, mask_j)
    neigh_j_ref = np.array(tf_neigh.apply(params0, key_apply, pos_j, resi_j, chain_j, batch_j, mask_j))
    assert np.array_equal(neigh_np, neigh_j_ref), "neighbours differ; feature parity is not meaningful"

    # Torch
    dir_t = _to_np(t_direction_features(pos_t, neigh_t))
    rot_t = _to_np(t_position_rotation_features(pos_t, neigh_t))
    vec_t = _to_np(t_pair_vector_features(pos_t, neigh_t))

    # JAX 
    def f_feats(pos, neigh):
        return (
            j_direction_features(pos, neigh),
            j_position_rotation_features(pos, neigh),
            j_pair_vector_features(pos, neigh),
        )

    tf = hk.transform(f_feats)
    params = tf.init(key_apply, pos_j, neigh_j)
    dir_j, rot_j, vec_j = tf.apply(params, key_apply, pos_j, neigh_j)
    dir_j, rot_j, vec_j = np.array(dir_j), np.array(rot_j), np.array(vec_j)

    print("\nDIR torch/jax mean,std:", dir_t.mean(), dir_t.std(), "/", dir_j.mean(), dir_j.std())
    print("ROT torch/jax mean,std:", rot_t.mean(), rot_t.std(), "/", rot_j.mean(), rot_j.std())
    print("VEC torch/jax mean,std:", vec_t.mean(), vec_t.std(), "/", vec_j.mean(), vec_j.std())

    np.testing.assert_allclose(dir_t, dir_j, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rot_t, rot_j, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vec_t, vec_j, rtol=1e-4, atol=1e-4)
