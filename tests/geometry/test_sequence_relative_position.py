import numpy as np
import pytest
import torch

from caesar.modules.utils.geometry import sequence_relative_position as t_relpos
from caesar.utils.geometry import Vec3Array as TVec3
from caesar.modules.utils.geometry import extract_neighbours as t_extract_neighbours


@pytest.mark.slow
def test_sequence_relative_position_matches_jax(cfg_deterministic, jax_keys, protein_data, torch_device):
    try:
        import jax.numpy as jnp
        import haiku as hk
        from salad.modules.utils.geometry import sequence_relative_position as j_relpos
        from salad.aflib.model.geometry import Vec3Array as JVec3
        from salad.modules.utils.geometry import extract_neighbours as j_extract_neighbours
    except Exception as e:
        pytest.skip(f"JAX/salad not available: {e}")

    cfg = cfg_deterministic
    count = int(cfg.relative_position_encoding_max)

    pos_gt = protein_data["pos_gt"].astype(np.float32)

    # torch inputs
    pos_t = TVec3.from_array(torch.tensor(pos_gt, device=torch_device))
    resi_t = torch.tensor(protein_data["residue_index"], device=torch_device).long()
    chain_t = torch.tensor(protein_data["chain_index"], device=torch_device).long()
    batch_t = torch.tensor(protein_data["batch_index"], device=torch_device).long()
    mask_t = torch.tensor(protein_data["mask"], device=torch_device).bool()

    # jax inputs
    pos_j = JVec3.from_array(jnp.array(pos_gt, dtype=jnp.float32))
    resi_j = jnp.array(protein_data["residue_index"], dtype=jnp.int32)
    chain_j = jnp.array(protein_data["chain_index"], dtype=jnp.int32)
    batch_j = jnp.array(protein_data["batch_index"], dtype=jnp.int32)
    mask_j = jnp.array(protein_data["mask"]).astype(jnp.bool_)

    _key_init, key_apply = jax_keys

    neigh_t = t_extract_neighbours(5, 5, 0)(pos_t, resi_t, chain_t, batch_t, mask_t).cpu().numpy()

    def f(pos, resi, chain, batch, mask, neigh):
        return j_relpos(count, one_hot=True)(resi, chain, batch, neigh)

    tf = hk.transform(lambda pos, resi, chain, batch, mask, neigh: f(pos, resi, chain, batch, mask, neigh))
    params = tf.init(key_apply, pos_j, resi_j, chain_j, batch_j, mask_j, jnp.array(neigh_t, dtype=jnp.int32))
    rel_j = np.array(tf.apply(params, key_apply, pos_j, resi_j, chain_j, batch_j, mask_j, jnp.array(neigh_t, dtype=jnp.int32)))

    rel_t = t_relpos(count, one_hot=True)(resi_t, chain_t, batch_t, torch.tensor(neigh_t, device=torch_device).long())
    rel_t = rel_t.detach().cpu().numpy()

    assert rel_t.shape == rel_j.shape
    assert np.array_equal(rel_t.astype(np.int32), rel_j.astype(np.int32))
