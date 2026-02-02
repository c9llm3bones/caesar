import numpy as np
import pytest
import torch

from caesar.utils.geometry import Vec3Array as TVec3
from caesar.modules.utils.geometry import extract_neighbours as t_extract_neighbours


@pytest.mark.slow
def test_extract_neighbours_matches_jax(jax_keys, protein_data, torch_device):
    try:
        import jax
        import jax.numpy as jnp
        import haiku as hk
        from salad.modules.utils.geometry import extract_neighbours as j_extract_neighbours
        from salad.aflib.model.geometry import Vec3Array as JVec3
    except Exception as e:
        pytest.skip(f"JAX/salad not available: {e}")

    _key_init, key_apply = jax_keys

    pos_gt = protein_data["pos_gt"].astype(np.float32)

    # Torch inputs
    pos_t = TVec3.from_array(torch.tensor(pos_gt, device=torch_device))
    resi_t = torch.tensor(protein_data["residue_index"], device=torch_device).long()
    chain_t = torch.tensor(protein_data["chain_index"], device=torch_device).long()
    batch_t = torch.tensor(protein_data["batch_index"], device=torch_device).long()
    mask_t = torch.tensor(protein_data["mask"], device=torch_device)
    mask_t = mask_t.bool() if mask_t.dtype != torch.bool else mask_t

    # JAX inputs 
    pos_j = JVec3.from_array(jnp.array(pos_gt))
    resi_j = jnp.array(protein_data["residue_index"]).astype(jnp.int32)
    chain_j = jnp.array(protein_data["chain_index"]).astype(jnp.int32)
    batch_j = jnp.array(protein_data["batch_index"]).astype(jnp.int32)
    mask_j = jnp.array(protein_data["mask"])
    mask_j = mask_j.astype(jnp.bool_) if mask_j.dtype != jnp.bool_ else mask_j

    def f(pos, resi, chain, batch, mask, ni, ns, nr):
        return j_extract_neighbours(num_index=ni, num_spatial=ns, num_random=nr)(pos, resi, chain, batch, mask)

    tf = hk.transform(lambda pos, resi, chain, batch, mask, ni, ns, nr: f(pos, resi, chain, batch, mask, ni, ns, nr))
    params = tf.init(key_apply, pos_j, resi_j, chain_j, batch_j, mask_j, 5, 5, 0)

    for (ni, ns, nr) in [(5, 5, 0), (16, 16, 0)]:
        neigh_t = t_extract_neighbours(num_index=ni, num_spatial=ns, num_random=nr)(
            pos_t, resi_t, chain_t, batch_t, mask_t
        ).detach().cpu().numpy()

        neigh_j = np.array(tf.apply(params, key_apply, pos_j, resi_j, chain_j, batch_j, mask_j, ni, ns, nr))

        print(f"\n(neighbours) ni={ni} ns={ns} nr={nr}")
        print("torch[:3,:10]:\n", neigh_t[:3, :10])
        print("jax  [:3,:10]:\n", neigh_j[:3, :10])

        assert neigh_t.shape == neigh_j.shape
        assert np.array_equal(neigh_t, neigh_j), "Neighbour indices mismatch"
