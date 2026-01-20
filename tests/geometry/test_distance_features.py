import numpy as np
import pytest
import torch
import haiku as hk

import jax
import jax.numpy as jnp

from salad.aflib.model.geometry import Vec3Array as JaxVec3
from salad.modules.utils.geometry import extract_neighbours as jax_extract_neighbours
from salad.modules.geometric import distance_features as jax_distance_features

from caesar.utils.geometry import Vec3Array as TorchVec3  
from caesar.modules.geometric import distance_features as torch_distance_features


@pytest.mark.parametrize(
    "num_index,num_spatial,num_random",
    [
        (8, 8, 0),   # deterministic
        (8, 8, 8),   # stochastic, but neighbours from jax
    ],
)
def test_distance_features_matches_jax_using_real_extract_neighbours(
    protein_data,
    num_index,
    num_spatial,
    num_random,
    atol,
    rtol,
    seed,   
):
    pos_np = protein_data["all_atom_positions"].astype(np.float32)  
    mask_np = protein_data["residue_mask"]
    resi_np = protein_data["residue_index"]
    chain_np = protein_data["chain_index"]
    batch_np = protein_data["batch_index"]

    def jax_neigh_fn(pos, resi, chain, batch, mask):
        return jax_extract_neighbours(num_index, num_spatial, num_random)(
            pos, resi, chain, batch, mask
        )

    jax_neigh = hk.transform(jax_neigh_fn)

    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, num_index)
    key = jax.random.fold_in(key, num_spatial)
    key = jax.random.fold_in(key, num_random)

    neighbours_jax = jax_neigh.apply(
        params=None,
        rng=key,
        pos=JaxVec3.from_array(pos_np),
        resi=resi_np,
        chain=chain_np,
        batch=batch_np,
        mask=mask_np,
    )

    jax_out = jax_distance_features(
        JaxVec3.from_array(pos_np),
        neighbours_jax,
        d_min=0.0,
        d_max=22.0,
    )
    jax_out_np = np.asarray(jax_out)

    pos_t = TorchVec3.from_array(torch.tensor(pos_np, dtype=torch.float32))
    neighbours_t = torch.tensor(np.asarray(neighbours_jax), dtype=torch.long)

    torch_out = torch_distance_features(
        pos_t,
        neighbours_t,
        d_min=0.0,
        d_max=22.0,
    ).detach().cpu().numpy()

    assert torch_out.shape == jax_out_np.shape
    np.testing.assert_allclose(torch_out, jax_out_np, atol=atol, rtol=rtol)
