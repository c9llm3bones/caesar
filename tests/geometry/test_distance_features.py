import numpy as np
import pytest
import torch

import jax
import jax.numpy as jnp
import haiku as hk
from salad.modules.geometric import distance_features as jax_distance_features
from salad.aflib.model.geometry import Vec3Array as JaxVec3

from caesar.modules.geometric import distance_features as torch_distance_features
from caesar.geometry import Vec3Array as TorchVec3

@pytest.mark.parametrize(
    "num_index,num_spatial,num_random",
    [
        (8, 8, 0),   # deterministic
        (8, 8, 8),   # stochastic
    ],
)
def test_distance_features_matches_jax(
    protein_data,
    num_index,
    num_spatial,
    num_random,
    atol,
    rtol,
):
    
    """
    Compare distance_features between JAX-SALAD and PyTorch implementation.

    - num_random = 0:  strict numerical equivalence
    - num_random > 0:  shape + statistical equivalence (stochastic case)
    """

    from salad.modules.utils.geometry import extract_neighbours as jax_extract_neighbours
    
    from caesar.geometry import extract_neighbours as torch_extract_neighbours
    pos_np = protein_data["all_atom_positions"]  # (N=113, M=14, 3)
    mask_np = protein_data["residue_mask"]
    resi = protein_data["residue_index"]
    chain = protein_data["chain_index"]
    batch = protein_data["batch_index"]
    
    # jax perverted syntax for hk.transform
    def jax_neigh_fn(pos, resi, chain, batch, mask):
        return jax_extract_neighbours(
            num_index,
            num_spatial,
            num_random,
        )(pos, resi, chain, batch, mask)

    jax_neigh = hk.transform(jax_neigh_fn)

    neighbours_jax = jax_neigh.apply(
        params=None,
        rng=jax.random.PRNGKey(0),
        pos=JaxVec3.from_array(pos_np),
        resi=resi,
        chain=chain,
        batch=batch,
        mask=mask_np,
    )

    jax_out = jax_distance_features(
        JaxVec3.from_array(pos_np),
        neighbours_jax,
        d_min=0.0,
        d_max=22.0,
    )
    jax_out = np.asarray(jax_out)

    pos_torch = TorchVec3.from_array(
        torch.tensor(pos_np, dtype=torch.float32)
    )

    neighbours_torch = torch_extract_neighbours(
        num_index,
        num_spatial,
        num_random,
    )(
        pos_torch,
        torch.tensor(resi),
        torch.tensor(chain),
        torch.tensor(batch),
        torch.tensor(mask_np),
    )

    torch_out = torch_distance_features(
        pos_torch,
        neighbours_torch,
        d_min=0.0,
        d_max=22.0,
    )
    torch_out = torch_out.detach().cpu().numpy()

    assert jax_out.shape == torch_out.shape


    if num_random == 0: # deterministic case
        np.testing.assert_allclose(
            torch_out,
            jax_out,
            rtol=rtol,
            atol=atol,
            err_msg="distance_features mismatch (deterministic case)",
        )
    # TODO: change stoch case for determenistic randomization.
    else: # stochastic case
        np.testing.assert_allclose(
            torch_out.mean(),
            jax_out.mean(),
            atol=1e-3,
            err_msg="mean mismatch (stochastic case)",
        )
        np.testing.assert_allclose(
            torch_out.std(),
            jax_out.std(),
            atol=1e-3,
            err_msg="std mismatch (stochastic case)",
        )
