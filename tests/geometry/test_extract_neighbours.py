import numpy as np
import pytest
import torch
import haiku as hk

def _jax_extract_neighbours(pos_np, resi_np, chain_np, batch_np, mask_np,
                           num_index, num_spatial, num_random, key):
    import jax
    from salad.aflib.model.geometry import Vec3Array as JaxVec3
    from salad.modules.utils.geometry import extract_neighbours as jax_extract_neighbours

    def f(pos, resi, chain, batch, mask):
        return jax_extract_neighbours(num_index, num_spatial, num_random)(
            pos, resi, chain, batch, mask
        )

    tr = hk.transform(f)
    neigh = tr.apply(
        params=None,
        rng=key,
        pos=JaxVec3.from_array(pos_np),
        resi=resi_np,
        chain=chain_np,
        batch=batch_np,
        mask=mask_np,
    )
    return np.asarray(neigh)


def _torch_extract_neighbours(pos_np, resi_np, chain_np, batch_np, mask_np,
                             num_index, num_spatial, num_random):
    from caesar.utils.geometry import Vec3Array as TorchVec3
    from caesar.modules.utils import extract_neighbours as torch_extract_neighbours

    pos_t = TorchVec3.from_array(torch.tensor(pos_np, dtype=torch.float32))
    neigh_t = torch_extract_neighbours(num_index, num_spatial, num_random)(
        pos_t,
        torch.tensor(resi_np, dtype=torch.long),
        torch.tensor(chain_np, dtype=torch.long),
        torch.tensor(batch_np, dtype=torch.long),
        torch.tensor(mask_np, dtype=torch.float32),
    )
    return neigh_t.detach().cpu().numpy()


@pytest.mark.parametrize("num_index,num_spatial", [(8, 8), (16, 16)])
def test_extract_neighbours_deterministic_matches_jax(protein_data, seed, num_index, num_spatial):
    """
    extract_neighbours deterministic test
    """
    import jax

    num_random = 0

    pos_np = protein_data["all_atom_positions"]
    mask_np = protein_data["residue_mask"]
    resi_np = protein_data["residue_index"]
    chain_np = protein_data["chain_index"]
    batch_np = protein_data["batch_index"]

    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, num_index)
    key = jax.random.fold_in(key, num_spatial)
    key = jax.random.fold_in(key, num_random)

    neigh_j = _jax_extract_neighbours(pos_np, resi_np, chain_np, batch_np, mask_np,
                                     num_index, num_spatial, num_random, key)

    torch.manual_seed(seed)
    neigh_t = _torch_extract_neighbours(pos_np, resi_np, chain_np, batch_np, mask_np,
                                        num_index, num_spatial, num_random)

    assert neigh_t.shape == neigh_j.shape
    assert np.array_equal(neigh_t, neigh_j), "extract_neighbours mismatch in deterministic mode"


@pytest.mark.parametrize("num_index,num_spatial,num_random", [(8, 8, 8), (16, 16, 32)])
def test_extract_neighbours_stochastic_is_reproducible_within_each_framework(
    protein_data, seed, num_index, num_spatial, num_random
):
    """
    test separately jax and torch stochastic extract_neighbours reproducibility
    given same rng / seed
    """
    import jax

    pos_np = protein_data["all_atom_positions"]
    mask_np = protein_data["residue_mask"]
    resi_np = protein_data["residue_index"]
    chain_np = protein_data["chain_index"]
    batch_np = protein_data["batch_index"]

    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, num_index)
    key = jax.random.fold_in(key, num_spatial)
    key = jax.random.fold_in(key, num_random)

    neigh_j1 = _jax_extract_neighbours(pos_np, resi_np, chain_np, batch_np, mask_np,
                                      num_index, num_spatial, num_random, key)
    neigh_j2 = _jax_extract_neighbours(pos_np, resi_np, chain_np, batch_np, mask_np,
                                      num_index, num_spatial, num_random, key)
    assert np.array_equal(neigh_j1, neigh_j2), "JAX stochastic neighbours are not reproducible for same rng"

    torch.manual_seed(seed)
    neigh_t1 = _torch_extract_neighbours(pos_np, resi_np, chain_np, batch_np, mask_np,
                                         num_index, num_spatial, num_random)
    torch.manual_seed(seed)
    neigh_t2 = _torch_extract_neighbours(pos_np, resi_np, chain_np, batch_np, mask_np,
                                         num_index, num_spatial, num_random)
    assert np.array_equal(neigh_t1, neigh_t2), "Torch stochastic neighbours are not reproducible for same seed"
