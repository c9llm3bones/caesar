import numpy as np
import pytest
import torch

from caesar.modules.utils.geometry import distance_rbf as t_rbf
from caesar.utils.geometry import Vec3Array as TVec3
from caesar.modules.utils.geometry import extract_neighbours as t_extract_neighbours


@pytest.mark.slow
def test_distance_rbf_matches_jax(cfg_deterministic, jax_keys, protein_data, torch_device):
    try:
        import jax.numpy as jnp
        from salad.modules.utils.geometry import distance_rbf as j_rbf
        from salad.aflib.model.geometry import Vec3Array as JVec3
    except Exception as e:
        pytest.skip(f"JAX/salad not available: {e}")

    bins = 64  

    pos_gt = protein_data["pos_gt"].astype(np.float32)
    N = pos_gt.shape[0]

    pos_t = TVec3.from_array(torch.tensor(pos_gt, device=torch_device))
    resi_t = torch.tensor(protein_data["residue_index"], device=torch_device).long()
    chain_t = torch.tensor(protein_data["chain_index"], device=torch_device).long()
    batch_t = torch.tensor(protein_data["batch_index"], device=torch_device).long()
    mask_t = torch.tensor(protein_data["mask"], device=torch_device).bool()

    neigh_t = t_extract_neighbours(5, 5, 0)(pos_t, resi_t, chain_t, batch_t, mask_t)  # (N,K)

    ca_t = pos_t[:, 1].to_tensor()  # (N,3)
    idx = torch.arange(N, device=torch_device)[:, None]
    nb = neigh_t.clamp_min(0)
    d_t = torch.linalg.norm(ca_t[idx] - ca_t[nb], dim=-1)  # (N,K)
    rbf_t = t_rbf(d_t, 0.0, 22.0, bins=bins).detach().cpu().numpy()

    pos_j = JVec3.from_array(jnp.array(pos_gt, dtype=jnp.float32))
    ca_j = pos_j[:, 1].to_array()  # (N,3)
    neigh_j = jnp.array(neigh_t.detach().cpu().numpy(), dtype=jnp.int32)
    nbj = jnp.maximum(neigh_j, 0)
    dj = jnp.linalg.norm(ca_j[:, None, :] - ca_j[nbj], axis=-1)
    rbf_j = np.array(j_rbf(dj, 0.0, 22.0, bins=bins))

    np.testing.assert_allclose(rbf_t[:2, :5, :10], rbf_j[:2, :5, :10], atol=1e-6, rtol=1e-6)
