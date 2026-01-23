import numpy as np
import pytest
import torch

import jax
import jax.numpy as jnp
import haiku as hk
from tests.utils import assert_allclose
from tests.utils import to_jax 
from tests.utils import to_torch


def test_prepare_data_matches_jax_deterministic_parts(
    protein_data,
    atol,
    rtol,
    jax_keys,
    cfg,
):
    """
    SALAD prepare_data vs Torch prepare_data.
    """

    from salad.modules.structure_autoencoder import StructureAutoencoder as JaxStructureAutoencoder

    from caesar.modules.autoencoder import prepare_data as torch_prepare_data

    data_np = dict(protein_data)

    data_jax = {k: to_jax(v) for k, v in data_np.items()}

    data_torch = {k: to_torch(v) for k, v in data_np.items()}

    def jax_fn(data):
        model = JaxStructureAutoencoder(config=cfg)
        return model.prepare_data(data)
    
    jax_t = hk.transform(jax_fn)
    
    key_init, key_apply = jax_keys
    params = jax_t.init(key_init, data_jax)
    out_jax = jax_t.apply(params, key_apply, data_jax)
    
    out_torch = torch_prepare_data(data_torch)

    deterministic_keys = [
        "pos_gt",
        "pos_input",
        "chain_index",
        "mask",
        "atom_pos",
        "atom_mask",
        "all_atom_positions",
        "all_atom_mask",
        "dmap_mask",
        # "dssp", # FIXME: DSSP is not matching currently (fix dssp.py)
    ]
    
    assert_allclose("atom_pos", out_torch["atom_pos"], out_jax["atom_pos"], atol, rtol)
    assert_allclose("mask", out_torch["mask"], out_jax["mask"], atol, rtol)

    np.testing.assert_array_equal(
        data_torch["batch_index"].cpu().numpy(),
        np.asarray(data_jax["batch_index"])
    )

    for k in deterministic_keys:
        assert k in out_jax, f"JAX output missing key {k}"
        assert k in out_torch, f"Torch output missing key {k}"

        if k in ("dmap_mask",):
            tx = out_torch[k].to(torch.bool)
            jx = np.asarray(out_jax[k]).astype(np.bool_)
            assert tx.shape == jx.shape, f"{k}: shape mismatch"
            assert np.array_equal(tx.detach().cpu().numpy(), jx), f"{k}: mismatch"
        else:
            assert_allclose(k, out_torch[k], out_jax[k], atol=atol, rtol=rtol)

    stochastic_keys = ["pos", "dmap"]
    for k in stochastic_keys:
        assert k in out_torch and k in out_jax
        tx = out_torch[k]
        jx = np.asarray(out_jax[k])

        assert tuple(tx.shape) == tuple(jx.shape), f"{k}: shape mismatch torch={tuple(tx.shape)} jax={tuple(jx.shape)}"
        assert torch.isfinite(tx).all().item(), f"{k}: torch contains NaN/Inf"
        assert np.isfinite(jx).all(), f"{k}: jax contains NaN/Inf"

    # extra sanity: pos is random init, shouldn't be identical to pos_gt
    assert not torch.allclose(out_torch["pos"], out_torch["pos_gt"]), "Torch pos init equals pos_gt unexpectedly"
