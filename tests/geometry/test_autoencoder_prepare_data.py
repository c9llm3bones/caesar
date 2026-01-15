import numpy as np
import pytest
import torch

import jax
import jax.numpy as jnp
import haiku as hk


def _to_jax(x):
    # np -> jax array, torch -> np -> jax
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return jnp.asarray(x)


def _to_torch(x, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def _assert_allclose(name, torch_x, jax_x, atol, rtol):
    tx = torch_x.detach().cpu().numpy() if isinstance(torch_x, torch.Tensor) else np.asarray(torch_x)
    jx = np.asarray(jax_x)
    assert tx.shape == jx.shape, f"{name}: shape mismatch torch={tx.shape} jax={jx.shape}"
    np.testing.assert_allclose(tx, jx, atol=atol, rtol=rtol, err_msg=f"{name}: mismatch")

@pytest.mark.parametrize("rng_seed", [0])
def test_prepare_data_matches_jax_deterministic_parts(
    protein_data,
    atol,
    rtol,
    rng_seed,
):
    """
    Golden test: JAX SALAD prepare_data vs Torch prepare_data.

    Пока сравниваем только детерминированные компоненты.
    RNG-зависимые:
      - out['pos'] (randn init)
      - out['dmap'] (CB + 0.3*noise)
    их пока проверяем только по shape/finite.
    """

    from salad.modules.structure_autoencoder import StructureAutoencoder as JaxStructureAutoencoder

    from caesar.modules.autoencoder import StructureAutoencoder as TorchStructureAutoencoder

    #  minimal config stub
    class _DummyCfg:
        pass

    # protein_data from npz
    data_np = dict(protein_data)

    data_jax = {k: _to_jax(v) for k, v in data_np.items()}

    data_torch = {k: _to_torch(v) for k, v in data_np.items()}

    def jax_fn(data):
        model = JaxStructureAutoencoder(config=_DummyCfg())
        return model.prepare_data(data)
    jax_t = hk.transform(jax_fn)
    rng = jax.random.PRNGKey(rng_seed)
    params = jax_t.init(rng, data_jax)
    rng_apply = jax.random.PRNGKey(rng_seed + 1)
    out_jax = jax_t.apply(params, rng_apply, data_jax)
    
    torch_model = TorchStructureAutoencoder(config=_DummyCfg())
    out_torch = torch_model.prepare_data(data_torch)

    # keys that should match exactly (deterministic parts) 
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
        # "dssp", # FIXME: DSSP is not matching currently
    ]
    
    _assert_allclose("atom_pos", out_torch["atom_pos"], out_jax["atom_pos"], atol, rtol)
    _assert_allclose("mask", out_torch["mask"], out_jax["mask"], atol, rtol)

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
            _assert_allclose(k, out_torch[k], out_jax[k], atol=atol, rtol=rtol)

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
