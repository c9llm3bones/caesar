import numpy as np
import jax.numpy as jnp
import torch

def to_jax(x):
    # np -> jax array, torch -> np -> jax
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return jnp.asarray(x)


def to_torch(x, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def assert_allclose(name, torch_x, jax_x, atol, rtol):
    tx = torch_x.detach().cpu().numpy() if isinstance(torch_x, torch.Tensor) else np.asarray(torch_x)
    jx = np.asarray(jax_x)
    assert tx.shape == jx.shape, f"{name}: shape mismatch torch={tx.shape} jax={jx.shape}"
    np.testing.assert_allclose(tx, jx, atol=atol, rtol=rtol, err_msg=f"{name}: mismatch")
