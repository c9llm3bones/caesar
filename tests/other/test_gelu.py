def test_gelu_parity(jax_keys):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import jax
    import jax.numpy as jnp

    x = np.random.RandomState(0).randn(1000).astype(np.float32) * 3
    j = np.array(jax.nn.gelu(jnp.array(x)))

    t_exact = F.gelu(torch.tensor(x), approximate="none").numpy()
    t_tanh  = F.gelu(torch.tensor(x), approximate="tanh").numpy()

    err_exact = np.max(np.abs(j - t_exact))
    err_tanh  = np.max(np.abs(j - t_tanh))
    dis  = np.max(np.abs(t_exact - t_tanh))
    print("maxabs(jax - torch_exact) =", err_exact)
    print("maxabs(jax - torch_tanh)  =", err_tanh)
    print("dis = ", dis)
    assert err_tanh < err_exact
