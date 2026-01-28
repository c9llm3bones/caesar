import pytest
import numpy as np
import torch
import haiku as hk
import jax
import jax.numpy as jnp

from caesar.modules.utils.collections import deepcopy
from tests.utils import to_jax, to_torch, assert_allclose

# deterministic helpers (no RNG)
def det_random_neighbours_jax(K: int, eps: float = 1e-6):
    def fn(distance, batch, mask):
        # distance: (N,N) with inf for invalid pairs
        N = distance.shape[0]
        tie = (jnp.arange(N, dtype=distance.dtype)[None, :] * jnp.asarray(eps, distance.dtype))
        dist_sort = distance + tie

        idx = jnp.argsort(dist_sort, axis=-1)[:, :K]
        dsel = jnp.take_along_axis(distance, idx, axis=-1)
        valid = jnp.isfinite(dsel)
        return jnp.where(valid, idx, -jnp.ones_like(idx))
    return fn


def det_random_neighbours_torch(K: int, eps: float = 1e-6):
    def fn(distance, batch, mask):
        N = distance.shape[0]
        tie = torch.arange(N, device=distance.device, dtype=distance.dtype)[None, :] * eps
        dist_sort = distance + tie

        idx = torch.argsort(dist_sort, dim=-1)[:, :K]
        dsel = distance.gather(dim=-1, index=idx)
        valid = torch.isfinite(dsel)
        return torch.where(valid, idx, torch.full_like(idx, -1))
    return fn



def det_spatial_neighbours_jax(count: int, eps: float = 1e-6):
    def fn(cb_pos, batch, mask):
        cb = jnp.asarray(cb_pos.to_array())  # (N,3)
        diff = cb[:, None, :] - cb[None, :, :]
        dist = jnp.sqrt((diff * diff).sum(axis=-1))

        pair = (batch[:, None] == batch[None, :]) & (mask[:, None] > 0) & (mask[None, :] > 0)
        dist = jnp.where(pair, dist, jnp.inf)
        dist = dist.at[jnp.arange(dist.shape[0]), jnp.arange(dist.shape[0])].set(jnp.inf)

       
        N = dist.shape[0]
        tie = (jnp.arange(N, dtype=dist.dtype)[None, :] * jnp.asarray(eps, dist.dtype))
        dist_sort = dist + tie

        idx = jnp.argsort(dist_sort, axis=-1)[:, :count]
        dsel = jnp.take_along_axis(dist, idx, axis=-1)   
        valid = jnp.isfinite(dsel)
        return jnp.where(valid, idx, -jnp.ones_like(idx))
    return fn


def det_spatial_neighbours_torch(count: int, eps: float = 1e-6):
    def fn(cb_pos, batch, mask):
        cb = cb_pos.to_tensor()  # (N,3)
        diff = cb[:, None, :] - cb[None, :, :]
        dist = torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=1e-12))

        pair = (batch[:, None] == batch[None, :]) & (mask[:, None] > 0) & (mask[None, :] > 0)
        dist = torch.where(pair, dist, torch.full_like(dist, float("inf")))
        dist = dist.clone()  
        dist.fill_diagonal_(float("inf"))

        N = dist.shape[0]
        tie = (torch.arange(N, device=dist.device, dtype=dist.dtype)[None, :] * eps)
        dist_sort = dist + tie

        idx = torch.argsort(dist_sort, dim=-1)[:, :count]
        dsel = dist.gather(dim=-1, index=idx)
        valid = torch.isfinite(dsel)
        return torch.where(valid, idx, torch.full_like(idx, -1))
    return fn


def fake_violation_loss_jax(*args, **kwargs):
    aa_gt = args[0]
    N = aa_gt.shape[0]
    return jnp.zeros((N,), dtype=jnp.float32), None


def fake_violation_loss_torch(*args, **kwargs):
    aa_gt = args[0]
    N = aa_gt.shape[0]
    return torch.zeros((N,), dtype=torch.float32, device=aa_gt.device), None

def make_max_cfg():
    from salad.modules.config import distance_to_structure_decoder as config_choices

    cfg = deepcopy(getattr(config_choices, "small_vq"))

    cfg.eval = True
    cfg.is_decoder = False         
    cfg.state = False

    # distogram
    cfg.distogram_block = getattr(cfg, "distogram_block", "inner")
    if cfg.distogram_block == "none":
        cfg.distogram_block = "inner"

    # kabsch
    cfg.kabsch_rmsd = True

    # diffusion + latent loss
    cfg.input_diffusion = True
    cfg.latent_diffusion = True
    cfg.latent_loss_scale = 1.0
    cfg.vp_diffusion = True
    cfg.time_embedding = True

    # violation
    cfg.violation_scale = 0.1

    # fape flags
    cfg.unclipped_weight = 0.1
    cfg.no_fape2 = False

    # ensure required VQ fields exist
    cfg.codebook_size = getattr(cfg, "codebook_size", 4096)
    cfg.codebook_loss_scale = getattr(cfg, "codebook_loss_scale", 1.0)
    cfg.codebook_b = getattr(cfg, "codebook_b", 0.25)

    # ensure local neighbours fields
    cfg.local_neighbours = getattr(cfg, "local_neighbours", 16)
    cfg.fape_neighbours = getattr(cfg, "fape_neighbours", 64)

    # some codes expect equivariance field
    if not hasattr(cfg, "equivariance"):
        cfg.equivariance = None

    return cfg


# synthetic data/result builder
def make_full_inputs(N=64, T=3, K=16, latent_size=20):
    rng = np.random.default_rng(0)

    residue_index = np.arange(N, dtype=np.int32)
    chain_index = np.zeros((N,), dtype=np.int32)

    # 2 proteins in a batch to test batch-masking
    batch_index = np.zeros((N,), dtype=np.int32)
    batch_index[N // 2:] = 1

    mask = np.ones((N,), dtype=np.float32)
    mask[0] = 0.0
    mask[-1] = 0.0

    aa_gt = rng.integers(0, 21, size=(N,), dtype=np.int32)  # includes 20

    # pos_gt: (N, 5, 3) with last atom = CB (as in original: pos_gt[:, -1])
    pos_gt = np.zeros((N, 5, 3), dtype=np.float32)
    for i in range(N):
        ca = np.array([float(i), 0.0, 0.0], dtype=np.float32)
        n  = np.array([float(i) - 0.5,  0.8,  0.0], dtype=np.float32)
        c  = np.array([float(i) + 0.5,  0.8,  0.2], dtype=np.float32)
        o  = c + np.array([0.1, 0.2, -0.1], dtype=np.float32)
        cb = ca + np.array([0.0, -0.8, 0.1], dtype=np.float32)
        pos_gt[i] = np.stack([n, ca, c, o, cb], axis=0)

    cb = pos_gt[:, 4, :]
    diff = cb[:, None, :] - cb[None, :, :]
    dmap = np.sqrt((diff * diff).sum(axis=-1)).astype(np.float32)

    # trajectory: (T,N,5,3)
    trajectory = pos_gt[None, ...] + rng.standard_normal((T, N, 5, 3), dtype=np.float32) * 1e-3

    # atom_pos/atom_mask for local + violation: (N,14,3), (N,14)
    atom_pos = np.zeros((N, 14, 3), dtype=np.float32)
    atom_mask = np.ones((N, 14), dtype=np.float32)
    atom_pos[:, :5, :] = pos_gt
    atom_pos[:, 5:, :] = pos_gt[:, 1:2, :] + rng.standard_normal((N, 9, 3), dtype=np.float32) * 0.1

    # predicted atom positions (for result["atom_pos"])
    pred_atom_pos = atom_pos + rng.standard_normal(atom_pos.shape, dtype=np.float32) * 1e-3

    # aa log-probs
    aa_logits = rng.standard_normal((N, 20), dtype=np.float32)

    # distogram supervision (enable branch): neighbours + logits per bin=16
    # deterministic K nearest by dmap (excluding self)
    order = np.argsort(dmap, axis=-1)
    sup_neighbours = order[:, 1:1 + K].astype(np.int32)  # (N,K)
    sup_logits = rng.standard_normal((T, N, K, 16), dtype=np.float32)  # (T,N,K,16)

    # diffusion latent fields
    clean_latent = rng.standard_normal((N, latent_size), dtype=np.float32)
    predicted_latent = clean_latent + rng.standard_normal((N, latent_size), dtype=np.float32) * 1e-3
    time = rng.uniform(0.01, 1.0, size=(N,),).astype(np.float32)

    # VQ losses (scalars)
    codebook_losses = dict(
        codebook=np.float32(0.3),
        unassigned=np.float32(0.1),
        commitment=np.float32(0.2),
    )

    data_np = dict(
        residue_index=residue_index,
        chain_index=chain_index,
        batch_index=batch_index,
        mask=mask,
        aa_gt=aa_gt,
        pos_gt=pos_gt,
        dmap=dmap,
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        clean_latent=clean_latent,
        time=time,
    )
    result_np = dict(
        aa_logits=aa_logits,
        trajectory=trajectory,
        sup_neighbours=sup_neighbours,
        sup_logits=sup_logits,
        atom_pos=pred_atom_pos,
        predicted_latent=predicted_latent,
        codebook_losses=codebook_losses,
    )
    return data_np, result_np


@pytest.mark.parametrize("N,T,K", [(64, 3, 16)])
def test_decoder_loss_full_parity_max(monkeypatch, torch_device, jax_keys, atol, rtol, N, T, K):
    cfg = make_max_cfg()
    latent_size = int(getattr(cfg, "latent_size", 20))

    # patch randomness / heavy pieces
    import salad.modules.structure_autoencoder as salad_dec
    import caesar.modules.decoder as caesar_dec

    monkeypatch.setattr(salad_dec, "get_random_neighbours", lambda KK: det_random_neighbours_jax(KK))
    monkeypatch.setattr(caesar_dec, "get_random_neighbours", lambda KK: det_random_neighbours_torch(KK))

    monkeypatch.setattr(salad_dec, "get_spatial_neighbours", lambda count: det_spatial_neighbours_jax(count))
    monkeypatch.setattr(caesar_dec, "get_spatial_neighbours", lambda count: det_spatial_neighbours_torch(count))

    monkeypatch.setattr(salad_dec, "violation_loss", fake_violation_loss_jax)
    monkeypatch.setattr(caesar_dec, "violation_loss", fake_violation_loss_torch)

    # inputs
    data_np, result_np = make_full_inputs(N=N, T=T, K=K, latent_size=latent_size)

    # JAX log-probs
    aa_logp_jax = jax.nn.log_softmax(jnp.asarray(result_np["aa_logits"]), axis=-1)
    sup_logp_jax = jax.nn.log_softmax(jnp.asarray(result_np["sup_logits"]), axis=-1)

    data_jax = {k: to_jax(v) for k, v in data_np.items()}
    result_jax = {
        "aa": aa_logp_jax,
        "trajectory": to_jax(result_np["trajectory"]),
        "sup_neighbours": to_jax(result_np["sup_neighbours"]),
        "sup_distogram": sup_logp_jax,
        "atom_pos": to_jax(result_np["atom_pos"]),
        "predicted_latent": to_jax(result_np["predicted_latent"]),
        "codebook_losses": {k: to_jax(v) for k, v in result_np["codebook_losses"].items()},
    }

    # Torch log-probs
    aa_logp_t = torch.log_softmax(torch.as_tensor(result_np["aa_logits"], device=torch_device), dim=-1)
    sup_logp_t = torch.log_softmax(torch.as_tensor(result_np["sup_logits"], device=torch_device), dim=-1)

    data_t = {k: to_torch(v, device=torch_device) for k, v in data_np.items()}
    result_t = {
        "aa": aa_logp_t,
        "trajectory": to_torch(result_np["trajectory"], device=torch_device),
        "sup_neighbours": to_torch(result_np["sup_neighbours"], device=torch_device),
        "sup_distogram": sup_logp_t,
        "atom_pos": to_torch(result_np["atom_pos"], device=torch_device),
        "predicted_latent": to_torch(result_np["predicted_latent"], device=torch_device),
        "codebook_losses": {k: to_torch(v, device=torch_device) for k, v in result_np["codebook_losses"].items()},
    }

    # run JAX Decoder.loss (no params needed, but instantiate module in hk.transform)
    from salad.modules.structure_autoencoder import Decoder as JaxDecoder

    def jax_apply(data, result):
        dec = JaxDecoder(cfg)
        total, losses = dec.loss(data, result)
        return total, losses

    key_init, _ = jax_keys
    f = hk.without_apply_rng(hk.transform(jax_apply))
    params = f.init(key_init, data_jax, result_jax)
    total_jax, losses_jax = f.apply(params, data_jax, result_jax)

    
    # run Torch Decoder.loss
    from caesar.modules.decoder import Decoder as TorchDecoder
    dec_t = TorchDecoder(cfg).to(torch_device)
    dec_t.eval()
    with torch.no_grad():
        total_t, losses_t = dec_t.loss(data_t, result_t)
    print("\n LOSS DIFFS ")
    print("total_t =", float(total_t.detach().cpu()))
    print("total_j =", float(np.asarray(total_jax)))

    keys = sorted(set(losses_t.keys()) | set(losses_jax.keys()))
    for k in keys:
        lt = losses_t.get(k, None)
        lj = losses_jax.get(k, None)
        if lt is None or lj is None:
            print(f"{k:24s} missing: torch={lt is None}, jax={lj is None}")
            continue
        lt_v = float(lt.detach().cpu()) if isinstance(lt, torch.Tensor) else float(np.asarray(lt))
        lj_v = float(np.asarray(lj))
        print(f"{k:24s} torch={lt_v: .6f}  jax={lj_v: .6f}  diff={lt_v - lj_v: .6f}")
    # compare
    atol2, rtol2 = max(atol, 1e-4), max(rtol, 1e-4)

    assert_allclose("total", total_t, total_jax, atol=atol2, rtol=rtol2)

    expected = [
        "aa",
        "fape", "fape_trajectory",
        "distogram", "distogram_trajectory",
        "kabsch_rmsd", "kabsch_rmsd_trajectory",
        "local",
        "latent",
        "violation",
        "codebook", "unassigned", "commitment",
    ]
    for k in expected:
        assert k in losses_t, f"torch losses missing key: {k}"
        assert k in losses_jax, f"jax losses missing key: {k}"
        assert_allclose(f"losses[{k}]", losses_t[k], losses_jax[k], atol=atol2, rtol=rtol2)

def _tree_map(fn, x):
    
    if isinstance(x, dict):
        return {k: _tree_map(fn, v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_tree_map(fn, v) for v in x)
    if x is None:
        return None
    return fn(x)

def to_jax(x):
    """Recursively convert numpy/torch leaves to jnp arrays (keeps dict/list structure)."""
    def _leaf(v):
        if isinstance(v, jnp.ndarray):
            return v
        if isinstance(v, torch.Tensor):
            return jnp.asarray(v.detach().cpu().numpy())
        if isinstance(v, np.ndarray):
           return jnp.asarray(v)
        if np.isscalar(v):
            return jnp.asarray(v)
        return v
    return _tree_map(_leaf, x)
 
def to_torch(x, device=None, dtype=None):
    """Recursively convert numpy/jax leaves to torch tensors (keeps dict/list structure)."""
    def _leaf(v):
        if isinstance(v, torch.Tensor):
            t = v
        elif isinstance(v, jnp.ndarray):
            t = torch.from_numpy(np.asarray(v))
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        elif np.isscalar(v):
            t = torch.tensor(v)
        else:
            return v
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t
    return _tree_map(_leaf, x)

def _np(x):
    return np.asarray(x)

def assert_allclose_np(name, a, b, atol, rtol):
    a = _np(a)
    b = _np(b)
    assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.max(np.abs(a - b))
        raise AssertionError(f"{name}: not close (max|diff|={diff}, atol={atol}, rtol={rtol})")

def grad_parity_one_key(
    *,
    key: str,
    cfg,
    data_jax: dict,
    result_jax_full: dict,
    data_t: dict,
    result_t_full: dict,
    torch_device,
    jax_keys,
    JaxDecoder,
    TorchDecoder,
    atol=1e-4,
    rtol=1e-4,
):
    import numpy as np
    import torch
    import jax
    import haiku as hk

    def np_allclose(name, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        assert a.shape == b.shape, f"{name}: shape mismatch {a.shape} vs {b.shape}"
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            diff = np.max(np.abs(a - b))
            raise AssertionError(f"{name}: not close (max|diff|={diff}, atol={atol}, rtol={rtol})")

    # JAX: grad wrt result[key]
    def jax_total(x):
        r = dict(result_jax_full)
        r[key] = x
        dec = JaxDecoder(cfg)
        total, _ = dec.loss(data_jax, r)
        return total

    key_init, key_apply = jax_keys
    f = hk.transform(lambda x: jax_total(x))
    params = f.init(key_init, result_jax_full[key])
    total_j = f.apply(params, key_apply, result_jax_full[key])
    grad_j = jax.grad(lambda x: f.apply(params, key_apply, x))(result_jax_full[key])

    # Torch: grad wrt result[key]
    r_t = dict(result_t_full)
    x0 = r_t[key]
    assert isinstance(x0, torch.Tensor), f"result_t[{key}] must be torch.Tensor, got {type(x0)}"
    x = x0.clone().detach().requires_grad_(True)
    r_t[key] = x

    dec_t = TorchDecoder(cfg).to(torch_device)
    dec_t.train()

    total_t, _ = dec_t.loss(data_t, r_t)
    total_t.backward()
    grad_t = x.grad
    if grad_t is None:
        grad_t = torch.zeros_like(x)

    # Compare
    np_allclose("total", float(total_t.detach().cpu()), float(np.asarray(total_j)))
    np_allclose(f"d(total)/d({key})", grad_t.detach().cpu().numpy(), np.asarray(grad_j))


@pytest.mark.parametrize("N,T,K", [(64, 3, 16)])
def test_grad_parity_many_keys(monkeypatch, torch_device, jax_keys, atol, rtol, N, T, K):
    cfg = make_max_cfg()
    latent_size = int(getattr(cfg, "latent_size", 20))

    # patch randomness / heavy pieces
    import salad.modules.structure_autoencoder as salad_dec
    import caesar.modules.decoder as caesar_dec

    monkeypatch.setattr(salad_dec, "get_random_neighbours", lambda KK: det_random_neighbours_jax(KK))
    monkeypatch.setattr(caesar_dec, "get_random_neighbours", lambda KK: det_random_neighbours_torch(KK))

    monkeypatch.setattr(salad_dec, "get_spatial_neighbours", lambda count: det_spatial_neighbours_jax(count))
    monkeypatch.setattr(caesar_dec, "get_spatial_neighbours", lambda count: det_spatial_neighbours_torch(count))

    monkeypatch.setattr(salad_dec, "violation_loss", fake_violation_loss_jax)
    monkeypatch.setattr(caesar_dec, "violation_loss", fake_violation_loss_torch)

    # inputs
    data_np, result_np = make_full_inputs(N=N, T=T, K=K, latent_size=latent_size)

    # JAX log-probs
    aa_logp_jax = jax.nn.log_softmax(jnp.asarray(result_np["aa_logits"]), axis=-1)
    sup_logp_jax = jax.nn.log_softmax(jnp.asarray(result_np["sup_logits"]), axis=-1)

    data_jax = {k: to_jax(v) for k, v in data_np.items()}
    result_jax = {
        "aa": aa_logp_jax,
        "trajectory": to_jax(result_np["trajectory"]),
        "sup_neighbours": to_jax(result_np["sup_neighbours"]),
        "sup_distogram": sup_logp_jax,
        "atom_pos": to_jax(result_np["atom_pos"]),
        "predicted_latent": to_jax(result_np["predicted_latent"]),
        "codebook_losses": {k: to_jax(v) for k, v in result_np["codebook_losses"].items()},
    }

    # Torch log-probs
    aa_logp_t = torch.log_softmax(torch.as_tensor(result_np["aa_logits"], device=torch_device), dim=-1)
    sup_logp_t = torch.log_softmax(torch.as_tensor(result_np["sup_logits"], device=torch_device), dim=-1)

    data_t = {k: to_torch(v, device=torch_device) for k, v in data_np.items()}
    result_t = {
        "aa": aa_logp_t,
        "trajectory": to_torch(result_np["trajectory"], device=torch_device),
        "sup_neighbours": to_torch(result_np["sup_neighbours"], device=torch_device),
        "sup_distogram": sup_logp_t,
        "atom_pos": to_torch(result_np["atom_pos"], device=torch_device),
        "predicted_latent": to_torch(result_np["predicted_latent"], device=torch_device),
        "codebook_losses": {k: to_torch(v, device=torch_device) for k, v in result_np["codebook_losses"].items()},
    }
    result_np = dict(result_np)
    result_np["aa"] = jax.device_get(np.log(np.exp(result_np["aa_logits"]) / np.exp(result_np["aa_logits"]).sum(-1, keepdims=True)))  # если у тебя logits
    
    from salad.modules.structure_autoencoder import Decoder as JaxDecoder
    from caesar.modules.decoder import Decoder as TorchDecoder

    keys_to_check = ["trajectory", "atom_pos", "aa", "sup_distogram", "predicted_latent"]

    atol2, rtol2 = max(atol, 1e-4), max(rtol, 1e-4)

    for k in keys_to_check:
        grad_parity_one_key(
            key=k,
            cfg=cfg,
            data_jax=data_jax,
            result_jax_full=result_jax,
            data_t=data_t,
            result_t_full=result_t,
            torch_device=torch_device,
            jax_keys=jax_keys,
            JaxDecoder=JaxDecoder,
            TorchDecoder=TorchDecoder,
            atol=atol2,
            rtol=rtol2,
        )