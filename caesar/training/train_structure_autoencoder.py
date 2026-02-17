import os
import time
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
from pathlib import Path
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.tensorboard import SummaryWriter
import numpy as np
LOG_INTERVAL = 10
DEBUG = False

def cast_float(batch: Dict[str, Any], dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
    """Like flexloop.loop.cast_float: cast only floating tensors."""
    out = {}
    for k, v in batch.items():
        t = v if torch.is_tensor(v) else torch.as_tensor(v)
        if t.is_floating_point() and t.dtype != dtype:
            t = t.to(dtype=dtype)
        out[k] = t
    return out


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v if v.device == device else v.to(device=device, non_blocking=True)
        else:
            out[k] = torch.as_tensor(v, device=device)
    return out


def prepare_batch(
    batch: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    float_dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    """
    Convert tree leaves to tensors, cast float tensors to `float_dtype`,
    and place tensors on `device` (if provided) in a single pass.
    """
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        t = v if torch.is_tensor(v) else torch.as_tensor(v, device=device)
        if t.is_floating_point() and t.dtype != float_dtype:
            t = t.to(dtype=float_dtype)
        if device is not None and t.device != device:
            t = t.to(device=device, non_blocking=True)
        out[k] = t
    return out


def slice_batch_first_dim(batch: Dict[str, Any], n: int) -> Dict[str, Any]:
    """mirrors jax: tree_map(lambda x: x[:n], item_0) for array-like tensors."""
    out = {}
    for k, v in batch.items():
        t = v if torch.is_tensor(v) else torch.as_tensor(v)
        if t.ndim >= 1 and t.shape[0] >= n:
            out[k] = t[:n]
        else:
            out[k] = t
    return out

def split_generator(gen: torch.Generator, device: Optional[torch.device] = None) -> Tuple[torch.Generator, torch.Generator]:
    device = device if device is not None else getattr(gen, "device", torch.device("cpu"))
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen, device=device).item())
    sub = torch.Generator(device=device).manual_seed(seed)
    return gen, sub


def _call_model(model, data: Dict[str, Any], generator: Optional[torch.Generator]):
    """Call model with generator kwarg if supported."""
    try:
        return model(data, generator=generator)
    except TypeError:
        return model(data)

def rebatch_call(
    fn: Callable[[Dict[str, Any], Optional[torch.Generator]], Tuple[torch.Tensor, Dict[str, Any]]],
    rebatch: int = 1,
) -> Callable[[Dict[str, Any], Optional[torch.Generator]], Tuple[torch.Tensor, Dict[str, Any]]]:
    """
    Split a big batch along dim0 into `rebatch` chunks and average loss + out["losses"].
    Mirrors flexloop.rebatch_call usage pattern in salad.
    """
    rebatch = int(rebatch)

    def _call(data: Dict[str, Any], generator: Optional[torch.Generator] = None):
        if rebatch <= 1:
            return fn(data, generator)

        ref = None
        for v in data.values():
            if torch.is_tensor(v) and v.ndim >= 1:
                ref = v
                break
        if ref is None:
            return fn(data, generator)

        n = int(ref.shape[0])
        assert n % rebatch == 0, f"leading dim {n} must be divisible by rebatch={rebatch}"
        m = n // rebatch

        loss_sum = None
        losses_sum: Dict[str, torch.Tensor] = {}
        out_last: Dict[str, Any] = {}

        for i in range(rebatch):
            if generator is not None:
                gen_device = getattr(ref, "device", None)
                generator, micro_gen = split_generator(generator, device=gen_device)
            else:
                micro_gen = None
            sl = slice(i * m, (i + 1) * m)
            micro = {}
            for k, v in data.items():
                if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n:
                    micro[k] = v[sl]
                else:
                    micro[k] = v

            loss_i, out_i = fn(micro, micro_gen)
            out_last = out_i

            loss_sum = loss_i if loss_sum is None else (loss_sum + loss_i)

            if isinstance(out_i, dict) and isinstance(out_i.get("losses", None), dict):
                for name, item in out_i["losses"].items():
                    losses_sum[name] = losses_sum.get(name, 0.0) + item

        loss = loss_sum / float(rebatch)
        out = dict(out_last)
        if losses_sum:
            out["losses"] = {k: v / float(rebatch) for k, v in losses_sum.items()}
        return loss, out

    return _call

# class EMA:
#     def __init__(self, model: torch.nn.Module, decay: float = 0.999):
#         self.decay = float(decay)
#         self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

#     @torch.no_grad()
#     def update(self, model: torch.nn.Module):
#         sd = model.state_dict()
#         for k, v in sd.items():
#             self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

#     @torch.no_grad()
#     def apply_to(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
#         backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
#         model.load_state_dict(self.shadow, strict=True)
#         return backup

#     @torch.no_grad()
#     def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]):
#         model.load_state_dict(backup, strict=True)

@dataclass
class State:
    key: torch.Generator
    step_id: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    aux_state: Dict[str, Any]  


def save_loop_state(path: str, loop_state: State):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "step_id": loop_state.step_id,
        "model": loop_state.model.state_dict(),
        "optimizer": loop_state.optimizer.state_dict(),
        "aux_state": loop_state.aux_state,
        "rng_state": loop_state.key.get_state(),
    }
    torch.save(ckpt, os.path.join(path, "checkpoint.pt"))


def load_loop_state(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> Optional[State]:
    ckpt_path = os.path.join(path, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    model.to(device)

    gen = torch.Generator(device=device)
    if "rng_state" in ckpt:
        try:
            gen.set_state(ckpt["rng_state"])
        except Exception as e:
            print(f"Warning: failed to restore RNG state ({e}); using fresh seed.")
            gen = torch.Generator(device=device).manual_seed(42)

    return State(
        key=gen,
        step_id=int(ckpt.get("step_id", 0)),
        model=model,
        optimizer=optimizer,
        aux_state=dict(ckpt.get("aux_state", {})),
    )

def _repeat0(t: torch.Tensor, times: int) -> torch.Tensor:
    return t.repeat((times,) + (1,) * (t.ndim - 1))

def make_smoke_batch(npz_path: str, device: torch.device, mul: int) -> Dict[str, torch.Tensor]:
    d = dict(np.load(npz_path))
    batch = prepare_batch(d, device=device, float_dtype=torch.float32)

    # infer L (residue axis length)
    L = None
    for key in ("pos", "residue_index", "aatype"):
        if key in batch and batch[key].ndim >= 1:
            L = int(batch[key].shape[0])
            break
    if L is None:
        raise ValueError("Smoke batch: cannot infer L (need pos/residue_index/aatype in npz).")

    mul = int(mul)
    if mul > 1:
        for k, t in list(batch.items()):
            if torch.is_tensor(t) and t.ndim >= 1 and int(t.shape[0]) == L:
                batch[k] = _repeat0(t, mul)

    return batch

def infinite_stream(batch: Dict[str, torch.Tensor]):
    while True:
        yield {k: v.clone() for k, v in batch.items()}

def clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, np.ndarray):
            out[k] = v.copy()
        else:
            out[k] = v
    return out

def repeat_batch(batch: Dict[str, Any]):
    while True:
        yield clone_batch(batch)

def repeat_batch_nocopy(batch: Dict[str, Any]):
    while True:
        yield batch

def accumulate_stream(stream: Iterator[Dict[str, Any]], count: int):
    count = int(count)
    assert count >= 1
    while True:
        items = [next(stream) for _ in range(count)]
        yield _merge_batches(items)

def _merge_batches(items: list[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    offset = 0
    for item in items:
        if "batch_index" in item:
            bi = item["batch_index"]
            if torch.is_tensor(bi):
                bi = bi.detach().cpu().numpy()
            if isinstance(bi, np.ndarray) and bi.size:
                item = dict(item)
                item["batch_index"] = bi + offset
                offset += int(bi.max()) + 1
    keys = items[0].keys()
    for k in keys:
        vals = [it[k] for it in items]
        v0 = vals[0]
        if torch.is_tensor(v0):
            out[k] = torch.cat(vals, dim=0)
        elif isinstance(v0, np.ndarray):
            out[k] = np.concatenate(vals, axis=0)
        else:
            out[k] = v0
    return out

def take_first_protein(batch: Dict[str, Any], *, target_size: Optional[int] = None) -> Dict[str, Any]:
    if "batch_index" not in batch:
        return batch
    batch_index = batch["batch_index"]
    if torch.is_tensor(batch_index):
        batch_index = batch_index.detach().cpu().numpy()
    size = int(target_size or batch_index.shape[0])
    keep = batch_index == batch_index.min()
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray) and v.shape[0] == size:
            out[k] = v[keep]
        else:
            out[k] = v
    out = pad_dict(out, size)
    out["batch_index"] = np.zeros_like(out["batch_index"])
    out["seq_mask"] = out["mask"] * (out["aa_gt"] != 20)
    out["residue_mask"] = out["mask"] * out["all_atom_mask"].any(axis=-1)
    return out

def _tensor_stats(t: torch.Tensor) -> str:
    t = t.detach()
    if t.is_floating_point():
        return f"sum={t.sum().item():.4g} mean={t.mean().item():.4g} std={t.std().item():.4g}"
    return f"sum={t.sum().item()} mean={t.float().mean().item():.4g}"

def summarize_batch(batch: Dict[str, Any], keys: Optional[Tuple[str, ...]] = None) -> str:
    parts = []
    if keys is None:
        keys = ("aa_gt", "residue_index", "chain_index", "batch_index", "mask", "seq_mask", "residue_mask", "all_atom_positions")
    for k in keys:
        if k not in batch:
            continue
        v = batch[k]
        t = v if torch.is_tensor(v) else torch.as_tensor(v)
        parts.append(f"{k}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} {_tensor_stats(t)}")
    return " | ".join(parts)

def dump_batch(batch: Dict[str, Any], keys: Optional[Tuple[str, ...]] = None) -> str:
    lines = []
    if keys is None:
        keys = ("aa_gt", "residue_index", "chain_index", "batch_index", "mask", "seq_mask", "residue_mask")
    for k in keys:
        if k not in batch:
            continue
        v = batch[k]
        t = v if torch.is_tensor(v) else torch.as_tensor(v)
        lines.append(f"{k}:\n{t}")
    return "\n".join(lines)
        
def cosine_decay_schedule(start_lr, decay_lr, warmup_steps, decay_steps):
    start_lr = float(start_lr)
    decay_lr = float(decay_lr)
    warmup_steps = int(warmup_steps)
    decay_steps = int(decay_steps)

    def schedule(count: int) -> float:
        if count <= warmup_steps:
            result = (count / max(warmup_steps, 1)) * start_lr
        else:
            t = (count - warmup_steps) % max(decay_steps, 1)
            result = (start_lr - decay_lr) * 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi) * (t / decay_steps))).item() + decay_lr
        return max(result, 0.0)

    return schedule

def model_step(model, config, rebatch: int = 1, is_training: bool = True, device: Optional[torch.device] = None):
    if not is_training:
        config = deepcopy(config)
        config.eval = True
        model.eval()
    else:
        model.train()

    def core_step(data: Dict[str, Any], generator: Optional[torch.Generator] = None):
        # Hot path: only touch tensors if batch is not already prepared.
        needs_prepare = False
        for v in data.values():
            if not torch.is_tensor(v):
                needs_prepare = True
                break
            if v.is_floating_point() and v.dtype != torch.float32:
                needs_prepare = True
                break
            if device is not None and v.device != device:
                needs_prepare = True
                break
        if needs_prepare:
            data = prepare_batch(data, device=device, float_dtype=torch.float32)

        loss, out = _call_model(model, data, generator)
        return loss, out

    step = rebatch_call(core_step, rebatch=rebatch)

    def final_step(data: Dict[str, Any], generator: Optional[torch.Generator] = None):
        loss, out = step(data, generator)
        losses = out.get("losses", {})
        res_dict = {f"{name}_loss": item.to(torch.float32) for name, item in losses.items()}
        return loss.to(torch.float32), res_dict

    return final_step

def make_training_inner(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[Dict[str, Any], Optional[torch.Generator]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    data: Iterator[Dict[str, Any]],
    *,
    schedule: Callable[[int], float],
    clip: float,
    accumulate: int = 1,
    ema_weight: None,#float = 0.999,
    debug_batch: bool = False,
    debug_batch_full: bool = False,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
):
    accumulate = int(accumulate)
    clip = float(clip)
    if DEBUG:
        bad = []
        for k, v in model.state_dict().items():
            if isinstance(v, UninitializedParameter):
                bad.append(k)

        print("Uninitialized params:", bad)
        assert len(bad) == 0, "Found uninitialized params; run a forward that touches them before EMA."
    # ema_weight = float(ema_weight)
    # ema = EMA(model, decay=ema_weight) if ema_weight is not None else None
    ema=None
    first_fp = {"value": None}

    use_amp = bool(amp) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    autocast = torch.autocast if hasattr(torch, "autocast") else torch.cuda.amp.autocast

    cache = {"raw": None, "prepared": None, "chunks": None}

    def training_inner(loop_state: State):
        t = time.time()
        item = next(data)
        load_time = time.time() - t

        key, subkey = split_generator(loop_state.key)  

        lr = float(schedule(loop_state.step_id))
        for pg in loop_state.optimizer.param_groups:
            pg["lr"] = lr

        model.train(True)
        optimizer.zero_grad(set_to_none=True)

        tr = time.time()

        if cache["raw"] is item:
            item_t = cache["prepared"]
            chunks = cache["chunks"]
        else:
            item_t = prepare_batch(item, device=device, float_dtype=torch.float32)
            if debug_batch:
                fp = summarize_batch(item_t)
                same = (fp == first_fp["value"]) if first_fp["value"] is not None else True
                if first_fp["value"] is None:
                    first_fp["value"] = fp
                print(f"[batch] same_as_first={same} :: {fp}")
                if debug_batch_full:
                    print("[batch-full]\n" + dump_batch(item_t))

            ref = None
            for v in item_t.values():
                if torch.is_tensor(v) and v.ndim >= 1:
                    ref = v
                    break

            chunks = [item_t]
            if ref is not None and accumulate > 1:
                n = int(ref.shape[0])
                assert n % accumulate == 0, f"leading dim {n} must be divisible by accumulate={accumulate}"
                m = n // accumulate
                chunks = []
                for i in range(accumulate):
                    sl = slice(i * m, (i + 1) * m)
                    ch = {}
                    for k, v in item_t.items():
                        if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n:
                            ch[k] = v[sl]
                        else:
                            ch[k] = v
                    chunks.append(ch)

            cache["raw"] = item
            cache["prepared"] = item_t
            cache["chunks"] = chunks

        log_sum: Dict[str, torch.Tensor] = {}
        loss_sum = 0.0
        for ch in chunks:
            subkey, ch_key = split_generator(subkey) # (c) gen for each chunk
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss, logs = step_fn(ch, ch_key)
            scaled_loss = loss / float(len(chunks))
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            loss_sum = loss_sum + loss.detach()
            for k, v in logs.items():
                log_sum[k] = log_sum.get(k, 0.0) + v.detach()

        if clip is not None and clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        # if ema is not None:
        #     ema.update(model)

        step_time = time.time() - tr
        total_time = time.time() - t

        denom = float(len(chunks))
        loggables = {k: (v / denom) for k, v in log_sum.items()}
        loggables["loss"] = (loss_sum / denom)
        loggables["lr"] = torch.tensor(lr, dtype=torch.float32)
        new_state = State(
            key=key,
            step_id=loop_state.step_id + 1,
            model=model,
            optimizer=optimizer,
            aux_state={"ema": ema.shadow if ema is not None else None, "ema_decay": ema.decay if ema is not None else None},
        )

        current_loss = float(loggables["loss"].detach().cpu().item())
        print(
            f"Step {loop_state.step_id}, loss: {current_loss:.6f}, "
            f"load time {load_time:.3f}s, step time {step_time:.3f}s, total {total_time:.3f}s"
        )
        return new_state, loggables

    return training_inner


def make_valid_inner(
    model: torch.nn.Module,
    step_fn: Callable[[Dict[str, Any], Optional[torch.Generator]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    data: Iterator[Dict[str, Any]],
    *,
    use_ema: bool = True,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    use_amp = bool(amp) and torch.cuda.is_available()
    autocast = torch.autocast if hasattr(torch, "autocast") else torch.cuda.amp.autocast

    def valid_inner(loop_state: State):
        t = time.time()
        item = next(data)

        device = next(model.parameters()).device
        item_t = prepare_batch(item, device=device, float_dtype=torch.float32)

        backup = None
        # ema_shadow = loop_state.aux_state.get("ema", None)
        # if use_ema and (ema_shadow is not None):
        #     backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        #     model.load_state_dict(ema_shadow, strict=True)

        model.eval()
        with torch.no_grad():
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss, logs = step_fn(item_t, loop_state.key)

        if backup is not None:
            model.load_state_dict(backup, strict=True)

        res = dict(logs)
        res["loss"] = loss.detach()

        print(f"Computed valid batch in {time.time() - t:.3f} seconds. w// loss = {res["loss"]}")
        return res

    return valid_inner

def training(
    path: str,
    training_inner: Callable[[State], Tuple[State, Dict[str, torch.Tensor]]],
    *,
    valid_inner: Optional[Callable[[State], Dict[str, torch.Tensor]]] = None,
    max_steps: int = 1000,
    valid_interval: int = 100,
    writer: Optional[SummaryWriter] = None,
):
    def _loop(loop_state: State):
        while loop_state.step_id < max_steps:
            loop_state, loggables = training_inner(loop_state)

            if writer is not None:
                for k, v in loggables.items():
                    writer.add_scalar(f"train/{k}", float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v), loop_state.step_id)

            if valid_inner is not None and (loop_state.step_id % valid_interval == 0):
                res = valid_inner(loop_state)
                if writer is not None:
                    for k, v in res.items():
                        writer.add_scalar(f"valid/{k}", float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v), loop_state.step_id)
                save_loop_state(path, loop_state)

        save_loop_state(path, loop_state)
        return loop_state

    return _loop

if __name__ == "__main__":
    from caesar.data.allpdb import BatchedProteinPDBStream, pad_dict
    from flexloop.data import BatchStream
    from flexloop.utils import parse_options
    from caesar.modules.config import distance_to_structure_decoder as config_choices

    from caesar.modules.autoencoder import StructureAutoencoder, StructureDecoder

    opt = parse_options(
        "train a distance-to-structure decoder on PDB.",
        path="network/",
        config="small_inner",
        data_path="",
        num_aa=1024,
        p_complex=0.5,
        lr=1e-3,
        decay_lr=1e-5,
        warmup_steps=1_000,
        decay_steps=500_000,
        clip=0.1,
        b1=0.9,
        b2=0.99,
        ema_weight=0.999,
        rebatch=1,
        accumulate=1,
        jax_seed=42,   
        multigpu="False",
        suffix="1",
        overfit_one="False",
        no_random="False",
        fixed_recycle=-1,
        deterministic="False",
        debug_batch="False",
        debug_batch_full="False",
        data_seed=42,
        num_workers=1,
        prefetch_factor=2,
        smoke="False",
        smoke_steps=5,
        smoke_valid_interval=1,
        amp="True",
        amp_dtype="fp16",
    )
    
    smoke = (opt.smoke == "True")
    
    multigpu = (opt.multigpu == "True")
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
    seed = int(opt.jax_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if opt.deterministic == "True":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    NUM_DEVICES = torch.cuda.device_count() if multigpu and torch.cuda.is_available() else 1

    config = getattr(config_choices, opt.config)
    sae = "sdd" if getattr(config, "is_decoder", False) else "sae"
    path = f"{opt.path}/torch/{sae}-{opt.config}-{opt.num_aa}-{opt.suffix}"
    writer = SummaryWriter(path)
    mul = opt.rebatch * opt.accumulate * NUM_DEVICES
    print(f"mul={mul}, multigpu={multigpu}, device={device}")
    if smoke:
        print("SMOKE MODE: using tests/data/test_structure.npz instead of full dataset")
        config.is_decoder = False
        config.eval = True

        npz_path = "tests/data/test_structure.npz"
        batch = make_smoke_batch(npz_path, device=device, mul=mul)
        data = iter(infinite_stream(batch))
        valid_data = iter(infinite_stream(batch))

        total_steps = int(opt.smoke_steps)
        valid_interval = int(opt.smoke_valid_interval)
    else:
        print("Attempting to load dataset...")
        seed_train = int(opt.data_seed)
        seed_valid = seed_train + 1
        data = BatchedProteinPDBStream(
            f"{opt.data_path}/allpdb/",
            seqres_aa="clusterSeqresAA",
            cutoff_resolution=4.0,
            p_complex=opt.p_complex,
            size=512,#1024,
            min_size=16,
            max_size=512,#1024,
            seed=seed_train,
        )
        num_workers = int(opt.num_workers)
        prefetch = int(opt.prefetch_factor) if num_workers > 0 else None
        accumulate = int(opt.rebatch) * int(opt.accumulate) * int(NUM_DEVICES)
        if num_workers == 0:
            data = iter(data)
            if accumulate != 1:
                data = accumulate_stream(data, accumulate)
        else:
            data = iter(
                BatchStream(
                    data,
                    num_workers=num_workers, # 32
                    accumulate=accumulate,
                    prefetch_factor=prefetch, # 32
                )
            )

        valid_data = BatchedProteinPDBStream(
            f"{opt.data_path}/allpdb/",
            seqres_aa="clusterSeqresAA",
            cutoff_resolution=4.0,
            p_complex=opt.p_complex,
            size=512,#1024,
            min_size=16,
            max_size=512,#1024,
            start_date="01/01/22",
            cutoff_date="12/31/23",
            seed=seed_valid,
        )
        if num_workers == 0:
            valid_data = iter(valid_data)
            if accumulate != 1:
                valid_data = accumulate_stream(valid_data, accumulate)
        else:
            valid_data = iter(
                BatchStream(
                    valid_data,
                    num_workers=num_workers, # 8
                    accumulate=accumulate,
                    prefetch_factor=prefetch, # 8
                )
            )
        print("Dataset successfully loaded.")

        if opt.overfit_one == "True":
            print("Overfitting on a single batch (reusing the first batch for train/valid).")
            first_item = next(data)
            first_item = take_first_protein(first_item)
            data = repeat_batch_nocopy(first_item)
            valid_data = repeat_batch_nocopy(first_item)
            item_0 = first_item
        else:
            item_0 = next(data)

    def flag_stream(stream, *, no_random: bool, fixed_recycle: int):
        if (not no_random) and fixed_recycle < 0:
            return stream
        def _gen():
            for item in stream:
                item = dict(item)
                if no_random:
                    item["no_random"] = True
                if fixed_recycle >= 0:
                    item["fixed_recycle"] = fixed_recycle
                yield item
        return _gen()

    data = flag_stream(data, no_random=(opt.no_random == "True"), fixed_recycle=int(opt.fixed_recycle))
    valid_data = flag_stream(valid_data, no_random=(opt.no_random == "True"), fixed_recycle=int(opt.fixed_recycle))

    if getattr(config, "is_decoder", False):
        model = StructureDecoder(config).to(device)
    else:
        model = StructureAutoencoder(config).to(device)

    # (с) "init params" analogue for LazyLinear: run one small forward
    print("Initializing model parameters...")
    if smoke:
        item_0 = next(data)
    print("INPUT OF SHAPE:")
    for name, value in item_0.items():
        v = value if torch.is_tensor(value) else torch.as_tensor(value)
        print("  ", name, tuple(v.shape))

    init_batch = slice_batch_first_dim(item_0, int(opt.rebatch) * 100)
    init_batch = prepare_batch(init_batch, device=device, float_dtype=torch.float32)

    # (с) one dry forward to materialize LazyLinear
    gen0 = torch.Generator(device=device).manual_seed(int(opt.jax_seed))
    model.train(True)
    with torch.no_grad():
        _ = _call_model(model, init_batch, gen0)

    print("Model parameters initialized.")

    print("Writing model description...")
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/model_description", "w") as f:
        f.write(str(model))
    print("Model description written.")

    schedule = cosine_decay_schedule(
        start_lr=opt.lr,
        decay_lr=opt.decay_lr,
        warmup_steps=opt.warmup_steps,
        decay_steps=opt.decay_steps,
    )
    total_steps = int(opt.warmup_steps + opt.decay_steps + 1)

    print("Initializing optimizer state...")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(opt.lr),
        betas=(float(opt.b1), float(opt.b2)),
        eps=1e-9,
    )
    print("Optimizer initialized.")

    # Batches are prepared on the target device before calling step_fn.
    step_train = model_step(model, config, rebatch=int(opt.rebatch), is_training=True, device=None)
    step_valid = model_step(model, config, rebatch=int(opt.rebatch), is_training=False, device=None)

    print("Constructing training loop...")
    key = torch.Generator(device=device).manual_seed(int(opt.jax_seed))
    aux_state: Dict[str, Any] = {}
    loop_state = State(key=key, step_id=0, model=model, optimizer=optimizer, aux_state=aux_state)

    print("Recovering previous state, if available...")
    loaded = load_loop_state(path, model, optimizer, device)
    if loaded is not None:
        loop_state = loaded

    amp_enabled = (opt.amp == "True")
    amp_dtype = torch.bfloat16 if str(opt.amp_dtype).lower() in ("bf16", "bfloat16") else torch.float16
    print(f"amp_dtype={amp_dtype }")
    training_inner = make_training_inner(
        model=model,
        optimizer=optimizer,
        step_fn=step_train,
        data=data,
        schedule=schedule,
        clip=float(opt.clip),
        accumulate=int(opt.accumulate),
        ema_weight=float(opt.ema_weight),
        debug_batch=(opt.debug_batch == "True"),
        debug_batch_full=(opt.debug_batch_full == "True"),
        amp=amp_enabled,
        amp_dtype=amp_dtype,
    )
    valid_inner = make_valid_inner(
        model=model,
        step_fn=step_valid,
        data=valid_data,
        use_ema=False, #True,
        amp=amp_enabled,
        amp_dtype=amp_dtype,
    )

    print("Starting training...")
    print(f"Log files and tensorboard records will be written to {path}")

    train_loop = training(
        path,
        training_inner,
        valid_inner=valid_inner,
        max_steps=total_steps,
        valid_interval=100,
        writer=writer,
    )
    train_loop(loop_state)


# python -m caesar.training.train_structure_autoencoder --data_path /disk/2tb/edelkin/data --overfit_one True --p_complex 0.0 --num_workers 1 --prefetch_factor 2

### full train det data
# python -m caesar.training.train_structure_autoencoder --data_path /disk/2tb/edelkin/data --data_seed 123 --num_workers 0 --prefetch_factor 2 --deterministic True
