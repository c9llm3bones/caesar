import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def cast_float(batch: Dict[str, Any], dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
    """Like flexloop.loop.cast_float: cast only floating tensors."""
    out = {}
    for k, v in batch.items():
        t = v if torch.is_tensor(v) else torch.as_tensor(v)
        if t.is_floating_point():
            t = t.to(dtype=dtype)
        out[k] = t
    return out


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
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


def split_generator(gen: torch.Generator) -> Tuple[torch.Generator, torch.Generator]:
    """Rough analogue of jax.random.split."""
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen).item())
    sub = torch.Generator(device="cpu").manual_seed(seed)
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
            sl = slice(i * m, (i + 1) * m)
            micro = {}
            for k, v in data.items():
                if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == n:
                    micro[k] = v[sl]
                else:
                    micro[k] = v

            loss_i, out_i = fn(micro, generator)
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

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]):
        model.load_state_dict(backup, strict=True)

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

    gen = torch.Generator(device="cpu")
    if "rng_state" in ckpt:
        gen.set_state(ckpt["rng_state"])

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
    batch = {k: torch.as_tensor(v, device=device) for k, v in d.items()}
    batch = cast_float(batch, dtype=torch.float32)

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
        data = {k: (v if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in data.items()}
        data = cast_float(data, dtype=torch.float32)
        if device is not None:
            data = move_to_device(data, device)

        loss, out = _call_model(model, data, generator)
        return loss, out

    step = rebatch_call(core_step, rebatch=rebatch)

    def final_step(data: Dict[str, Any], generator: Optional[torch.Generator] = None):
        loss, out = step(data, generator)
        res_dict = {f"{name}_loss": item for name, item in out["losses"].items()}
        res_dict = {k: v.to(torch.float32) for k, v in res_dict.items()}
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
    ema_weight: float = 0.999,
):
    accumulate = int(accumulate)
    clip = float(clip)
    ema_weight = float(ema_weight)

    ema = EMA(model, decay=ema_weight) if ema_weight is not None else None

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

        device = next(model.parameters()).device
        item_t = {k: (v if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in item.items()}
        item_t = cast_float(item_t, dtype=torch.float32)
        item_t = move_to_device(item_t, device)

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

        log_sum: Dict[str, torch.Tensor] = {}
        loss_sum = 0.0
        for ch in chunks:
            loss, logs = step_fn(ch, subkey)
            (loss / float(len(chunks))).backward()
            loss_sum = loss_sum + loss.detach()
            for k, v in logs.items():
                log_sum[k] = log_sum.get(k, 0.0) + v.detach()

        if clip is not None and clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        optimizer.step()
        if ema is not None:
            ema.update(model)

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

        print(
            f"Step {loop_state.step_id}, load time {load_time:.3f} s, step time {step_time:.3f} s, total {total_time:.3f} s"
        )
        return new_state, loggables

    return training_inner


def make_valid_inner(
    model: torch.nn.Module,
    step_fn: Callable[[Dict[str, Any], Optional[torch.Generator]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    data: Iterator[Dict[str, Any]],
    *,
    use_ema: bool = True,
):
    def valid_inner(loop_state: State):
        t = time.time()
        item = next(data)

        device = next(model.parameters()).device
        item_t = {k: (v if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in item.items()}
        item_t = cast_float(item_t, dtype=torch.float32)
        item_t = move_to_device(item_t, device)

        backup = None
        ema_shadow = loop_state.aux_state.get("ema", None)
        if use_ema and (ema_shadow is not None):
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema_shadow, strict=True)

        model.eval()
        with torch.no_grad():
            loss, logs = step_fn(item_t, loop_state.key)

        if backup is not None:
            model.load_state_dict(backup, strict=True)

        res = dict(logs)
        res["loss"] = loss.detach()

        print(f"Computed valid batch in {time.time() - t:.3f} seconds.")
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
    from caesar.data.allpdb import BatchedProteinPDBStream
    from flexloop.data import BatchStream
    from flexloop.utils import parse_options
    from caesar.modules.config import distance_to_structure_decoder as config_choices

    from caesar.modules.autoencoder import StructureAutoencoder, StructureDecoder

    opt = parse_options(
        "train a distance-to-structure decoder on PDB.",
        path="network/",
        config="default",
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
        multigpu="True",
        suffix="1",
        smoke="False",
        smoke_steps=5,
        smoke_valid_interval=1,
    )
    
    smoke = (opt.smoke == "True")
    
    multigpu = (opt.multigpu == "True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(opt.jax_seed))

    NUM_DEVICES = torch.cuda.device_count() if multigpu and torch.cuda.is_available() else 1

    config = getattr(config_choices, opt.config)
    sae = "sdd" if getattr(config, "is_decoder", False) else "sae"
    path = f"{opt.path}/torch/{sae}-{opt.config}-{opt.num_aa}-{opt.suffix}"
    writer = SummaryWriter(path)
    mul = opt.rebatch * opt.accumulate * NUM_DEVICES
    
    if smoke:
        print("SMOKE MODE: using tests/data/test_structure.npz instead of full dataset")
        config.is_decoder = False
        config.eval = True

        npz_path = "tests/data/test_structure.npz"
        batch = make_smoke_batch(npz_path, device="cpu", mul=mul)
        data = iter(infinite_stream(batch))
        valid_data = iter(infinite_stream(batch))

        # быстро и дёшево
        total_steps = int(opt.smoke_steps)
        valid_interval = int(opt.smoke_valid_interval)
    else:
        print("Attempting to load dataset...")
        data = BatchedProteinPDBStream(
            f"{opt.data_path}/allpdb/",
            seqres_aa="clusterSeqresAA",
            cutoff_resolution=4.0,
            p_complex=opt.p_complex,
            size=1024,
            min_size=16,
            max_size=1024,
        )
        data = iter(
            BatchStream(
                data,
                num_workers=0, # 32
                accumulate=int(opt.rebatch) * int(opt.accumulate) * int(NUM_DEVICES),
                prefetch_factor=2, # 32
            )
        )

        valid_data = BatchedProteinPDBStream(
            f"{opt.data_path}/allpdb/",
            seqres_aa="clusterSeqresAA",
            cutoff_resolution=4.0,
            p_complex=opt.p_complex,
            size=1024,
            min_size=16,
            max_size=1024,
            start_date="01/01/22",
            cutoff_date="12/31/23",
        )
        valid_data = iter(
            BatchStream(
                valid_data,
                num_workers=0, # 8
                accumulate=int(opt.rebatch) * int(opt.accumulate) * int(NUM_DEVICES),
                prefetch_factor=2, # 8
            )
        )
        print("Dataset successfully loaded.")

    if getattr(config, "is_decoder", False):
        model = StructureDecoder(config).to(device)
    else:
        model = StructureAutoencoder(config).to(device)

    # (с) "init params" analogue for LazyLinear: run one small forward
    print("Initializing model parameters...")
    item_0 = next(data)
    print("INPUT OF SHAPE:")
    for name, value in item_0.items():
        v = value if torch.is_tensor(value) else torch.as_tensor(value)
        print("  ", name, tuple(v.shape))

    init_batch = slice_batch_first_dim(item_0, int(opt.rebatch) * 100)
    init_batch = cast_float({k: torch.as_tensor(v) for k, v in init_batch.items()}, dtype=torch.float32)
    init_batch = move_to_device(init_batch, device)

    # (с) one dry forward to materialize LazyLinear
    gen0 = torch.Generator(device="cpu").manual_seed(int(opt.jax_seed))
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt.lr),
        betas=(float(opt.b1), float(opt.b2)),
        eps=1e-9,
    )
    print("Optimizer initialized.")

    step_train = model_step(model, config, rebatch=int(opt.rebatch), is_training=True, device=device)
    step_valid = model_step(model, config, rebatch=int(opt.rebatch), is_training=False, device=device)

    print("Constructing training loop...")
    key = torch.Generator(device="cpu").manual_seed(int(opt.jax_seed))
    aux_state: Dict[str, Any] = {}
    loop_state = State(key=key, step_id=0, model=model, optimizer=optimizer, aux_state=aux_state)

    print("Recovering previous state, if available...")
    loaded = load_loop_state(path, model, optimizer, device)
    if loaded is not None:
        loop_state = loaded

    training_inner = make_training_inner(
        model=model,
        optimizer=optimizer,
        step_fn=step_train,
        data=data,
        schedule=schedule,
        clip=float(opt.clip),
        accumulate=int(opt.accumulate),
        ema_weight=float(opt.ema_weight),
    )
    valid_inner = make_valid_inner(
        model=model,
        step_fn=step_valid,
        data=valid_data,
        use_ema=True,
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
