#!/usr/bin/env python3
"""PyTorch тренинг: инициализация, train/valid, lr schedule, EMA, чекпоинты, tensorboard."""

import os
import time
import argparse
import math
import numbers
from copy import deepcopy

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import csv
from caesar.data.allpdb import BatchedProteinPDBStream
from flexloop.data import BatchStream
from caesar.modules.config import distance_to_structure_decoder as config_choices

from caesar.modules.autoencoder import StructureAutoencoder, StructureDecoder
DEBUG = True

@dataclass
class LoopState:
    step_id: int
    model_state: Dict[str, Any]
    opt_state: Dict[str, Any]
    aux_state: Dict[str, Any]
from torch import profiler
import os
import torch

def profile_train_inner(
    *,
    train_inner,
    loop_state,
    optimizer=None,
    lr_fn=None,
    steps: int = 60,
    logdir: str = "./tb_prof_train_inner",
    wait: int = 2,
    warmup: int = 2,
    active: int = 20,
    export_chrome: bool = True,
    with_stack: bool = True,
    record_shapes: bool = True,
    profile_memory: bool = True,
    sync_each_step: bool = False,
    annotate_steps: bool = True,
    export_csv: bool = True,
    csv_path: str = "prof_step_breakdown.csv",
):
    os.makedirs(logdir, exist_ok=True)

    activities = [profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(profiler.ProfilerActivity.CUDA)

    sched = profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)

    def on_trace_ready_fn(p: profiler.profile):
        # ВАЖНО: tensorboard_trace_handler уже сохраняет chrome-trace.
        # Если дополнительно вызвать export_chrome_trace(), PyTorch упадёт с
        # RuntimeError: Trace is already saved.
        profiler.tensorboard_trace_handler(logdir)(p)

    # короткий прогрев (вне профиля)
    for _ in range(3):
        step = loop_state.step_id
        lr = lr_fn(step) if lr_fn is not None else None
        if lr_fn is not None and optimizer is not None:
            for g in optimizer.param_groups:
                g["lr"] = lr
        loop_state, _, _ = train_inner(loop_state, lr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    total_iters = max(steps, wait + warmup + active)
    step_rows = []  # будет заполнено после warmup, только во время окна профилирования

    with profiler.profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=on_trace_ready_fn,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        for _ in range(total_iters):
            step = loop_state.step_id
            lr = lr_fn(step) if lr_fn is not None else None
            if lr_fn is not None and optimizer is not None:
                for g in optimizer.param_groups:
                    g["lr"] = lr

            step_wall_t0 = time.time()
            if annotate_steps:
                with profiler.record_function(f"TRAIN_STEP_{step:06d}"):
                    with profiler.record_function("TRAIN_INNER_FULL_PASS"):
                        loop_state, step_stats, loss_dict = train_inner(loop_state, lr)
            else:
                with profiler.record_function("TRAIN_INNER_FULL_PASS"):
                    loop_state, step_stats, loss_dict = train_inner(loop_state, lr)

            step_wall_s = time.time() - step_wall_t0

            sync_s = 0.0
            if sync_each_step and torch.cuda.is_available():
                s0 = time.time()
                torch.cuda.synchronize()
                sync_s = time.time() - s0

            # Сохраняем поминутную (по шагам) статистику, чтобы потом легко сводить в Perfetto/Excel.
            if export_csv:
                row = {"step": step, "lr": float(lr), "step_wall_s": float(step_wall_s), "sync_s": float(sync_s)}
                if isinstance(step_stats, dict):
                    for k, v in step_stats.items():
                        # типично: load_time/step_time/…
                        try:
                            row[k] = float(v)
                        except Exception:
                            pass
                step_rows.append(row)
            prof.step()

    print("\n=== TOP CUDA (cuda_time_total) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n=== TOP CPU (self_cpu_time_total) ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

    cpu_only = []
    for e in prof.key_averages():
        if e.key.startswith("aten::") and getattr(e, "cuda_time_total", 0.0) == 0.0:
            cpu_only.append(e)
    cpu_only = sorted(cpu_only, key=lambda x: x.self_cpu_time_total, reverse=True)[:30]
    print("\n=== Suspicious CPU-only aten:: ops (top self_cpu, cuda=0) ===")
    for e in cpu_only:
        print(f"{e.key:45s} self_cpu={e.self_cpu_time_total/1e3:.3f}ms  calls={e.count}")

    if export_csv and step_rows:
        csv_file = os.path.join(logdir, csv_path)
        # Динамические колонки: объединяем все ключи из row dict.
        fieldnames = []
        seen = set()
        for r in step_rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
        with open(csv_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(step_rows)
        print(f"Step breakdown CSV written to: {csv_file}")

    print(f"\nTrace written to: {logdir}")
    if export_chrome:
        print(f"Chrome trace: {os.path.join(logdir, 'trace.json')}")
        

    return loop_state

def print_batch_structure(batch, indent=0):
    prefix = "  " * indent
    if isinstance(batch, dict):
        print(f"{prefix}dict with keys:")
        for k, v in batch.items():
            print(f"{prefix}  {k}: ", end="")
            if hasattr(v, 'shape'):
                print(f"Tensor shape {v.shape}, dtype {v.dtype}")
            else:
                print(f"{type(v)}")
            if isinstance(v, (dict, list, tuple)):
                print_batch_structure(v, indent+2)
    elif isinstance(batch, (list, tuple)):
        print(f"{prefix}{type(batch).__name__} of length {len(batch)}")
        if len(batch) > 0:
            print_batch_structure(batch[0], indent+1)
    else:
        print(f"{prefix}{type(batch)}")

def to_torch_batch(
    batch: dict,
    device: torch.device,
    float_dtype: torch.dtype = torch.float32,  
    non_blocking: bool = True,
) -> dict:
    
    import numpy as np
    if DEBUG:
       # print(batch)
       pass
    def _convert(x):
        if isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(_convert(v) for v in x)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not torch.is_tensor(x):
            try:
                x = torch.as_tensor(x)
            except Exception:
                return x  

        if torch.is_tensor(x):
            needs_device = x.device != device
            if x.is_floating_point():
                needs_dtype = x.dtype != float_dtype
                x = x.to(
                    device=device if needs_device else None,
                    dtype=float_dtype if needs_dtype else None,
                    non_blocking=non_blocking
                )
            else:
                if needs_device:
                    x = x.to(device, non_blocking=non_blocking)
        return x

    return {k: _convert(v) for k, v in batch.items()}

def forward_and_loss(
    model: nn.Module,
    batch: dict,
    rebatch: int = 1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward with optional rebatch and return loss + named losses."""
    if rebatch <= 1:
        if DEBUG:
            for k, v in batch.items():
                if torch.is_tensor(v) and v.ndim >= 2 and v.shape[0] % 8 != 0:
                    print(f"Warning: {k}.shape[0]={v.shape[0]} not multiple of 8 → Tensor Cores may not engage")
        result = model(batch)
        if isinstance(result, tuple) and len(result) == 2:
            loss, out = result
        elif isinstance(result, dict):
            loss = result.get("loss") or sum(result.get("losses", {}).values())
            out = result
        else:
            raise RuntimeError("Model forward must return (loss, out_dict) or out_dict containing 'loss' or 'losses'.")
        losses_dict = {}
        if isinstance(out, dict):
            def _collect_losses(prefix: str, value: Any):
                if torch.is_tensor(value):
                    losses_dict[f"{prefix}_loss"] = value.detach()
                    return
                if isinstance(value, numbers.Number):
                    losses_dict[f"{prefix}_loss"] = torch.tensor(value, device=loss.device)
                    return
                if isinstance(value, dict):
                    for sub_k, sub_v in value.items():
                        _collect_losses(f"{prefix}/{sub_k}", sub_v)

            for k, v in out.get("losses", {}).items():
                _collect_losses(k, v)
        return loss, losses_dict

    batch_size = None
    for v in batch.values():
        if torch.is_tensor(v):
            batch_size = v.shape[0]
            break
        elif isinstance(v, (list, tuple)):
            batch_size = len(v)
            break
    if batch_size is None:
        raise RuntimeError("Не удалось определить batch size для rebatch.")
    chunk_size = batch_size // rebatch
    total_loss = None
    total_losses = {}
    for i in range(rebatch):
        subbatch = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                subbatch[k] = v[i * chunk_size : (i + 1) * chunk_size]
            else:
                if isinstance(v, (list, tuple)):
                    subbatch[k] = type(v)(v[i * chunk_size : (i + 1) * chunk_size])
                else:
                    subbatch[k] = v
        loss, losses_dict = forward_and_loss(model, subbatch, rebatch=1)
        total_loss = loss if total_loss is None else (total_loss + loss)
        for kk, vv in losses_dict.items():
            total_losses.setdefault(kk, 0.0)
            total_losses[kk] = total_losses[kk] + (vv.detach() if torch.is_tensor(vv) else vv)
    avg_loss = total_loss / float(rebatch)
    avg_losses = {k: (v / float(rebatch)) for k, v in total_losses.items()}
    return avg_loss, avg_losses

def cosine_decay_schedule(start_lr, decay_lr, warmup_steps, decay_steps):
    def get_lr(count):
        if count <= warmup_steps:
            return float(count) / float(max(1, warmup_steps)) * start_lr
        x = ((count - warmup_steps) % decay_steps) / float(decay_steps)
        val = (start_lr - decay_lr) * 0.5 * (1.0 + math.cos(x * math.pi)) + decay_lr
        return float(max(val, 0.0))
    return get_lr

def init_ema_params(model: nn.Module):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

def update_ema(model: nn.Module, ema: Dict[str, torch.Tensor], ema_weight: float):
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ema_p = ema[n].to(p.device)
        ema_p.mul_(ema_weight).add_(p.detach() * (1.0 - ema_weight))
        ema[n] = ema_p.cpu().clone()

def load_ema_to_model(model: nn.Module, ema: Dict[str, torch.Tensor], device: torch.device):
    for n, p in model.named_parameters():
        if n in ema:
            p.data.copy_(ema[n].to(device))

def save_checkpoint(path, loop_state: LoopState):
    os.makedirs(path, exist_ok=True)
    fname = os.path.join(path, "checkpoint.pth")
    payload = {
        "step_id": loop_state.step_id,
        "model_state": loop_state.model_state,
        "opt_state": loop_state.opt_state,
        "aux_state": loop_state.aux_state,
    }
    torch.save(payload, fname)

def load_checkpoint(path) -> Optional[LoopState]:
    fname = os.path.join(path, "checkpoint.pth")
    if not os.path.exists(fname):
        return None
    payload = torch.load(fname, map_location="cpu")
    return LoopState(
        step_id=payload["step_id"],
        model_state=payload["model_state"],
        opt_state=payload["opt_state"],
        aux_state=payload["aux_state"],
    )

def make_training_inner(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_iter,
    device: torch.device,
    accumulate: int = 1,
    rebatch: int = 1,
    ema_weight: float = 0.999,
    writer: Optional[SummaryWriter] = None,
    with_state: bool = False,
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    log_interval: int = 10,
    *,
    profile_markers: bool = False,
    debug_device_checks: bool = False,
    zero_grad_set_to_none: bool = True,
):
    model.train()

    model_params_devices = {name: param.device for name, param in model.named_parameters()}
    params_on_wrong_device = {name: dev for name, dev in model_params_devices.items() if dev != device}
    check_all_params(model_params_devices, params_on_wrong_device, device)
  
    use_amp = bool(amp) and torch.cuda.is_available()
    if use_amp:
        # Для V100 (sm70) bf16 не поддерживается; для Tensor Cores нужен fp16.
        try:
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            if (cc_major, cc_minor) < (8, 0) and amp_dtype == torch.bfloat16:
                print("Warning: bf16 AMP requested but GPU < sm80; forcing fp16 for tensor cores.")
                amp_dtype = torch.float16
        except Exception:
            if amp_dtype == torch.bfloat16:
                print("Warning: bf16 AMP requested; forcing fp16 for tensor cores.")
                amp_dtype = torch.float16

    # torch.cuda.amp.GradScaler is deprecated in новых версиях; используем torch.amp.GradScaler('cuda', ...)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16) if torch.cuda.is_available() else None
    if scaler is not None:
        print(f"GradScaler enabled: {scaler.is_enabled()} (amp_dtype={amp_dtype})")

    last_log_time = time.time()

    def training_inner(loop_state: LoopState, lr: float):
        nonlocal last_log_time
        t0 = time.time()
        if profile_markers:
            with profiler.record_function("DATALOAD"):
                batch = next(data_iter)
        else:
            batch = next(data_iter)
        load_time = time.time() - t0

        if profile_markers:
            with profiler.record_function("H2D"):
                batch = to_torch_batch(batch, device)
        else:
            batch = to_torch_batch(batch, device)

        
        #assert next(model.parameters()).device == device, f"Model is not on the correct device: {device}"
        #model_params_devices = {name: param.device for name, param in model.named_parameters()}
        #params_on_wrong_device = {name: dev for name, dev in model_params_devices.items() if dev != device}
        #check_all_params(model_params_devices, params_on_wrong_device, device)
    
        if debug_device_checks:
            for key, tensor in batch.items():
                assert tensor.device == device, f"Batch tensor {key} is not on the correct device: {tensor.device}"

        step_t0 = time.time()
        if profile_markers:
            with profiler.record_function("ZERO_GRAD"):
                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
        else:
            optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
        if use_amp:
            if debug_device_checks:
                assert next(model.parameters()).device == device, f"Model is not on the correct device: {device}"
                for key, tensor in batch.items():
                    assert tensor.device == device, f"Batch tensor {key} is not on the correct device: {tensor.device}"

            if profile_markers:
                with profiler.record_function("FWD"):
                    with profiler.record_function("AMP_FWD"):
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            loss, loss_dict = model(batch) if rebatch == 1 else forward_and_loss(model, batch, rebatch=rebatch)
            else:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss, loss_dict = model(batch) if rebatch == 1 else forward_and_loss(model, batch, rebatch=rebatch)

            if debug_device_checks:
                assert next(model.parameters()).device == device, f"Model is not on the correct device: {device}"
                for key, tensor in batch.items():
                    assert tensor.device == device, f"Batch tensor {key} is not on the correct device: {tensor.device}"

            if scaler is not None and scaler.is_enabled():
                if profile_markers:
                    with profiler.record_function("BWD"):
                        scaler.scale(loss).backward()
                    with profiler.record_function("OPT"):
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if profile_markers:
                    with profiler.record_function("BWD"):
                        loss.backward()
                    with profiler.record_function("OPT"):
                        optimizer.step()
                else:
                    loss.backward()
                    optimizer.step()
        else:
            if profile_markers:
                with profiler.record_function("FWD"):
                    loss, loss_dict = model(batch) if rebatch == 1 else forward_and_loss(model, batch, rebatch=rebatch)
                with profiler.record_function("BWD"):
                    loss.backward()
                with profiler.record_function("OPT"):
                    optimizer.step()
            else:
                loss, loss_dict = model(batch) if rebatch == 1 else forward_and_loss(model, batch, rebatch=rebatch)
                loss.backward()
                optimizer.step()

        # EMA disabled for now.
        # if "ema" in loop_state.aux_state:
        #     update_ema(model, loop_state.aux_state["ema"], ema_weight)
        step_time = time.time() - step_t0

        step_id = loop_state.step_id + 1

        loggables = {"loss": loss.detach()}
        if isinstance(loss_dict, dict):
            def _collect_logs(prefix: str, value: Any):
                if torch.is_tensor(value):
                    loggables[prefix] = value.detach()
                    return
                if isinstance(value, numbers.Number):
                    loggables[prefix] = torch.tensor(value, device=loss.device)
                    return
                if isinstance(value, dict):
                    for sub_k, sub_v in value.items():
                        _collect_logs(f"{prefix}/{sub_k}", sub_v)
            for k, v in loss_dict.items():
                _collect_logs(k, v)
        checkpointables = {
            "state_getter": lambda: {
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
            }
        }
        new_loop_state = LoopState(
            step_id,
            loop_state.model_state,
            loop_state.opt_state,
            loop_state.aux_state,
        )
        total_time = time.time() - t0
        if step_id % max(1, int(log_interval)) == 0:
            loss_scalar = float(loggables["loss"].detach().cpu().item())
            elapsed = time.time() - last_log_time
            last_log_time = time.time()
            print(
                f"Step {step_id}, load {load_time:.3f}s, "
                f"step {step_time:.3f}s, total {total_time:.3f}s, "
                f"loss {loss_scalar:.6f}, interval {elapsed:.3f}s"
            )
        return new_loop_state, loggables, checkpointables
    return training_inner
    
def make_valid_inner(model: nn.Module, data_iter, device: torch.device, rebatch: int = 1,
                     amp: bool = False, amp_dtype: torch.dtype = torch.float16):
    model.eval()
    @torch.no_grad()
    def valid_inner(loop_state: LoopState):
        t0 = time.time()
        batch = next(data_iter)
        batch = to_torch_batch(batch, device)
        saved = None
        if "ema" in loop_state.aux_state:
            saved = deepcopy(model.state_dict())
            load_ema_to_model(model, loop_state.aux_state["ema"], device)
        if amp and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss, loss_dict = model(batch) if rebatch == 1 else forward_and_loss(model, batch, rebatch=rebatch)
        else:
            loss, loss_dict = model(batch) if rebatch == 1 else forward_and_loss(model, batch, rebatch=rebatch)
        if saved is not None:
            model.load_state_dict(saved)
        
        metrics = {"val_loss": float(loss.cpu().item())}
        if isinstance(loss_dict, dict):
            def _collect_metrics(prefix: str, value: Any):
                if torch.is_tensor(value):
                    t = value.detach()
                    if t.numel() == 1:
                        metrics[f"val_{prefix}"] = float(t.cpu().item())
                    return
                if isinstance(value, numbers.Number):
                    metrics[f"val_{prefix}"] = float(value)
                    return
                if isinstance(value, dict):
                    for sub_k, sub_v in value.items():
                        _collect_metrics(f"{prefix}/{sub_k}", sub_v)

            for k, v in loss_dict.items():
                _collect_metrics(k, v)
        
        print(f"Computed valid batch in {time.time() - t0:.3f} seconds. val_loss={metrics['val_loss']:.6f}")
        return metrics
    return valid_inner

def training_loop(
    path: str,
    train_inner,
    valid_inner,
    writer: SummaryWriter,
    loop_state: LoopState,
    max_steps: int,
    valid_interval: int = 100,
    *,
    lr_fn=None,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    os.makedirs(path, exist_ok=True)
    step = loop_state.step_id
    checkpointables = {}
    while step < max_steps:
        lr = None
        if lr_fn is not None:
            lr = lr_fn(step)
            if optimizer is not None:
                for g in optimizer.param_groups:
                    g["lr"] = lr
        new_state, loggables, checkpointables = train_inner(loop_state, lr)
        loop_state = new_state
        step = loop_state.step_id

        if step % valid_interval == 0:
            val_res = valid_inner(loop_state)
            if writer:
                writer.add_scalar("valid/loss", val_res.get("val_loss", 0.0), step)
                for k, v in val_res.items():
                    if k != "val_loss":
                        writer.add_scalar(f"valid/{k}", v, step)
            state_getter = checkpointables.get("state_getter")
            if state_getter is not None:
                state = state_getter()
                loop_state.model_state = state["model_state"]
                loop_state.opt_state = state["opt_state"]
            save_checkpoint(path, loop_state)
    state_getter = checkpointables.get("state_getter")
    if state_getter is not None:
        state = state_getter()
        loop_state.model_state = state["model_state"]
        loop_state.opt_state = state["opt_state"]
    save_checkpoint(path, loop_state)
    print("Training finished.")

def check_all_params(model_params_devices, params_on_wrong_device, device):

    if params_on_wrong_device:
        print(f"ОШИБКА: Найдены параметры не на устройстве {device}:")
        for name, dev in params_on_wrong_device.items():
            print(f"  {name}: {dev}")
        # Можно даже показать распределение
        devices_count = {}
        for dev in model_params_devices.values():
            devices_count[dev] = devices_count.get(dev, 0) + 1
        print(f"\nРаспределение параметров по устройствам: {devices_count}")
        raise AssertionError(f"Модель имеет параметры не на устройстве {device}")
    else:
        print(f"Все параметры модели на устройстве {device}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="network/", help="output path")
    parser.add_argument("--config", default="small_inner")
    parser.add_argument("--data_path", default="/disk/2tb/edelkin/data", help="path to allpdb")
    parser.add_argument("--num_aa", type=int, default=1024)
    parser.add_argument("--p_complex", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--decay_steps", type=int, default=500000)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.99)
    parser.add_argument("--ema_weight", type=float, default=0.999)
    parser.add_argument("--rebatch", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--jax_seed", type=int, default=42)
    parser.add_argument("--multigpu", default="False")
    parser.add_argument("--suffix", default="1")
    parser.add_argument("--amp", default="True")
    parser.add_argument("--amp_dtype", default="fp16")
    parser.add_argument("--matmul_precision", default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--allow_tf32", default="False")
    parser.add_argument("--cudnn_benchmark", default="True")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    return parser.parse_args()

def main():
    opt = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(opt.jax_seed)

    # perf knobs
    try:
        torch.set_float32_matmul_precision(opt.matmul_precision)
    except Exception:
        pass

    if torch.cuda.is_available():
        allow_tf32 = (opt.allow_tf32 == "True")
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = (opt.cudnn_benchmark == "True")

    # model
    config = getattr(config_choices, opt.config)
    is_decoder = getattr(config, "is_decoder", False)
    ModelClass = StructureDecoder if is_decoder else StructureAutoencoder
    model = ModelClass(config).to(device)

    # data
    num_devices_for_batch = 1
    loader_workers = int(opt.num_workers)
    loader_prefetch = int(opt.prefetch_factor) if loader_workers > 0 else None

    data_stream = BatchedProteinPDBStream(
        f"{opt.data_path}/allpdb/",
        seqres_aa="clusterSeqresAA",
        cutoff_resolution=4.0,
        p_complex=opt.p_complex,
        size=opt.num_aa,
        min_size=16,
        max_size=opt.num_aa,
        seed=opt.jax_seed,
    )

    data_iter = iter(BatchStream(
        dataset=data_stream,
        num_workers=loader_workers,
        accumulate=opt.rebatch * opt.accumulate * num_devices_for_batch,
        prefetch_factor=loader_prefetch,
    ))

    # dry run (инициализация lazy/compile-кэшей)
    sample = to_torch_batch(next(data_iter), device)
    amp_enabled = (opt.amp == "True")
    amp_dtype = torch.float16 if str(opt.amp_dtype).lower() in ("fp16", "float16") else torch.bfloat16
    if amp_enabled and torch.cuda.is_available():
        try:
            cc_major, cc_minor = torch.cuda.get_device_capability(0)
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA device: {gpu_name} (sm{cc_major}{cc_minor})")
            if (cc_major, cc_minor) < (8, 0) and amp_dtype == torch.bfloat16:
                print("bf16 AMP requested but GPU < sm80; switching to fp16 for V100 Tensor Cores.")
                amp_dtype = torch.float16
        except Exception:
            if amp_dtype == torch.bfloat16:
                print("bf16 AMP requested; switching to fp16 for Tensor Cores.")
                amp_dtype = torch.float16

        # TF32 не актуален для V100, но пусть будет явно выключен.
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        # В некоторых версиях PyTorch это может влиять на скорость fp16 GEMM.
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        except Exception:
            pass
    with torch.no_grad():
        if amp_enabled and torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _ = model(sample)
        else:
            _ = model(sample)

    # optimizer + lr schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-9)
    lr_fn = cosine_decay_schedule(opt.lr, opt.decay_lr, opt.warmup_steps, opt.decay_steps)

    # loop_state
    loop_state = LoopState(
        step_id=0,
        model_state=model.state_dict(),
        opt_state=optimizer.state_dict(),
        aux_state={},
    )

    # train_inner (writer=None, log_interval очень большой, чтобы не было .cpu().item() в логах)
    train_inner = make_training_inner(
        model=model,
        optimizer=optimizer,
        data_iter=data_iter,
        device=device,
        accumulate=opt.accumulate,
        rebatch=opt.rebatch,
        ema_weight=opt.ema_weight,
        writer=None,
        with_state=False,
        amp=amp_enabled,
        amp_dtype=amp_dtype,
        log_interval=10**9,
        profile_markers=True,
    )

    # PROFILE ONLY (никакого training_loop)
    outdir = f"{opt.path}/prof_only/{'sdd' if is_decoder else 'sae'}-{opt.config}-{opt.num_aa}-{opt.suffix}"
    profile_train_inner(
        train_inner=train_inner,
        loop_state=loop_state,
        optimizer=optimizer,
        lr_fn=lr_fn,
        steps=60,
        logdir=os.path.join(outdir, "tb_prof_train_inner"),
        wait=2,
        warmup=2,
        active=20,
        export_chrome=True,
        with_stack=True,
        sync_each_step=False,
    )
    return

if __name__ == "__main__":
    main()