import os
import time
import math
import argparse
from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from caesar.data.allpdb import BatchedProteinPDBStream
from flexloop.data import BatchStream
from caesar.modules.autoencoder import StructureAutoencoder, StructureDecoder
from caesar.modules.config import distance_to_structure_decoder as config_choices

@dataclass
class LoopState:
    step_id: int


def flatten_dict(d: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{prefix}/{k}" if prefix else str(k)
            flat.update(flatten_dict(v, kk))
    else:
        flat[prefix] = d
    return flat


def cosine_decay_schedule(start_lr: float, decay_lr: float, warmup_steps: int, decay_steps: int):
    warmup_steps = int(warmup_steps)
    decay_steps = int(decay_steps)

    def schedule(step: int) -> float:
        step = int(step)
        if warmup_steps > 0 and step <= warmup_steps:
            lr = (step / warmup_steps) * start_lr
        else:
            t = (step - warmup_steps) % max(1, decay_steps)
            lr = (start_lr - decay_lr) * 0.5 * (1.0 + math.cos(math.pi * (t / max(1, decay_steps)))) + decay_lr
        lr = max(float(lr), 0.0)
        return lr

    return schedule


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = float(lr)


def to_torch_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            t = v
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
        else:
            out[k] = v
            continue

        if t.is_floating_point():
            t = t.float()

        if device.type == "cuda":
            if t.device.type == "cpu":
                try:
                    t = t.pin_memory()
                except Exception:
                    pass
            t = t.to(device=device, non_blocking=True)
        else:
            t = t.to(device=device)

        out[k] = t
    return out


def split_batch_dim0(batch: Dict[str, Any], i0: int, i1: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] >= i1:
            out[k] = v[i0:i1]
        else:
            out[k] = v
    return out


def forward_and_loss(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    rebatch: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if rebatch <= 1:
        return model(batch, generator=generator)

    n = None
    for v in batch.values():
        if torch.is_tensor(v) and v.dim() >= 1:
            n = int(v.shape[0])
            break
    if n is None:
        return model(batch, generator=generator)

    chunk = int(math.ceil(n / rebatch))
    total_loss = None

    losses_sum: Dict[str, torch.Tensor] = {}
    last_out: Dict[str, Any] = {}

    for r in range(rebatch):
        i0, i1 = r * chunk, min(n, (r + 1) * chunk)
        if i0 >= i1:
            continue
        mb = split_batch_dim0(batch, i0, i1)
        loss_r, out_r = model(mb, generator=generator)
        if isinstance(out_r, dict):
            last_out = out_r

        total_loss = loss_r if total_loss is None else (total_loss + loss_r)

        losses_r = out_r.get("losses", {}) if isinstance(out_r, dict) else {}
        losses_r = flatten_dict(losses_r)
        for k, v in losses_r.items():
            if torch.is_tensor(v):
                losses_sum[k] = v if k not in losses_sum else (losses_sum[k] + v)

    denom = float(rebatch)
    total_loss = total_loss / denom

    avg_losses = {k: v / denom for k, v in losses_sum.items()}
    out = dict(last_out) if isinstance(last_out, dict) else {}
    out["losses"] = avg_losses
    return total_loss, out


def save_checkpoint(path: str, step: int, model, optimizer, scaler: Optional[torch.amp.GradScaler]):
    ckpt = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model, optimizer, scaler: Optional[torch.amp.GradScaler]) -> Optional[int]:
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["step"])


def train_one_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    batch: Dict[str, Any],
    rebatch: int,
    accumulate: int,
    clip: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    generator: Optional[torch.Generator],
    log_losses: bool,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    acc_n = max(1, int(accumulate))

    total_loss_scalar = 0.0
    losses_sum_scalar: Dict[str, float] = {}

    for _ in range(acc_n):
        if use_amp:
            assert scaler is not None
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss, out = forward_and_loss(model, batch, rebatch=rebatch, generator=generator)
                loss_to_backward = loss / float(acc_n)
            scaler.scale(loss_to_backward).backward()
        else:
            loss, out = forward_and_loss(model, batch, rebatch=rebatch, generator=generator)
            (loss / float(acc_n)).backward()

        if log_losses:
            total_loss_scalar += float(loss.detach().float().cpu().item())

            losses = out.get("losses", {}) if isinstance(out, dict) else {}
            losses = flatten_dict(losses)
            for k, v in losses.items():
                kk = f"{k}_loss"
                if torch.is_tensor(v):
                    losses_sum_scalar[kk] = losses_sum_scalar.get(kk, 0.0) + float(v.detach().float().cpu().item())
                elif isinstance(v, (float, int)):
                    losses_sum_scalar[kk] = losses_sum_scalar.get(kk, 0.0) + float(v)
    if clip is not None and float(clip) > 0.0:
        if use_amp and scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip))

    # optimizer step
    if use_amp and scaler is not None and scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    if log_losses:
        denom = float(acc_n)
        losses_avg = {k: v / denom for k, v in losses_sum_scalar.items()}
        return total_loss_scalar, losses_avg
    else:
        return 0.0, {}


@torch.no_grad()
def valid_one_step(
    *,
    model: torch.nn.Module,
    batch: Dict[str, Any],
    rebatch: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> Tuple[float, Dict[str, float]]:
    model.eval()

    if use_amp:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            loss, out = forward_and_loss(model, batch, rebatch=rebatch, generator=generator)
    else:
        loss, out = forward_and_loss(model, batch, rebatch=rebatch, generator=generator)

    loss_scalar = float(loss.detach().float().cpu().item())
    losses = out.get("losses", {}) if isinstance(out, dict) else {}
    losses = flatten_dict(losses)

    losses_scalar: Dict[str, float] = {}
    for k, v in losses.items():
        kk = f"{k}_loss"
        if torch.is_tensor(v):
            losses_scalar[kk] = float(v.detach().float().cpu().item())
        elif isinstance(v, (float, int)):
            losses_scalar[kk] = float(v)

    return loss_scalar, losses_scalar


def parse_args():
    p = argparse.ArgumentParser("train_structure_autoencoder (torch)")

    # paths/config
    p.add_argument("--path", type=str, default="network/")
    p.add_argument("--config", type=str, default="small_inner")
    p.add_argument("--data_path", type=str, default="")
    p.add_argument("--suffix", type=str, default="1")

    # data
    p.add_argument("--num_aa", type=int, default=1024)
    p.add_argument("--p_complex", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=8)
    p.add_argument("--valid_num_workers", type=int, default=8)
    p.add_argument("--valid_prefetch_factor", type=int, default=8)

    # optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decay_lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=1_000)
    p.add_argument("--decay_steps", type=int, default=500_000)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--b1", type=float, default=0.9)
    p.add_argument("--b2", type=float, default=0.99)
    p.add_argument("--rebatch", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=1)

    # training loop
    p.add_argument("--valid_interval", type=int, default=100)
    p.add_argument("--valid_batches", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--resume", type=str, default="True")

    # compute/seed
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multigpu", type=str, default="False")  # placeholder (no DDP)
    p.add_argument("--cudnn_benchmark", type=str, default="True")

    # AMP (V100: fp16)
    p.add_argument("--amp", type=str, default="True")
    p.add_argument("--amp_dtype", type=str, default="fp16")  # fp16 recommended on V100

    return p.parse_args()


def main():
    opt = parse_args()

    torch.manual_seed(opt.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # V100: Tensor Cores via AMP fp16
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = (opt.cudnn_benchmark == "True")

    # AMP config (V100)
    amp_enabled = (opt.amp == "True") and (device.type == "cuda")
    amp_dtype = torch.float16 if str(opt.amp_dtype).lower() in ("fp16", "float16") else torch.bfloat16
    if device.type == "cuda" and amp_enabled and amp_dtype != torch.float16:
        print(f"[warn] V100: amp_dtype={amp_dtype} обычно не ускоряет. Лучше fp16.")

    # config + model class
    config = getattr(config_choices, opt.config)
    is_decoder = bool(getattr(config, "is_decoder", False))

    ModelClass = StructureDecoder if is_decoder else StructureAutoencoder
    model = ModelClass(config).to(device)

    # outdir
    sae = "sdd" if is_decoder else "sae"
    outdir = f"{opt.path}/salad_torch/{sae}-{opt.config}-{opt.num_aa}-{opt.suffix}"
    os.makedirs(outdir, exist_ok=True)
    writer = SummaryWriter(outdir)

    # write model description
    try:
        with open(os.path.join(outdir, "model_description"), "w", encoding="utf-8") as f:
            f.write(str(model))
    except Exception as e:
        print("Could not write model_description:", e)

    # dataset
    print("Attempting to load dataset...")
    train_stream = BatchedProteinPDBStream(
        f"{opt.data_path}/allpdb/",
        seqres_aa="clusterSeqresAA",
        cutoff_resolution=4.0,
        p_complex=opt.p_complex,
        size=int(opt.num_aa),
        min_size=16,
        max_size=int(opt.num_aa),
    )
    train_iter = iter(
        BatchStream(
            train_stream,
            num_workers=int(opt.num_workers),
            accumulate=int(opt.rebatch) * int(opt.accumulate) * 1,
            prefetch_factor=int(opt.prefetch_factor),
        )
    )

    valid_stream = BatchedProteinPDBStream(
        f"{opt.data_path}/allpdb/",
        seqres_aa="clusterSeqresAA",
        cutoff_resolution=4.0,
        p_complex=opt.p_complex,
        size=int(opt.num_aa),
        min_size=16,
        max_size=int(opt.num_aa),
        start_date="01/01/22",
        cutoff_date="12/31/23",
    )
    valid_iter = iter(
        BatchStream(
            valid_stream,
            num_workers=int(opt.valid_num_workers),
            accumulate=int(opt.rebatch) * int(opt.accumulate) * 1,
            prefetch_factor=int(opt.valid_prefetch_factor),
        )
    )
    print("Dataset successfully loaded.")

    if device.type == "cuda":
        generator = torch.Generator(device="cuda")
    else:
        generator = torch.Generator(device="cpu")
    generator.manual_seed(int(opt.seed))

    # dry forward to initialize lazy stuff
    model.train()
    with torch.no_grad():
        sample = to_torch_batch(next(train_iter), device)
        if amp_enabled:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _ = model(sample, generator=generator)
        else:
            _ = model(sample, generator=generator)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # optimizer + scaler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(opt.lr),
        betas=(float(opt.b1), float(opt.b2)),
        eps=1e-9,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16)) if device.type == "cuda" else None

    # schedule + total steps 
    lr_fn = cosine_decay_schedule(opt.lr, opt.decay_lr, opt.warmup_steps, opt.decay_steps)
    total_steps = int(opt.warmup_steps) + int(opt.decay_steps) + 1

    # resume
    ckpt_path = os.path.join(outdir, "checkpoint.pt")
    start_step = 0
    if (opt.resume == "True"):
        loaded = load_checkpoint(ckpt_path, model, optimizer, scaler)
        if loaded is not None:
            start_step = loaded
            print(f"Resumed from step {start_step}")

    print("Starting training...")
    print(f"Log files and tensorboard records will be written to {outdir}")
    print(f"Device: {device} | AMP: {amp_enabled} | amp_dtype: {amp_dtype}")

    # loop
    for step in range(start_step, total_steps):
        t0 = time.time()

        # lr
        lr = lr_fn(step)
        set_optimizer_lr(optimizer, lr)

        # load batch
        t_load = time.time()
        raw = next(train_iter)
        load_time = time.time() - t_load

        batch = to_torch_batch(raw, device)

        # train step
        t_step = time.time()
        log_now = (step % int(opt.log_interval) == 0)
        total_loss, losses_avg = train_one_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            batch=batch,
            rebatch=int(opt.rebatch),
            accumulate=int(opt.accumulate),
            clip=float(opt.clip),
            use_amp=amp_enabled,
            amp_dtype=amp_dtype,
            generator=generator,
            log_losses=log_now,
        )
        step_time = time.time() - t_step
        total_time = time.time() - t0

        # prints + TB
        if log_now:
            print(
                f"Step {step}, load time {load_time:.3f} s, step time {step_time:.3f} s, total {total_time:.3f} s, "
                f"lr {lr:.3e}, loss {total_loss:.6f}"
            )
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/loss", total_loss, step)
            for k, v in losses_avg.items():
                writer.add_scalar(f"train/{k}", v, step)

        # valid
        if (step % int(opt.valid_interval) == 0) and (step != start_step):
            v_losses = []
            v_parts: Dict[str, float] = {}
            for _ in range(int(opt.valid_batches)):
                v_raw = next(valid_iter)
                v_batch = to_torch_batch(v_raw, device)
                v_loss, v_dict = valid_one_step(
                    model=model,
                    batch=v_batch,
                    rebatch=int(opt.rebatch),
                    use_amp=amp_enabled,
                    amp_dtype=amp_dtype,
                    generator=generator,
                )
                v_losses.append(v_loss)
                for k, v in v_dict.items():
                    v_parts[k] = v_parts.get(k, 0.0) + float(v)

            v_loss_mean = float(np.mean(v_losses)) if v_losses else float("nan")
            print(f"Valid @ step {step}: loss={v_loss_mean:.6f}")
            writer.add_scalar("valid/loss", v_loss_mean, step)
            for k, v in v_parts.items():
                writer.add_scalar(f"valid/{k}", v / float(max(1, int(opt.valid_batches))), step)

        # checkpoint
        if (step % int(opt.save_interval) == 0) and (step > 0):
            save_checkpoint(ckpt_path, step, model, optimizer, scaler)
            print(f"Saved checkpoint: {ckpt_path} (step {step})")

    # final checkpoint
    save_checkpoint(ckpt_path, total_steps, model, optimizer, scaler)
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()