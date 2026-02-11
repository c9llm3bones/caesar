import argparse
import os
import time
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from caesar.data.allpdb import BatchedProteinPDBStream
from caesar.modules.config import distance_to_structure_decoder as config_choices
from caesar.modules.autoencoder import StructureAutoencoder, StructureDecoder

from caesar.training.train_structure_autoencoder import (
    slice_batch_first_dim,
    split_generator,
    make_smoke_batch,
    accumulate_stream,
    repeat_batch,
    prepare_batch,
    _call_model,
    model_step,
)

try:
    from flexloop.data import BatchStream
except Exception:
    BatchStream = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile structure autoencoder training step with torch.profiler."
    )
    p.add_argument("--data_path", type=str, default="", help="Path to dataset root.")
    p.add_argument("--config", type=str, default="small_inner", help="Config name.")
    p.add_argument("--num_aa", type=int, default=1024)
    p.add_argument("--p_complex", type=float, default=0.5)
    p.add_argument("--rebatch", type=int, default=1)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Use tests/data/test_structure.npz.")
    p.add_argument("--steps", type=int, default=10, help="Total profiled steps.")
    p.add_argument("--warmup", type=int, default=2, help="Warmup steps before profiling.")
    p.add_argument("--trace_dir", type=str, default="outputs/torch_profiler")
    p.add_argument("--overfit_one", action="store_true", help="Reuse first batch for all steps.")
    p.add_argument("--device", type=str, default="", help="cuda, cuda:0, cpu")
    p.add_argument("--use_cuda", action="store_true", help="Force CUDA if available.")
    p.add_argument("--profile_memory", action="store_true")
    p.add_argument("--record_shapes", action="store_true")
    return p.parse_args()


def build_stream(
    *,
    smoke: bool,
    data_path: str,
    num_aa: int,
    p_complex: float,
    num_workers: int,
    prefetch_factor: int,
    rebatch: int,
    accumulate: int,
    seed: int,
) -> Iterator[Dict[str, Any]]:
    if smoke:
        npz_path = "tests/data/test_structure.npz"
        batch = make_smoke_batch(npz_path, device="cpu", mul=rebatch * accumulate)
        return iter(repeat_batch(batch))

    stream = BatchedProteinPDBStream(
        f"{data_path}/allpdb/",
        seqres_aa="clusterSeqresAA",
        cutoff_resolution=4.0,
        p_complex=p_complex,
        size=num_aa,
        min_size=16,
        max_size=num_aa,
        seed=seed,
    )

    if num_workers <= 0:
        data_iter = iter(stream)
        if rebatch * accumulate != 1:
            data_iter = accumulate_stream(data_iter, rebatch * accumulate)
        return data_iter

    if BatchStream is None:
        raise RuntimeError("flexloop.data.BatchStream not available; set --num_workers 0.")

    return iter(
        BatchStream(
            stream,
            num_workers=num_workers,
            accumulate=rebatch * accumulate,
            prefetch_factor=prefetch_factor,
        )
    )


def init_model(config_name: str, device: torch.device) -> Tuple[torch.nn.Module, Any]:
    config = getattr(config_choices, config_name)
    if getattr(config, "is_decoder", False):
        model = StructureDecoder(config).to(device)
    else:
        model = StructureAutoencoder(config).to(device)
    return model, config


def main() -> None:
    opt = parse_args()

    seed = int(opt.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if opt.device:
        device = torch.device(opt.device)
    else:
        if opt.use_cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    data = build_stream(
        smoke=opt.smoke,
        data_path=opt.data_path,
        num_aa=opt.num_aa,
        p_complex=opt.p_complex,
        num_workers=int(opt.num_workers),
        prefetch_factor=int(opt.prefetch_factor),
        rebatch=int(opt.rebatch),
        accumulate=int(opt.accumulate),
        seed=int(opt.seed),
    )

    model, config = init_model(opt.config, device)

    # Initialize lazy params
    item_0 = next(data)
    init_batch = slice_batch_first_dim(item_0, int(opt.rebatch) * 100)
    init_batch = prepare_batch(init_batch, device=device, float_dtype=torch.float32)
    gen0 = torch.Generator(device=device).manual_seed(int(opt.seed))
    model.train(True)
    with torch.no_grad():
        _ = _call_model(model, init_batch, gen0)

    # Optionally overfit to one batch for compute-only profiling
    if opt.overfit_one:
        fixed = item_0
        data = iter(repeat_batch(fixed))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-9)
    step_fn = model_step(model, config, rebatch=int(opt.rebatch), is_training=True, device=None)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_dir = opt.trace_dir
    os.makedirs(trace_dir, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=0,
        warmup=int(opt.warmup),
        active=int(opt.steps),
        repeat=1,
    )

    prof = torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=bool(opt.record_shapes),
        profile_memory=bool(opt.profile_memory),
        with_stack=False,
    )

    key = torch.Generator(device=device).manual_seed(int(opt.seed))

    total_steps = int(opt.warmup) + int(opt.steps)
    step = 0

    with prof:
        while step < total_steps:
            t0 = time.time()
            with torch.profiler.record_function("data_loading"):
                item = next(data)

            with torch.profiler.record_function("cpu_cast"):
                item_t = prepare_batch(item, device=None, float_dtype=torch.float32)

            with torch.profiler.record_function("cpu_to_gpu"):
                item_t = prepare_batch(item_t, device=device, float_dtype=torch.float32)

            key, subkey = split_generator(key)

            with torch.profiler.record_function("forward"):
                loss, logs = step_fn(item_t, subkey)

            with torch.profiler.record_function("backward"):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

            with torch.profiler.record_function("optimizer_step"):
                optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            elapsed = time.time() - t0
            print(f"step {step} done in {elapsed:.3f}s | loss={float(loss.detach().cpu().item()):.6f}")

            prof.step()
            step += 1

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print("\nTop ops by", sort_key)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
    print(f"Traces written to: {trace_dir}")


if __name__ == "__main__":
    main()
