import time
import random
import numpy as np

from copy import deepcopy

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from torch.utils.tensorboard import SummaryWriter

from flexloop.simple_loop import (
    training, log, load_loop_state, update_step, valid_step, rebatch_call, State)
from salad.modules.structure_autoencoder import StructureAutoencoder, StructureDecoder
from salad.modules.config import distance_to_structure_decoder as config_choices
from flexloop.utils import parse_options
from flexloop.loop import cast_float

def model_step(config, rebatch=1, is_training=True):
    if config.is_decoder:
        module = StructureDecoder
    else:
        module = StructureAutoencoder
    if not is_training:
        config = deepcopy(config)
        config.eval = True
    def step(data):
        data = jax.tree_util.tree_map(lambda x: jnp.array(x), data)
        data = cast_float(data, dtype=jnp.float32)
        loss, out = rebatch_call(module(config), rebatch=rebatch)(data)
        res_dict = {
            f"{name}_loss": item
            for name, item in out["losses"].items()
        }
        return cast_float(loss, jnp.float32), cast_float(res_dict, jnp.float32)
    return step

def make_training_inner(optimizer, step, data, accumulate=1, multigpu=True,
                        ema_weight=0.999, nanhunt=False, with_state=False):
    update = update_step(step, optimizer, accumulate=accumulate, multigpu=multigpu, nanhunt=False, with_state=with_state)
    def training_inner(loop_state: State):
        t = time.time()
        item = next(data)
        load_time = time.time() - t
        key, subkey = jax.random.split(loop_state.key)
        tr = time.time()
        _, loggables, params, opt_state = update(loop_state.params, loop_state.opt_state, subkey, item)
        step_time = time.time() - tr
        new_state = State(key, loop_state.step_id, params, opt_state, aux_state)
        checkpointables = dict(checkpoint=params)
        total_time = time.time() - t
        print(f"Step {loop_state.step_id}, load time {load_time:.3f} s, step time {step_time:.3f} s, loss {loggables['total']}, total {total_time:.3f} s")
        print(loggables)
        return new_state, loggables, checkpointables
    return training_inner

def make_valid_inner(step, data, multigpu=True, with_state=False):
    step = valid_step(step, multigpu=multigpu, with_state=with_state)
    def valid_inner(loop_state: State):
        t = time.time()
        item = next(data)
        params = loop_state.params
        res_dict = step(
            params, loop_state.key, item)
        print(f"Computed valid batch in {time.time() - t:.3f} seconds.")
        return res_dict
    return valid_inner

def cosine_decay_schedule(start_lr, decay_lr, warmup_steps, decay_steps):
    def schedule(count):
        result = jnp.where(
            count <= warmup_steps,
            count / warmup_steps * start_lr,
            (start_lr - decay_lr) * 0.5 * (1 + jnp.cos(jnp.pi * ((count - warmup_steps) % decay_steps) / decay_steps)) + decay_lr)
        result = jnp.maximum(result, 0.0)
        return result
    return schedule

def take_first_protein(batch):
    """Keep only the first protein (by batch_index) and zero-out the rest, preserving shapes."""
    if "batch_index" not in batch:
        return batch
    bidx = batch["batch_index"]
    keep = bidx == jnp.min(bidx)
    out = {}
    for k, v in batch.items():
        if isinstance(v, jnp.ndarray) and v.shape[0] == bidx.shape[0]:
            if v.dtype == jnp.bool_:
                out[k] = v & keep
            else:
                out[k] = jnp.where(keep.reshape((keep.shape[0],) + (1,) * (v.ndim - 1)), v, jnp.zeros_like(v))
        else:
            out[k] = v
    if "batch_index" in out:
        out["batch_index"] = jnp.zeros_like(out["batch_index"])
    if "mask" in out:
        out["mask"] = out["mask"] & keep
    if "seq_mask" in out and "mask" in out:
        out["seq_mask"] = out["mask"] & (out.get("aa_gt", 0) != 20)
    if "residue_mask" in out and "mask" in out and "all_atom_mask" in out:
        out["residue_mask"] = out["mask"] & jnp.any(out["all_atom_mask"], axis=-1)
    return out

def _merge_batches(items):
    out = {}
    offset = 0
    for item in items:
        if "batch_index" in item:
            bi = item["batch_index"]
            if isinstance(bi, jnp.ndarray):
                bi = np.array(bi)
            if isinstance(bi, np.ndarray) and bi.size:
                item = dict(item)
                item["batch_index"] = bi + offset
                offset += int(bi.max()) + 1
        for k, v in item.items():
            out.setdefault(k, []).append(v)
    merged = {}
    for k, vals in out.items():
        v0 = vals[0]
        if isinstance(v0, jnp.ndarray):
            merged[k] = jnp.concatenate(vals, axis=0)
        elif isinstance(v0, np.ndarray):
            merged[k] = np.concatenate(vals, axis=0)
        else:
            merged[k] = v0
    return merged

def accumulate_stream(stream, count: int):
    count = int(count)
    assert count >= 1
    while True:
        items = [next(stream) for _ in range(count)]
        yield _merge_batches(items)

if __name__ == "__main__":
    from salad.data.allpdb import BatchedProteinPDBStream
    from flexloop.data import BatchStream

    opt = parse_options(
        "train a distance-to-structure decoder on PDB.",
        path="network/",
        config="small_inner",
        data_path="",
        data_format="atom24",
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
        overfit_one="False",
        no_random="False",
        fixed_recycle=-1,
        deterministic="False",
        data_seed=42,
        num_workers=0,
        prefetch_factor=2,
        suffix="1"
    )
    seed = int(opt.jax_seed)
    random.seed(seed)
    np.random.seed(seed)
    if opt.data_format not in ("atom24", "atom37"):
        raise ValueError(f"Unsupported data_format={opt.data_format!r}")
    multigpu = opt.multigpu == "True"
    NUM_DEVICES = jax.device_count()
    if not multigpu:
        NUM_DEVICES = 1
    config = getattr(config_choices, opt.config)
    path = opt.path
    sae = "sae"
    if config.is_decoder:
        sae = "sdd"
    path = f"{path}/salad/{sae}-{opt.config}-{opt.num_aa}-{opt.suffix}"
    writer = SummaryWriter(path)

    print("Attempting to load dataset...")
    num_workers = int(opt.num_workers)
    prefetch = int(opt.prefetch_factor) if num_workers > 0 else None
    seed_train = int(opt.data_seed)
    seed_valid = seed_train + 1
    data = BatchedProteinPDBStream(f"{opt.data_path}/allpdb/",
                                   seqres_aa="clusterSeqresAA",
                                   cutoff_resolution=4.0,
                                   p_complex=opt.p_complex,
                                   format=opt.data_format,
                                   size=512, #1024,
                                   min_size=16,
                                   max_size=512, #1024,
                                   seed=seed_train)
    accumulate = opt.rebatch * opt.accumulate * NUM_DEVICES
    if num_workers == 0:
        data = iter(data)
        if accumulate != 1:
            data = accumulate_stream(data, accumulate)
    else:
        data = iter(BatchStream(data, num_workers=num_workers,
                                accumulate=accumulate,
                                prefetch_factor=prefetch))  
    valid_data = BatchedProteinPDBStream(f"{opt.data_path}/allpdb/",
                                         seqres_aa="clusterSeqresAA",
                                         cutoff_resolution=4.0,
                                         p_complex=opt.p_complex,
                                         format=opt.data_format,
                                         size=512, #1024,
                                         min_size=16,
                                         max_size=512, #1024,
                                         start_date="01/01/22",
                                         cutoff_date="12/31/23",
                                         seed=seed_valid)
    if num_workers == 0:
        valid_data = iter(valid_data)
        if accumulate != 1:
            valid_data = accumulate_stream(valid_data, accumulate)
    else:
        valid_data = iter(BatchStream(valid_data, num_workers=num_workers,
                                      accumulate=accumulate,
                                      prefetch_factor=prefetch))
    print("Dataset successfully loaded.")

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

    # overfit on a single batch to sanity-check loss convergence.
    if opt.overfit_one == "True":
        print("Overfitting on a single batch (reusing the first batch for train/valid).")
        first_item = take_first_protein(next(data))

        def repeat_item(item):
            while True:
                yield jax.tree_util.tree_map(lambda x: x.copy(), item)

        data = repeat_item(first_item)
        valid_data = repeat_item(first_item)
        item_0 = first_item
    else:
        item_0 = next(data)

    data = flag_stream(data, no_random=(opt.no_random == "True"), fixed_recycle=int(opt.fixed_recycle))
    valid_data = flag_stream(valid_data, no_random=(opt.no_random == "True"), fixed_recycle=int(opt.fixed_recycle))

    with_state = config.state is not None
    key = jax.random.PRNGKey(opt.jax_seed)
    transform_function = hk.transform_with_state if with_state else hk.transform
    transformed_single = transform_function(
        model_step(deepcopy(config), rebatch=1, is_training=True))
    if opt.multigpu == "True":
        config.multigpu = True
    init, step = transformed = transform_function(
        model_step(config, rebatch=opt.rebatch, is_training=True))
    _, valid = transform_function(
        model_step(config, rebatch=opt.rebatch, is_training=False))

    print("Initializing model parameters...")
    print("INPUT OF SHAPE:")
    for name, value in item_0.items():
        print("  ", name, value.shape)
    init_batch = jax.tree_util.tree_map(lambda x: x[:opt.rebatch * 100], item_0)
    tabulate_batch = jax.tree_util.tree_map(lambda x: x[:100], item_0)
    params = init(key, init_batch)
    print("Model parameters initialized.")

    print("Writing model description...")
    tabulated = hk.experimental.tabulate(transformed_single)(tabulate_batch)
    with open(f"{path}/model_description", "w") as f:
      f.write(tabulated)
    print("Model description written.")

    schedule = cosine_decay_schedule(
        start_lr=opt.lr, decay_lr=opt.decay_lr,
        warmup_steps=opt.warmup_steps, decay_steps=opt.decay_steps)
    total_steps = opt.warmup_steps + opt.decay_steps + 1

    print("Initializing optimizer state...")
    optimizer = optax.chain(
        # clip gradients by their norm
        optax.clip_by_global_norm(opt.clip),
        # scale gradients using Adam
        optax.scale_by_adam(opt.b1, opt.b2, eps=1e-9),
        # scale resulting learning rate by a cosine schedule
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0))
    if with_state:
        opt_state = optimizer.init(params[0])
    else:
        opt_state = optimizer.init(params)
    aux_state = {}
    print("Optimizer initialized.")

    print("Constructing training loop...")
    loop_state = State(key, 0, params, opt_state, aux_state)
    
    training_loop = training(
        path,
        make_training_inner(optimizer, step, data,
                            accumulate=opt.accumulate,
                            multigpu=multigpu,
                            ema_weight=opt.ema_weight,
                            with_state=with_state),
        valid_inner=make_valid_inner(valid, valid_data,
                                     multigpu=multigpu,
                                     with_state=with_state),
        max_steps=total_steps,
        valid_interval=100,
        logger=log())
    print("Recovering previous state, if available...")
    loop_state = load_loop_state(path) or loop_state
    print("Starting training...")
    print(f"Log files and tensorboard records will be written to {path}")
    training_loop(writer, loop_state)

# python -m salad.training.train_structure_autoencoder  --data_path /disk/2tb/edelkin/data --overfit_one True --p_complex 0.0

### full train det data
# python -m salad.training.train_structure_autoencoder  --data_path /disk/2tb/edelkin/data --data_seed 123 --num_workers 0 --prefetch_factor 2 --multigpu False
