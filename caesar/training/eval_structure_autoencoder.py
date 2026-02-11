import os
import time
from copy import deepcopy

import numpy as np
import torch

from caesar.aflib.common.protein import to_pdb, from_pdb_string, Protein
from caesar.utils.all_atom_multimer import atom14_to_atom37, atom37_to_atom14
from caesar.modules.autoencoder import (
    StructureAutoencoderInference,
    StructureDecoderInference,
    prepare_data,
)
from caesar.modules.config import distance_to_structure_decoder as config_choices
from caesar.data.allpdb import slice_dict
from caesar.training.train_structure_autoencoder import prepare_batch, slice_batch_first_dim
from flexloop.utils import parse_options


def _load_state_dict(path: str) -> dict:
    # torch>=2.6 defaults weights_only=True; for trusted checkpoints allow full load.
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def parse_input_data(path: str, size: int = 1024):
    file_names = os.listdir(path)
    for name in file_names:
        if not name.endswith(".pdb"):
            continue
        with open(f"{path}/{name}", "rt") as f:
            protein = from_pdb_string(f.read())

        atom_pos_37 = protein.atom_positions
        atom_mask_37 = protein.atom_mask
        aatype = protein.aatype
        resi = protein.residue_index
        chain = protein.chain_index
        batch = np.zeros_like(protein.residue_index)

        pos37_t = torch.as_tensor(atom_pos_37, dtype=torch.float32)
        mask37_t = torch.as_tensor(atom_mask_37, dtype=torch.float32)
        aa_t = torch.as_tensor(aatype, dtype=torch.long)

        atom_pos_14, atom_mask_14 = atom37_to_atom14(aa_t, pos37_t, mask37_t)
        atom_pos_14 = atom_pos_14.cpu().numpy()
        atom_mask_14 = atom_mask_14.cpu().numpy()

        data = dict(
            aa_gt=aatype,
            residue_index=resi,
            chain_index=chain,
            batch_index=batch,
            all_atom_positions=atom_pos_14,
            all_atom_mask=atom_mask_14,
            seq_mask=(aatype != 20),
            residue_mask=atom_mask_37[:, 1],
        )
        data = pad_to_size(data, size=size)
        yield name, data


def pad_to_size(data: dict, size: int):
    result = {}
    for key, item in data.items():
        if item.shape[0] < size:
            delta = size - item.shape[0]
            item = np.concatenate(
                (item, np.zeros([delta] + list(item.shape[1:]), dtype=item.dtype))
            )
        result[key] = item
    return result


if __name__ == "__main__":
    opt = parse_options(
        "sample from a protein diffusion model.",
        params="checkpoint.jax",
        out_path="outputs/",
        config="small_inner",
        path="inputs/",
        diagnostics="False",
        trace="False",
        no_random="False",
        time=0.0,
        num_recycle=4,
        jax_seed=42,
    )

    print(f"Running decoder with {int(opt.num_recycle) + 1} steps on files in {opt.path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = deepcopy(getattr(config_choices, opt.config))
    config.eval = True
    config.num_recycle = int(opt.num_recycle)
    if opt.no_random == "True":
        config.num_random_neighbours = 0
        config.fape_neighbours = 0

    if getattr(config, "is_decoder", False):
        model = StructureDecoderInference(
            config,
            prepare_data_fn=prepare_data,
            device=device,
            strict=True,
        )
    else:
        model = StructureAutoencoderInference(config).to(device)

    model.eval()

    print("Loading model parameters...")
    params_path = opt.params
    is_jax = str(params_path).endswith(".jax")

    if is_jax:
        # Warmup to materialize LazyLinear, then import salad weights.
        first_item = next(parse_input_data(opt.path, size=1024), None)
        if first_item is None:
            raise FileNotFoundError(f"No .pdb files found in {opt.path}")
        _, warmup_data = first_item
        warmup = prepare_batch(warmup_data, device=device, float_dtype=torch.float32)
        warmup["time"] = torch.tensor(float(opt.time), device=device, dtype=torch.float32)
        if opt.no_random == "True":
            warmup["no_random"] = torch.tensor(True, device=device)
        with torch.no_grad():
            _ = model(warmup, generator=torch.Generator(device=device).manual_seed(int(opt.jax_seed)))

        from caesar.scripts.import_salad_weights import import_salad_weights_
        import_salad_weights_(model, params_path, verbose=False, strict_missing=True, report=True)
    else:
        state_dict = _load_state_dict(params_path)
        model.load_state_dict(state_dict, strict=True)

    print("Model parameters loaded.")

    # warmup (after weights load) to ensure lazy params are set on device
    first_item = next(parse_input_data(opt.path, size=1024), None)
    if first_item is None:
        raise FileNotFoundError(f"No .pdb files found in {opt.path}")
    _, warmup_data = first_item
    warmup = prepare_batch(warmup_data, device=device, float_dtype=torch.float32)
    warmup["time"] = torch.tensor(float(opt.time), device=device, dtype=torch.float32)
    if opt.no_random == "True":
        warmup["no_random"] = torch.tensor(True, device=device)
    with torch.no_grad():
        _ = model(warmup, generator=torch.Generator(device=device).manual_seed(int(opt.jax_seed)))

    print("Start decoding...")
    os.makedirs(opt.out_path, exist_ok=True)
    os.makedirs(f"{opt.out_path}/diagnostics/", exist_ok=True)
    with open(f"{opt.out_path}/diagnostics/scores.csv", "wt") as f_scores:
        f_scores.write("name,num_aa,recovery,perplexity,rmsd_ca,tm,lddt\n")
        key = torch.Generator(device=device).manual_seed(int(opt.jax_seed))
        for name, data in parse_input_data(opt.path, size=1024):
            data_t = prepare_batch(data, device=device, float_dtype=torch.float32)
            data_t["time"] = torch.tensor(float(opt.time), device=device, dtype=torch.float32)
            if opt.no_random == "True":
                data_t["no_random"] = torch.tensor(True, device=device)

            key, subkey = key, torch.Generator(device=device).manual_seed(
                int(torch.randint(0, 2**31 - 1, (1,), generator=key, device=device).item())
            )

            mask = data_t["all_atom_mask"][:, 1] > 0

            with torch.no_grad():
                if opt.trace == "True":
                    out, trace = model(data_t, generator=subkey, return_trace=True)
                else:
                    out = model(data_t, generator=subkey)

            out = slice_dict(out, mask)
            atom37, atom37_mask = atom14_to_atom37(out["atom_pos"], out["aatype"])

            protein = Protein(
                np.array(atom37.detach().cpu().numpy()),
                np.array(out["aatype"].detach().cpu().numpy()),
                np.array(atom37_mask.detach().cpu().numpy()),
                np.array(data["residue_index"]),
                np.array(data["chain_index"]),
                np.stack([100 * np.array(out["lddt"].detach().cpu().numpy())] * 37, axis=-1),
            )
            pdb_string = to_pdb(protein)

            mean_lddt = out["lddt"].sum() / torch.clamp(
                data_t["all_atom_mask"][:, 1].sum(), min=1.0
            )
            f_scores.write(
                f"{name},{int(mask.to(torch.int32).sum())},"
                f"{float(out['recovery'])},{float(out['perplexity'])},"
                f"{float(out['rmsd_ca'])},{float(out['tm'])},{float(mean_lddt)}\n"
            )
            f_scores.flush()
            with open(f"{opt.out_path}/decoder_{name}", "wt") as f:
                f.write(pdb_string)

            if opt.diagnostics == "True":
                diagnostics = dict(
                    latent=out["latent"].detach().cpu().numpy(),
                    local=out["local"].detach().cpu().numpy(),
                )
                if "codebook_index" in out:
                    diagnostics["codebook_index"] = out["codebook_index"].detach().cpu().numpy()
                np.savez_compressed(
                    f"{opt.out_path}/diagnostics/decoder_{'.'.join(name.split('.')[:-1])}.npz",
                    **diagnostics,
                )
            if opt.trace == "True":
                trace_np = {k: v.detach().cpu().numpy() for k, v in trace.items()}
                np.savez_compressed(
                    f"{opt.out_path}/diagnostics/trace_{'.'.join(name.split('.')[:-1])}.npz",
                    **trace_np,
                )

    print("All proteins decoded.")
