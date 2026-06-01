import os
import time
from copy import deepcopy

import numpy as np
import torch

from caesar.aflib.common.protein import to_pdb, from_pdb_string, Protein
from caesar.utils.all_atom_multimer import atom14_to_atom37
from caesar.modules.autoencoder import (
    StructureAutoencoderInference,
    StructureDecoderInference,
    prepare_data,
)
from caesar.modules.config import distance_to_structure_decoder as config_choices
from caesar.data.allpdb import slice_dict
from caesar.training.train_structure_autoencoder import to_torch_batch
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
    file_names = sorted(os.listdir(path))
    for name in file_names:
        if not name.endswith(".pdb"):
            continue
        with open(f"{path}/{name}", "rt") as f:
            protein = from_pdb_string(f.read())

        atom_pos_37 = np.asarray(protein.atom_positions, dtype=np.float32)
        atom_mask_37 = np.asarray(protein.atom_mask, dtype=np.float32)
        aatype = np.asarray(protein.aatype)
        resi = np.asarray(protein.residue_index)
        chain = np.asarray(protein.chain_index)

        unique_chains = np.unique(chain)
        if unique_chains.size > 1:
            ca_present = atom_mask_37[:, 1] > 0.5
            best_chain = None
            best_count = -1
            for chain_id in unique_chains:
                count = int(np.sum((chain == chain_id) & ca_present))
                if count > best_count:
                    best_count = count
                    best_chain = chain_id
            if best_chain is None or best_count <= 0:
                chain_lengths = [int(np.sum(chain == chain_id)) for chain_id in unique_chains]
                best_chain = unique_chains[int(np.argmax(chain_lengths))]
            chain_mask = chain == best_chain
            kept = int(np.sum(chain_mask))
            print(
                f"  [mono] {name}: {unique_chains.size} chains -> "
                f"keep chain {best_chain!r} ({kept} residues)",
                flush=True,
            )
            atom_pos_37 = atom_pos_37[chain_mask]
            atom_mask_37 = atom_mask_37[chain_mask]
            aatype = aatype[chain_mask]
            resi = resi[chain_mask]
            chain = chain[chain_mask]

        if aatype.shape[0] == 0:
            print(f"  [skip] {name}: empty chain after monomer filtering", flush=True)
            continue

        batch = np.zeros(resi.shape[0], dtype=np.int32)

        data = dict(
            aa_gt=aatype,
            residue_index=resi,
            chain_index=chain,
            batch_index=batch,
            all_atom_positions=atom_pos_37,
            all_atom_mask=atom_mask_37,
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


def prepare_eval_batch(data: dict, device: torch.device):
    return to_torch_batch(data, device=device)


if __name__ == "__main__":
    opt = parse_options(
        "sample from a protein diffusion model.",
        params="checkpoint.jax",
        out_path="outputs/",
        config="small_inner",
        path="/disk/10tb/home/kostikov/casp14.targets.T-dom.public_11.29.2020/",
        diagnostics="True",
        trace="False",
        no_random="False",
        time=0.0,
        num_recycle=4,
        jax_seed=42,
        save_imported_path="",
        compute_violation="False",
    )

    print(f"Running decoder with {int(opt.num_recycle) + 1} steps on files in {opt.path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = deepcopy(getattr(config_choices, opt.config))
    config.eval = True
    config.num_recycle = int(opt.num_recycle)
    config.compute_violation = (opt.compute_violation == "True")
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

    # Warmup before loading to materialize lazy / conditional submodules so
    # strict checkpoint loading sees the final module structure.
    first_item = next(parse_input_data(opt.path, size=1024), None)
    if first_item is None:
        raise FileNotFoundError(f"No .pdb files found in {opt.path}")
    _, warmup_data = first_item
    warmup = prepare_eval_batch(warmup_data, device=device)
    warmup["time"] = torch.tensor(float(opt.time), device=device, dtype=torch.float32)
    if opt.no_random == "True":
        warmup["no_random"] = torch.tensor(True, device=device)
    with torch.no_grad():
        _ = model(warmup, generator=torch.Generator(device=device).manual_seed(int(opt.jax_seed)))

    if is_jax:
        from caesar.scripts.import_salad_weights import import_salad_weights_
        import_salad_weights_(
            model,
            params_path,
            verbose=False,
            strict_missing=True,
            report=True,
            save_path=str(opt.save_imported_path or ""),
        )
    else:
        state_dict = _load_state_dict(params_path)
        model.load_state_dict(state_dict, strict=True)

    print("Model parameters loaded.")

    # warmup (after weights load) to ensure lazy params are set on device
    warmup = prepare_eval_batch(warmup_data, device=device)
    warmup["time"] = torch.tensor(float(opt.time), device=device, dtype=torch.float32)
    if opt.no_random == "True":
        warmup["no_random"] = torch.tensor(True, device=device)
    with torch.no_grad():
        _ = model(warmup, generator=torch.Generator(device=device).manual_seed(int(opt.jax_seed)))

    print("Start decoding...")
    os.makedirs(opt.out_path, exist_ok=True)
    os.makedirs(f"{opt.out_path}/diagnostics/", exist_ok=True)
    with open(f"{opt.out_path}/diagnostics/scores.csv", "wt") as f_scores:
        f_scores.write(
            "name,num_aa,recovery,perplexity,rmsd_ca,rmsd_full_atom,"
            "rmsd_full_atom37,rmsd_valid_atom37,rmsd_nonvalid_atom37,tm,lddt\n")
        key = torch.Generator(device=device).manual_seed(int(opt.jax_seed))
        for name, data in parse_input_data(opt.path, size=1024):
            data_t = prepare_eval_batch(data, device=device)
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
                f"{float(out['rmsd_ca'])},{float(out['rmsd_full_atom'])},"
                f"{float(out['rmsd_full_atom37'])},{float(out['rmsd_valid_atom37'])},"
                f"{float(out['rmsd_nonvalid_atom37'])},"
                f"{float(out['tm'])},{float(mean_lddt)}\n"
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
