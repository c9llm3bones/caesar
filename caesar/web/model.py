"""Torch model wrapper for the web interface."""
import os
import io
import numpy as np
import torch
from copy import deepcopy

from caesar.aflib.common.protein import from_structure_file, to_pdb, Protein
from caesar.aflib.common.residue_constants import restype_atom37_mask as RESTYPE_ATOM37_MASK
from caesar.modules.autoencoder import StructureAutoencoderInference, prepare_data
from caesar.modules.config import distance_to_structure_decoder as config_choices
from caesar.utils.all_atom_multimer import atom14_to_atom37
from caesar.scripts.import_salad_weights import import_salad_weights_

# ------------------------------------------------------------------ #
# Configuration — edit these or set env vars                          #
# ------------------------------------------------------------------ #
CHECKPOINT = os.environ.get("CAESAR_CHECKPOINT", "")
CONFIG_NAME = os.environ.get("CAESAR_CONFIG", "small_inner_atom37_main_sc")
DEVICE = torch.device(os.environ.get("CAESAR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
PAD_SIZE = int(os.environ.get("CAESAR_PAD_SIZE", "512"))

# ------------------------------------------------------------------ #
# Build and load model                                                #
# ------------------------------------------------------------------ #
print(f"[model] config={CONFIG_NAME}  device={DEVICE}  checkpoint={CHECKPOINT or '<none>'}")
_config = deepcopy(getattr(config_choices, CONFIG_NAME))
_config.eval = True
_config.num_recycle = int(os.environ.get("CAESAR_NUM_RECYCLE", "3"))
_config.compute_violation = os.environ.get("CAESAR_COMPUTE_VIOLATION", "False") == "True"
LATENT_DIM = int(getattr(_config, "latent_size", 320))

_model = StructureAutoencoderInference(_config).to(DEVICE)

# Warmup with a tiny batch so lazy modules are materialised.
def _warmup():
    N = 8
    dummy = {
        "aa_gt":              torch.zeros(N, dtype=torch.long, device=DEVICE),
        "residue_index":      torch.arange(N, device=DEVICE),
        "chain_index":        torch.zeros(N, dtype=torch.long, device=DEVICE),
        "batch_index":        torch.zeros(N, dtype=torch.long, device=DEVICE),
        "all_atom_positions": torch.zeros(N, 37, 3, device=DEVICE),
        "all_atom_mask":      torch.ones(N, 37, device=DEVICE),
        "seq_mask":           torch.ones(N, device=DEVICE),
        "residue_mask":       torch.ones(N, device=DEVICE),
    }
    with torch.no_grad():
        _model(dummy, generator=torch.Generator(device=DEVICE).manual_seed(0))

_warmup()

if CHECKPOINT:
    is_jax = CHECKPOINT.endswith(".jax")
    print(f"[model] Loading {'JAX' if is_jax else 'torch'} checkpoint…")
    if is_jax:
        import_salad_weights_(_model, CHECKPOINT, verbose=False, strict_missing=True, report=True)
    else:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        _model.load_state_dict(sd, strict=True)
        _model.to(DEVICE)
    _warmup()
    print("[model] Checkpoint loaded.")
else:
    print("[model] WARNING: no checkpoint set — using random weights.")

_model.eval()
_GEN = torch.Generator(device=DEVICE).manual_seed(42)


# ------------------------------------------------------------------ #
# Data helpers                                                        #
# ------------------------------------------------------------------ #

def _pad(arr: np.ndarray, target: int, fill=0) -> np.ndarray:
    delta = target - arr.shape[0]
    if delta <= 0:
        return arr
    return np.pad(arr, [(0, delta)] + [(0, 0)] * (arr.ndim - 1), constant_values=fill)


def _make_dummy_scaffold(length: int) -> np.ndarray:
    """Linear backbone in atom37 format (N/CA/C/CB/O filled)."""
    x = np.arange(length, dtype=np.float32) * 3.8
    pos = np.zeros((length, 37, 3), dtype=np.float32)
    pos[:, 1, 0] = x
    pos[:, 0, 0] = x - 1.45;  pos[:, 0, 1] = 0.60
    pos[:, 2, 0] = x + 1.52;  pos[:, 2, 1] = -0.50
    pos[:, 4, 0] = x + 2.10;  pos[:, 4, 1] = 0.35
    pos[:, 3, 0] = x + 2.20;  pos[:, 3, 1] = -1.30
    return pos


def _keep_monomer_chain(atom_pos, atom_mask, aatype, resi, chain):
    unique_chains = np.unique(chain)
    if unique_chains.size <= 1:
        return atom_pos, atom_mask, aatype, resi, chain

    ca_present = atom_mask[:, 1] > 0.5
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
    keep = chain == best_chain
    return atom_pos[keep], atom_mask[keep], aatype[keep], resi[keep], chain[keep]


def _to_torch(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            if t.is_floating_point():
                t = t.float()
            out[k] = t.to(DEVICE)
        else:
            out[k] = v
    return out


def load_pdb(path: str) -> tuple[dict, int, str]:
    """Load PDB/mmCIF → padded numpy data dict.
    Returns (data_dict, actual_length, original_pdb_string).
    """
    protein = from_structure_file(path)
    atom_pos  = np.asarray(protein.atom_positions, dtype=np.float32)
    atom_mask = np.asarray(protein.atom_mask,       dtype=np.float32)
    aatype    = np.asarray(protein.aatype,           dtype=np.int32)
    resi      = np.asarray(protein.residue_index,    dtype=np.int32)
    chain     = np.asarray(protein.chain_index,      dtype=np.int32)
    atom_pos, atom_mask, aatype, resi, chain = _keep_monomer_chain(
        atom_pos, atom_mask, aatype, resi, chain)
    L = aatype.shape[0]

    if L > PAD_SIZE:
        atom_pos, atom_mask, aatype, resi, chain = (
            a[:PAD_SIZE] for a in (atom_pos, atom_mask, aatype, resi, chain))
        L = PAD_SIZE

    support = np.zeros(PAD_SIZE, dtype=np.float32)
    support[:L] = 1.0

    data = dict(
        all_atom_positions=_pad(atom_pos,  PAD_SIZE).astype(np.float32),
        all_atom_mask     =_pad(atom_mask, PAD_SIZE).astype(np.float32),
        aa_gt             =_pad(aatype,    PAD_SIZE).astype(np.int32),
        residue_index     =_pad(resi,      PAD_SIZE).astype(np.int32),
        chain_index       =_pad(chain,     PAD_SIZE).astype(np.int32),
        batch_index       =np.zeros(PAD_SIZE, dtype=np.int32),
        seq_mask          =support,
        residue_mask      =_pad((atom_mask[:, 1] > 0.5).astype(np.float32), PAD_SIZE),
    )
    original = Protein(
        atom_positions=atom_pos[:L],
        aatype=aatype[:L],
        atom_mask=atom_mask[:L],
        residue_index=resi[:L],
        chain_index=chain[:L],
        b_factors=np.zeros((L, 37), dtype=np.float32),
    )
    return data, L, to_pdb(original)


def _make_stub(L: int) -> dict:
    support = np.zeros(PAD_SIZE, dtype=np.float32)
    support[:L] = 1.0
    scaffold = _make_dummy_scaffold(L)
    pos = np.zeros((PAD_SIZE, 37, 3), dtype=np.float32)
    pos[:L] = scaffold
    mask = np.zeros((PAD_SIZE, 37), dtype=np.float32)
    mask[:L] = 1.0
    return dict(
        all_atom_positions=pos,
        all_atom_mask     =mask,
        aa_gt             =np.zeros(PAD_SIZE, dtype=np.int32),
        residue_index     =np.arange(PAD_SIZE, dtype=np.int32),
        chain_index       =np.zeros(PAD_SIZE, dtype=np.int32),
        batch_index       =np.zeros(PAD_SIZE, dtype=np.int32),
        seq_mask          =support,
        residue_mask      =support,
    )


def _out_to_pdb(out: dict, L: int, resi=None, chain=None) -> str:
    aatype_pred = out["aatype"].cpu().numpy()[:L].astype(np.int32)
    atom_pos14  = out["atom_pos"].cpu().numpy()[:L]

    atom_pos14_t  = torch.from_numpy(atom_pos14).float()
    aatype_pred_t = torch.from_numpy(aatype_pred).long()
    atom_pos37, atom_mask37 = atom14_to_atom37(atom_pos14_t, aatype_pred_t)
    atom_pos37  = atom_pos37.numpy()
    atom_mask37 = atom_mask37.numpy()

    if resi  is None: resi  = np.arange(L, dtype=np.int32)
    if chain is None: chain = np.zeros(L, dtype=np.int32)

    lddt = out.get("lddt")
    if lddt is not None:
        b = np.stack([100.0 * lddt.cpu().numpy()[:L]] * 37, axis=-1).astype(np.float32)
    else:
        b = np.zeros((L, 37), dtype=np.float32)

    return to_pdb(Protein(
        atom_positions=atom_pos37,
        aatype=aatype_pred,
        atom_mask=atom_mask37,
        residue_index=resi,
        chain_index=chain,
        b_factors=b,
    ))


# ------------------------------------------------------------------ #
# Public API (same interface as JAX model.py)                        #
# ------------------------------------------------------------------ #

def reconstruct(data: dict, L: int, original_pdb: str) -> tuple[dict, str, str, np.ndarray]:
    with torch.no_grad():
        out = _model(_to_torch(data), generator=_GEN)

    mask = data["seq_mask"][:L]
    lddt_vals = out["lddt"].cpu().numpy()[:L]
    metrics = {
        "rmsd_ca":      float(out["rmsd_ca"]),
        "tm_score":     float(out["tm"]),
        "lddt":         float((lddt_vals * mask).sum() / max(mask.sum(), 1.0)),
        "recovery":     float(out["recovery"]),
        "num_residues": int(L),
    }

    pdb_recon = _out_to_pdb(
        out, L,
        resi =data["residue_index"][:L],
        chain=data["chain_index"][:L],
    )
    latent = out["latent"].cpu().numpy()[:L]
    return metrics, original_pdb, pdb_recon, latent


def encode(data: dict, L: int) -> np.ndarray:
    td = _to_torch(data)
    prepared = prepare_data(td, generator=_GEN)
    td.update(prepared)
    with torch.no_grad():
        latent = _model.encoder(td, generator=_GEN)
    return latent.cpu().numpy()[:L]


def decode(latent: np.ndarray) -> str:
    L = latent.shape[0]
    if L > PAD_SIZE:
        latent = latent[:PAD_SIZE]
        L = PAD_SIZE

    stub = _make_stub(L)
    td = _to_torch(stub)
    prepared = prepare_data(td, generator=_GEN)
    td.update(prepared)

    latent_full = np.zeros((PAD_SIZE, LATENT_DIM), dtype=np.float32)
    latent_full[:L] = latent
    td["latent"] = torch.from_numpy(latent_full).float().to(DEVICE)

    atom37_main = (
        getattr(_config, "atom37_parallel_mode", "none") == "parallel"
        and getattr(_config, "atom37_main_branch", False)
    )
    prev = _model.decoder.init_prev(td)
    for _ in range(_config.num_recycle):
        with torch.no_grad():
            r = _model.decoder(td, prev, generator=_GEN)
        if atom37_main:
            prev = {"pos37": r["pos37"].detach(), "local": r["local"].detach()}
        else:
            prev = {"pos": r["pos"].detach(), "local": r["local"].detach()}
    with torch.no_grad():
        result = _model.decoder(td, prev, generator=_GEN)

    aatype_pred = result["aa"].argmax(dim=-1).cpu().numpy()[:L].astype(np.int32)
    atom_pos14  = result["atom_pos"].cpu().numpy()[:L]
    atom_pos14_t  = torch.from_numpy(atom_pos14).float()
    aatype_pred_t = torch.from_numpy(aatype_pred).long()
    atom_pos37, atom_mask37 = atom14_to_atom37(atom_pos14_t, aatype_pred_t)
    return to_pdb(Protein(
        atom_positions=atom_pos37.numpy(),
        aatype=aatype_pred,
        atom_mask=atom_mask37.numpy(),
        residue_index=np.arange(L, dtype=np.int32),
        chain_index=np.zeros(L, dtype=np.int32),
        b_factors=np.zeros((L, 37), dtype=np.float32),
    ))


def interpolate(
    data1: dict, L1: int,
    data2: dict, L2: int,
    alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
) -> list[str]:
    z1 = encode(data1, L1)
    z2 = encode(data2, L2)
    L  = min(L1, L2)
    z1, z2 = z1[:L], z2[:L]
    return [decode((1.0 - a) * z1 + a * z2) for a in alphas]
