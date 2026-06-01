"""Autoencoder combining Encoder and Decoder."""

import pickle
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Any, Callable, List, Optional, Dict, Tuple

import torch.nn.functional as F
from caesar.utils.geometry import Vec3Array 

from caesar.modules.basic import Linear
from caesar.modules.utils.geometry import (
    unique_chain, compute_pseudo_cb, positions_to_ncacocb,
    index_mean, index_align)
from caesar.utils.all_atom_multimer import atom37_to_atom14
from caesar.modules.decoder import _masked_rmsd_no_align
from caesar.modules.encoder import Encoder
from caesar.modules.decoder import Decoder
from caesar.modules.utils.dssp import assign_dssp
from caesar.utils.all_atom_multimer import get_atom14_mask
from caesar.utils.loss import violation_loss

def _stat(t: torch.Tensor):
    t = t.detach()
    return dict(mean=float(t.mean()), std=float(t.std()), max=float(t.abs().max()))


def _as_bool(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return False
        return bool(value.detach().bool().any().item())
    return bool(value)


def _assert_finite(name: str, tensor: torch.Tensor):
    if not torch.is_tensor(tensor):
        return
    finite = torch.isfinite(tensor)
    if bool(finite.all().item()):
        return
    bad_count = int((~finite).sum().item())
    finite_vals = tensor[finite]
    if finite_vals.numel() > 0:
        min_val = float(finite_vals.min().item())
        max_val = float(finite_vals.max().item())
    else:
        min_val = float("nan")
        max_val = float("nan")
    raise RuntimeError(
        f"Non-finite tensor detected in {name}: bad_count={bad_count}, "
        f"finite_min={min_val}, finite_max={max_val}"
    )

# utility for fixed recycle
def _as_int(value) -> Optional[int]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return int(value.detach().cpu().item())
    return int(value)

def _randint_count(generator: Optional[torch.Generator], device: torch.device) -> int:
    # Always sample on CPU to avoid GPU sync from .item() when generator is CUDA.
    if generator is None:
        return int(torch.randint(0, 4, (1,), device="cpu").item())
    if hasattr(generator, "device") and generator.device.type == "cuda":
        cpu_gen = getattr(_randint_count, "_cpu_gen", None)
        if cpu_gen is None:
            cpu_gen = torch.Generator(device="cpu").manual_seed(int(generator.initial_seed()))
            _randint_count._cpu_gen = cpu_gen
        return int(torch.randint(0, 4, (1,), device="cpu", generator=cpu_gen).item())
    return int(torch.randint(0, 4, (1,), device="cpu", generator=generator).item())

def _sample_by_batch(
    batch: torch.Tensor,
    *,
    tail_shape: tuple[int, ...] = (),
    uniform: bool = False,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample one random value per batch id and broadcast by `batch` via index_select."""
    if batch.dtype != torch.long:
        batch = batch.long()
    if batch.numel() == 0:
        return torch.empty((0, *tail_shape), device=device, dtype=dtype)
    num_items = int(batch.max().item()) + 1
    sample_shape = (num_items, *tail_shape)
    if uniform:
        table = torch.rand(sample_shape, device=device, dtype=dtype, generator=generator)
    else:
        table = torch.randn(sample_shape, device=device, dtype=dtype, generator=generator)
    return torch.index_select(table, dim=0, index=batch)


class StructureAutoencoder(nn.Module):
    """Wrapper class for protein structure autoencoder training"""
    def __init__(self, config, name: Optional[str] = "structure_autoencoder"):
        super().__init__()
        self.config = config
        c = self.config

        self.encoder = Encoder(c)
        self.decoder = Decoder(c)

        self.quantize = None
        if getattr(c, "codebook_size", 0):
            vq_cls = VQState if getattr(c, "state", False) else VQ
            # (c) mapped_axes is Haiku-specific; ignored here
            self.quantize = vq_cls(c.codebook_size, c.affine)
            
        # or FSQ (not used in the manuscript)
        self.fsq = FSQ() if getattr(c, "fsq", False) else None
        
    def forward(
        self,
        data: Dict[str, Any],
        *,
        generator: Optional[torch.Generator] = None,
        running_init: bool = False,
        return_trace: bool = False,
    ):
        c = self.config

        data = dict(data)
        detect_nonfinite = _as_bool(data.get("detect_nonfinite", False))
        trace: Dict[str, torch.Tensor] = {} if return_trace else {}

        # convert coordinates, center & add encoded features
        data.update(prepare_data(data, generator=generator, detect_nonfinite=detect_nonfinite))
        if return_trace:
            trace["prep/pos"] = data["pos"].detach()
            trace["prep/pos_gt"] = data["pos_gt"].detach()
            trace["prep/mask"] = data["mask"].detach()
            trace["prep/dmap"] = data["dmap"].detach()

        # optionally apply noise to the inputs
        if getattr(c, "input_diffusion", False):
            clean_latent = self.encoder(data, generator=generator)
            data["clean_latent"] = clean_latent
            data.update(self.prepare_input_diffusion(data, generator=generator))

        latent = self.encoder(data, generator=generator, trace=trace if return_trace else None)
        if return_trace:
            trace["enc/latent"] = latent.detach()

        # optionally apply noise to the latents (not used in the manuscript)
        if getattr(c, "latent_diffusion", False):
            latent = F.layer_norm(latent, (latent.shape[-1],), weight=None, bias=None)
            data["clean_latent"] = latent
            latent, time = self.prepare_latent_diffusion(latent, data, generator=generator)
            if not getattr(c, "vp_diffusion", False):
                denom = torch.clamp(1.0 + time[:, None] ** 2, min=1e-3)
                data["skip_latent"] = latent / denom
                latent = latent / torch.clamp(torch.sqrt(1.0 + time[:, None] ** 2), min=1e-3)
            data["time"] = time

        codebook_losses = None
        state_update = None
        if getattr(c, "codebook_size", 0):
            if getattr(c, "state", False):
                latent, codebook_index, codebook_losses, state_update = self.quantize(latent, data["mask"])
            else:
                latent, codebook_index, codebook_losses = self.quantize(latent, data["mask"])
            data["codebook_index"] = codebook_index

        if getattr(c, "fsq", False):
            latent, _ = self.fsq(latent)

        data["latent"] = latent

        # decoder recycling
        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )
        if atom37_main:
            prev = dict(
                pos37=data["pos37"],
                local=torch.zeros((data["pos37"].shape[0], c.local_size),
                                   device=latent.device, dtype=latent.dtype),
            )
        else:
            prev = dict(
                pos=data["pos"],
                local=torch.zeros((data["pos"].shape[0], c.local_size),
                                   device=latent.device, dtype=latent.dtype),
            )

        if not running_init:
            fixed_recycle = _as_int(data.get("fixed_recycle", None))
            if fixed_recycle is not None:
                count = int(fixed_recycle)
            elif getattr(c, "eval", False):
                count = int(c.num_recycle)
            else:
                count = _randint_count(generator, latent.device)

            for _ in range(count):
                result_i = self.decoder(data, prev, generator=generator)
                if atom37_main:
                    prev = {
                        "pos37": result_i["pos37"].detach(),
                        "local": result_i["local"].detach(),
                    }
                else:
                    prev = {
                        "pos": result_i["pos"].detach(),
                        "local": result_i["local"].detach(),
                    }

        result = self.decoder(data, prev, generator=generator)

        if return_trace:
            trace["dec/pos"] = result["pos"].detach()
            trace["dec/local"] = result["local"].detach()
            trace["dec/atom_pos"] = result["atom_pos"].detach()
            trace["dec/aa"] = result["aa"].detach()

        if getattr(c, "codebook_size", 0):
            result["codebook_losses"] = codebook_losses

        total, losses = self.decoder.loss(data, result, generator=generator)
        
        out_dict = dict(results=result, losses=losses)
        if getattr(c, "codebook_size", 0) and getattr(c, "state", False):
            out_dict["_state_update"] = state_update
        if return_trace:
            return total, out_dict, trace
        return total, out_dict
    
    def add_noise(self, latent: torch.Tensor, batch: torch.Tensor, *, generator: torch.Generator | None = None):
        """Add noise to latents. 
        
        Not used in the manuscript.
        """
        if batch.dtype != torch.long:
            batch = batch.long()

        noise_level = _sample_by_batch(
            batch,
            tail_shape=(),
            uniform=False,
            device=latent.device,
            dtype=latent.dtype,
            generator=generator,
        )
        noise_level = torch.exp(noise_level)

        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator) * noise_level
        return latent + noise

    def prepare_input_diffusion(self, data: dict, *, generator: torch.Generator | None = None):
        """When training as a structure diffusion model, prepare model input.
        
        Not used in the manuscript.
        """
        c = self.config

        batch = data["batch_index"]
        if batch.dtype != torch.long:
            batch = batch.long()
        pos = data["pos_input"]

        pos = pos - index_mean(pos[:, 1], batch, data["mask"][:, None])[:, None]  # FIXME (ported as-is)

        time = _sample_by_batch(
            batch,
            tail_shape=(),
            uniform=True,
            device=pos.device,
            dtype=pos.dtype,
            generator=generator,
        )
        if "time" in data:
            t = data["time"]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, device=pos.device, dtype=pos.dtype)
            else:
                needs_device = t.device != pos.device
                needs_dtype = t.dtype != pos.dtype
                if needs_device or needs_dtype:
                    t = t.to(
                        device=pos.device if needs_device else None,
                        dtype=pos.dtype if needs_dtype else None,
                    )
            if t.ndim == 0:
                t = t.expand_as(time)
            time = t

        s = 0.01
        denom = math.cos(s / (1.0 + s) * math.pi / 2.0)
        time = torch.cos((time + s) / (1.0 + s) * math.pi / 2.0) / denom
        time = torch.sqrt(torch.clamp(1.0 - time ** 2, 0.0, 1.0))

        noise = float(c.sigma_data) * torch.randn(pos.shape, device=pos.device, dtype=pos.dtype, generator=generator)
        pos = torch.sqrt(1.0 - time[:, None, None] ** 2) * pos + time[:, None, None] * noise

        return dict(pos_input=pos, time=time)

    def prepare_latent_diffusion(self, latent: torch.Tensor, data: dict, *, generator: torch.Generator | None = None):
        """When training as a latent diffusion model, prepare model input. 
        
        not used in the manuscript."""
        c = self.config

        batch = data["batch_index"]
        if batch.dtype != torch.long:
            batch = batch.long()
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)

        if bool(c.vp_diffusion):
            time = _sample_by_batch(
                batch,
                tail_shape=(),
                uniform=True,
                device=latent.device,
                dtype=latent.dtype,
                generator=generator,
            )
        else:
            time = _sample_by_batch(
                batch,
                tail_shape=(),
                uniform=False,
                device=latent.device,
                dtype=latent.dtype,
                generator=generator,
            )
            time = torch.exp(torch.tensor(1.0, device=latent.device, dtype=latent.dtype) + 1.2 * time)

        if "time" in data:
            t = data["time"]
            if not torch.is_tensor(t):
                t = torch.as_tensor(t, device=latent.device, dtype=latent.dtype)
            else:
                needs_device = t.device != latent.device
                needs_dtype = t.dtype != latent.dtype
                if needs_device or needs_dtype:
                    t = t.to(
                        device=latent.device if needs_device else None,
                        dtype=latent.dtype if needs_dtype else None,
                    )
            if t.ndim == 0:
                t = t.expand_as(time)
            time = t

        if bool(c.vp_diffusion):
            s = 0.01
            denom = math.cos(s / (1.0 + s) * math.pi / 2.0)
            time = torch.cos((time + s) / (1.0 + s) * math.pi / 2.0) / denom
            time = torch.sqrt(torch.clamp(1.0 - time ** 2, 0.0, 1.0))
            latent = torch.sqrt(1.0 - time[:, None] ** 2) * latent + time[:, None] * noise * 5.0
        else:
            latent = latent + time[:, None] * noise

        return latent, time

def prepare_data(data, generator=None, detect_nonfinite: bool = False):
    """Prepare model inputs from a batch of data."""
    pos = data["all_atom_positions"]   # [N, M, 3]
    atom_mask = data["all_atom_mask"]  # [N, M]
    chain = data["chain_index"]        # [N]
    batch = data["batch_index"]        # [N]

    if not torch.is_tensor(pos):
        pos = torch.as_tensor(pos)

    device = pos.device
    dtype = pos.dtype if pos.dtype.is_floating_point else torch.float32
    if pos.dtype != dtype:
        pos = pos.to(dtype=dtype)

    if not torch.is_tensor(atom_mask):
        atom_mask = torch.as_tensor(atom_mask, device=device)
    elif atom_mask.device != device:
        raise ValueError(f"all_atom_mask is on {atom_mask.device}, expected {device}")

    if not torch.is_tensor(chain):
        chain = torch.as_tensor(chain, device=device)
    elif chain.device != device:
        raise ValueError(f"chain_index is on {chain.device}, expected {device}")

    if not torch.is_tensor(batch):
        batch = torch.as_tensor(batch, device=device)
    elif batch.device != device:
        raise ValueError(f"batch_index is on {batch.device}, expected {device}")

    if batch.dtype != torch.long:
        batch = batch.long()
    no_random = data.get("no_random", False)

    # handle atom37 input vs atom14 input
    atom_pos_37 = None
    atom_mask_37 = None
    if pos.shape[-2] == 37:
        atom_pos_37 = pos
        atom_mask_37 = atom_mask
        # convert to atom14 for backbone processing
        aa_gt_t = data.get("aa_gt")
        if aa_gt_t is None:
            raise ValueError("aa_gt required when all_atom_positions has 37 atoms")
        if not torch.is_tensor(aa_gt_t):
            aa_gt_t = torch.as_tensor(aa_gt_t, device=device, dtype=torch.long)
        elif aa_gt_t.dtype != torch.long:
            aa_gt_t = aa_gt_t.long()
        pos, atom_mask = atom37_to_atom14(aa_gt_t, pos.to(dtype=dtype), atom_mask.to(dtype=dtype))
    else:
        # recast positions into atom14: truncate to atom14 format
        pos = pos[:, :14]
        atom_mask = atom_mask[:, :14]

    # boolean atom mask for logic
    atom_mask_bool = atom_mask.bool()

    # uniquify chain IDs across batches:
    if chain.dtype != torch.long:
        chain = chain.long()
    chain = unique_chain(chain, batch)

    # mask = seq_mask * residue_mask * atom_mask[:,:3].all(axis=-1)
    seq_mask = data["seq_mask"]
    if not torch.is_tensor(seq_mask):
        seq_mask = torch.as_tensor(seq_mask, device=device, dtype=dtype)
    else:
        if seq_mask.device != device:
            raise ValueError(f"seq_mask is on {seq_mask.device}, expected {device}")
        if seq_mask.dtype != dtype:
            seq_mask = seq_mask.to(dtype=dtype)

    residue_mask = data["residue_mask"]
    if not torch.is_tensor(residue_mask):
        residue_mask = torch.as_tensor(residue_mask, device=device, dtype=dtype)
    else:
        if residue_mask.device != device:
            raise ValueError(f"residue_mask is on {residue_mask.device}, expected {device}")
        if residue_mask.dtype != dtype:
            residue_mask = residue_mask.to(dtype=dtype)
    mask_bool = seq_mask.bool() & residue_mask.bool() & atom_mask_bool[:, :3].all(dim=-1)
    mask = mask_bool.to(dtype)

    # subtract the center from all positions
    center = index_mean(pos[:, 1], batch, atom_mask[:, 1, None])
    pos = pos - center[:, None, :]

    # set the positions of all masked atoms to the pseudo Cb position
    pseudo_cb = compute_pseudo_cb(pos)  # [N, 3]
    pos = torch.where(
        atom_mask_bool[..., None],
        pos,
        pseudo_cb[:, None, :].expand_as(pos),
    )

    pos_14 = pos
    atom_mask_14 = atom_mask_bool

    # get ncacocb positions (GT)
    pos_ncacocb = positions_to_ncacocb(pos)  

    cb = Vec3Array.from_array(pos_ncacocb[:, -1, :])    
    noise_scale = 0.0 if no_random else 0.3
    noise = noise_scale * torch.randn(list(cb.shape) + [3], device=pos_ncacocb.device, dtype=pos_ncacocb.dtype, generator=generator)
    cb = cb + Vec3Array.from_array(noise)                
    dmap = (cb[:, None] - cb[None, :]).norm()            

    dmap_mask = batch[:, None] == batch[None, :]

    # set all-atom-position target (GT targets)
    atom_pos = pos_14
    atom_mask_out = atom_mask_14.to(dtype)

    # assign dssp
    # dssp, _, _ = assign_dssp(atom_pos, batch, mask)

    # set initial backbone positions (decoder init)
    if no_random:
        pos_init = pos_ncacocb
    else:
        pos_init = torch.randn(pos_ncacocb.shape, device=pos_ncacocb.device, dtype=pos_ncacocb.dtype, generator=generator) 

    out_data = dict(
        pos=pos_init,
        pos_gt=pos_ncacocb,
        pos_input=pos_ncacocb,
        # dssp=dssp,
        dmap=dmap,
        dmap_mask=dmap_mask,
        chain_index=chain,
        mask_bool=mask_bool,
        mask=mask,
        atom_pos=atom_pos,
        atom_mask=atom_mask_out,
        all_atom_positions=pos_14,
        all_atom_mask=atom_mask_out,
    )

    if atom_pos_37 is not None:
        # center and fill missing positions with pseudo-CB
        atom_pos_37_c = (atom_pos_37 - center[:, None, :]).to(dtype=dtype)
        atom_mask_37_f = atom_mask_37.to(dtype=dtype)
        pseudo_cb_37 = compute_pseudo_cb(atom_pos_37_c)
        atom_pos_37_c = torch.where(
            atom_mask_37.bool()[..., None],
            atom_pos_37_c,
            pseudo_cb_37[:, None, :].expand_as(atom_pos_37_c))
        pos37_gt = atom_pos_37_c
        pos37_init = torch.randn(
            pos37_gt.shape, device=device, dtype=dtype, generator=generator)
        if no_random:
            pos37_init = pos37_gt
        out_data.update(
            all_atom_positions_37=atom_pos_37_c,
            all_atom_mask_37=atom_mask_37_f,
            physical_all_atom_mask_37=atom_mask_37_f,
            pos37=pos37_init,
            pos37_gt=pos37_gt,
            pos37_input=pos37_gt,
        )

    if detect_nonfinite:
        _assert_finite("prepare_data/pos_gt", out_data["pos_gt"])
        _assert_finite("prepare_data/pos", out_data["pos"])
        _assert_finite("prepare_data/dmap", out_data["dmap"])
        _assert_finite("prepare_data/mask", out_data["mask"])

    return out_data 
        
class StructureDecoderInference(nn.Module):
    """Wrapper class for StructureDecoder evaluation."""

    def __init__(
        self,
        config,
        prepare_data_fn,  # external function
        *,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ):
        super().__init__()
        self.config = config
        self.prepare_data_fn = prepare_data_fn
        c = self.config

        # fixed encoder apply fn (returns (latent, codebook_index))
        self.encoder_apply = assign_state(
            c,
            c.param_path,
            prepare_data_fn=prepare_data_fn,
            device=device,
            strict=strict,
        )

        # learnable decoder 
        self.decoder = Decoder(c)
        if device is not None:
            self.decoder = self.decoder.to(device)

    @torch.no_grad()
    def forward(self, data: Dict[str, Any], *, generator: Optional[torch.Generator] = None, return_trace: bool = False) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        c = self.config
        raw_data = dict(data)
        data = dict(raw_data)
        trace: Dict[str, torch.Tensor] = {} if return_trace else {}
        if generator is None:
            data.update(self.prepare_data_fn(raw_data))
        else:
            data.update(self.prepare_data_fn(raw_data, generator=generator))
        if return_trace:
            trace["prep/pos"] = data["pos"].detach()
            trace["prep/pos_gt"] = data["pos_gt"].detach()
            trace["prep/mask"] = data["mask"].detach()
            trace["prep/dmap"] = data["dmap"].detach()

        latent, codebook_index = self.encoder_apply(raw_data, generator=generator)
        latent = latent.detach()
        data["latent"] = latent
        if return_trace:
            trace["enc/latent"] = latent.detach()

        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )
        prev = self.decoder.init_prev(data)

        count = int(c.num_recycle)
        for _ in range(count):
            result_i = self.decoder(data, prev, generator=generator)
            if atom37_main:
                prev = {"pos37": result_i["pos37"].detach(), "local": result_i["local"].detach()}
            else:
                prev = {"pos": result_i["pos"].detach(), "local": result_i["local"].detach()}

        result = self.decoder(data, prev, generator=generator)
        if return_trace:
            trace["dec/pos"] = result["pos"].detach()
            trace["dec/local"] = result["local"].detach()
            trace["dec/atom_pos"] = result["atom_pos"].detach()
            trace["dec/aa"] = result["aa"].detach()

        mask = data.get("mask_bool", data["mask"])
        if mask.dtype != torch.bool:
            mask = mask.bool()
        aa_gt = data["aa_gt"]
        if aa_gt.dtype != torch.long:
            aa_gt = aa_gt.long()

        aa_pred = result["aa"]  
        aa_onehot = F.one_hot(aa_gt.clamp(0, 19), 20).to(dtype=aa_pred.dtype)

        aa_nll = -(aa_pred * aa_onehot).sum(dim=-1)
        aa_nll = torch.where(mask, aa_nll, torch.zeros_like(aa_nll))
        mask_f = mask.to(dtype=aa_nll.dtype)
        aa_nll = aa_nll.sum() / torch.clamp(mask_f.sum().to(aa_nll.dtype), min=1.0)

        perplexity = torch.exp(aa_nll)
        aatype = torch.argmax(aa_pred, dim=-1)
        recovery = ((aa_gt == aatype) & mask).to(dtype=aa_nll.dtype).sum() / torch.clamp(mask_f.sum(), min=1.0)

        pos_gt = data["pos_gt"]
        pos = result["pos"]

        pos_gt = index_align(pos_gt, pos, data["batch_index"], mask)

        di2 = ((pos[:, 1] - pos_gt[:, 1]) ** 2).sum(dim=-1)
        rmsd_ca = torch.sqrt((di2 * mask_f.to(di2.dtype)).sum() / torch.clamp(mask_f.sum().to(di2.dtype), min=1.0))

        L = torch.clamp(mask_f.sum(), min=1.0)
        d02 = (1.24 * torch.pow(torch.clamp(L - 15.0, min=0.0), 1.0 / 3.0) - 1.8) ** 2
        inner = (1.0 / (1.0 + di2 / torch.clamp(d02, min=1e-8))) * mask_f.to(di2.dtype)
        tm = inner.sum() / torch.clamp(mask_f.sum().to(inner.dtype), min=1.0)

        dca_gt = torch.linalg.norm(pos_gt[:, None, 1] - pos_gt[None, :, 1], dim=-1)
        dca = torch.linalg.norm(pos[:, None, 1] - pos[None, :, 1], dim=-1)

        pair_mask = (mask[:, None] & mask[None, :])
        derr = (dca_gt - dca).abs() * pair_mask.to(dtype=dca.dtype)

        threshold = torch.tensor([0.5, 1.0, 2.0, 4.0], device=derr.device, dtype=derr.dtype)
        Rinc = 15.0
        pair_mask = pair_mask & (dca_gt < Rinc)

        in_threshold = (derr[..., None] < threshold) & pair_mask[..., None]
        denom = torch.clamp(pair_mask[..., None].sum(dim=1).to(dtype=aa_nll.dtype), min=1.0)
        lddt_ca = (in_threshold.sum(dim=1).to(dtype=aa_nll.dtype) / denom).mean(dim=-1)
        lddt_ca = torch.where(mask, lddt_ca, torch.zeros_like(lddt_ca))

        out = dict(
            atom_pos=result["atom_pos"],
            aatype=aatype,
            latent=data["latent"],
            local=result["local"],
            perplexity=perplexity,
            recovery=recovery,
            rmsd_ca=rmsd_ca,
            tm=tm,
            lddt=lddt_ca,
            # dssp=data["dssp"],
        )
        if getattr(c, "codebook_size", 0):
            out["codebook_index"] = codebook_index

        if return_trace:
            return out, trace
        return out

class AssignState(nn.Module):
    """Fixed encoder wrapper."""

    def __init__(
        self,
        config,
        prepare_data_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    ):
        super().__init__()
        self.config = config
        self.prepare_data_fn = prepare_data_fn

        c = self.config
        self.encoder = Encoder(c)

        self.quantize = None
        if getattr(c, "codebook_size", 0):
            vq_cls = VQState if getattr(c, "state", False) else VQ
            self.quantize = vq_cls(c.codebook_size)

    def forward(
        self,
        data: Dict[str, Any],
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        c = self.config

        data = dict(data)
        data.update(self.prepare_data_fn(data) if generator is None else self.prepare_data_fn(data, generator=generator))

        latent = self.encoder(data, generator=generator)

        if getattr(c, "codebook_size", 0):
            latent, codebook_index, _ = self.quantize(latent, data["mask"])
            return latent, codebook_index

        return latent, None

def assign_state(
    config,
    param_path: str,
    prepare_data_fn: Callable[..., Dict[str, Any]],
    *,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Callable[..., Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Construct a fixed encoder from config and a parameter path.
    """
    model = AssignState(config, prepare_data_fn=prepare_data_fn)

    state_obj = None
    try:
        state_obj = torch.load(param_path, map_location="cpu")
    except Exception:
        with open(param_path, "rb") as f:
            state_obj = pickle.load(f)

    if isinstance(state_obj, dict) and "state_dict" in state_obj and isinstance(state_obj["state_dict"], dict):
        state_dict = state_obj["state_dict"]
    elif isinstance(state_obj, dict):
        state_dict = state_obj
    else:
        raise TypeError(
            "Checkpoint is not a dict/state_dict. "
            "If this is a Haiku params pickle, you need a JAX->Torch converter."
        )

    model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model = model.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def inner(data: Dict[str, Any], *, generator: Optional[torch.Generator] = None):
        return model(data, generator=generator)

    return inner

class StructureDecoder(StructureAutoencoder):
    """Wrapper class for training a structure autoencoder with a fixed encoder.
    
    Not used in the manuscript.
    """

    def __init__(
        self,
        config,
        prepare_data_fn: Callable[..., Dict[str, Any]],
        *,
        device: Optional[torch.device] = None,
        strict: bool = True,
        name: Optional[str] = "structure_decoder",
    ):
        super().__init__(config=config)
        self.config = config
        self.prepare_data_fn = prepare_data_fn

        self.encoder_apply = assign_state(
            config,
            config.param_path,
            prepare_data_fn=prepare_data_fn,
            device=device,
            strict=strict,
        )

        self.decoder = Decoder(config)
        if device is not None:
            self.decoder = self.decoder.to(device)

    def forward(
        self,
        data: Dict[str, Any],
        *,
        generator: Optional[torch.Generator] = None,
        running_init: bool = False,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]] | Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
        c = self.config
        data = dict(data)
        trace: Dict[str, torch.Tensor] = {} if return_trace else {}

        data.update(self.prepare_data_fn(data) if generator is None else self.prepare_data_fn(data, generator=generator))
        if return_trace:
            trace["prep/pos"] = data["pos"].detach()
            trace["prep/pos_gt"] = data["pos_gt"].detach()
            trace["prep/mask"] = data["mask"].detach()
            trace["prep/dmap"] = data["dmap"].detach()

        latent, codebook_index = self.encoder_apply(data, generator=generator)

        latent = latent.detach()
        data["latent"] = latent
        if return_trace:
            trace["enc/latent"] = latent.detach()

        prev = dict(
            pos=data["pos"],
            local=torch.zeros(
                (data["pos"].shape[0], c.local_size),
                device=data["pos"].device,
                dtype=latent.dtype,
            ),
        )

        if not running_init:
            if getattr(c, "eval", False):
                count = int(c.num_recycle)
            else:
                count = _randint_count(generator, latent.device)

            for _ in range(count):
                result_i = self.decoder(data, prev, generator=generator)
                prev = dict(
                    pos=result_i["pos"].detach(),
                    local=result_i["local"].detach(),
                )

        result = self.decoder(data, prev, generator=generator)
        if return_trace:
            trace["dec/pos"] = result["pos"].detach()
            trace["dec/local"] = result["local"].detach()
            trace["dec/atom_pos"] = result["atom_pos"].detach()
            trace["dec/aa"] = result["aa"].detach()

        total, losses = self.decoder.loss(data, result, generator=generator)

        out_dict = dict(
            results=result,
            losses=losses,
        )
        if return_trace:
            return total, out_dict, trace
        return total, out_dict

    def add_noise(
        self,
        latent: torch.Tensor,
        batch: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Add noise to latents."""
        if batch.dtype != torch.long:
            batch = batch.long()

        if generator is None:
            noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype)
        else:
            noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)

        noise_level = _sample_by_batch(
            batch,
            tail_shape=(),
            uniform=False,
            device=latent.device,
            dtype=latent.dtype,
            generator=generator,
        )
        noise_level = torch.exp(noise_level)

        return latent + noise * noise_level[..., None]

class StructureAutoencoderInference(StructureAutoencoder):
    """Wrapper class for autoencoder evaluation."""

    def __init__(self, config, name: Optional[str] = "structure_autoencoder_inference"):
        super().__init__(config)
        self.config = config
        c = self.config

        self.encoder = Encoder(c)
        self.decoder = Decoder(c)

        self.quantize = None
        if getattr(c, "codebook_size", 0):
            vq_cls = VQState if getattr(c, "state", False) else VQ
            self.quantize = vq_cls(c.codebook_size)

    @torch.no_grad()
    def forward(
        self,
        data: Dict[str, Any],
        *,
        generator: Optional[torch.Generator] = None,
        return_trace: bool = False,
    ) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        c = self.config

        data = dict(data)
        data.update(prepare_data(data, generator=generator))
        trace: Dict[str, torch.Tensor] = {}
        if return_trace:
            trace["prep/pos"] = data["pos"].detach()
            trace["prep/pos_gt"] = data["pos_gt"].detach()
            trace["prep/mask"] = data["mask"].detach()
            trace["prep/dmap"] = data["dmap"].detach()

        # optionally apply noise to inputs
        if getattr(c, "input_diffusion", False):
            clean_latent = self.encoder(data, generator=generator, trace=trace if return_trace else None)
            data["clean_latent"] = clean_latent
            data.update(self.prepare_input_diffusion(data, generator=generator))

        latent = self.encoder(data, generator=generator, trace=trace if return_trace else None)
        if return_trace:
            trace["enc/latent"] = latent.detach()

        # optionally apply noise to latents
        if getattr(c, "latent_diffusion", False):
            if "latent" in data:
                latent = data["latent"]
            data["clean_latent"] = latent
            latent, time = self.prepare_latent_diffusion(latent, data, generator=generator)

            if not getattr(c, "vp_diffusion", False):
                denom = torch.clamp(1.0 + time[:, None] ** 2, min=1e-3)
                data["skip_latent"] = latent / denom
                latent = latent / torch.clamp(torch.sqrt(1.0 + time[:, None] ** 2), min=1e-3)

            data["time"] = time
        else:
            data.setdefault("time", torch.zeros((data["mask"].shape[0],), device=latent.device, dtype=latent.dtype))

        codebook_index = None
        if getattr(c, "codebook_size", 0):
            latent, codebook_index, _ = self.quantize(latent, data["mask"])

        data["latent"] = latent

        atom37_main = (
            getattr(c, 'atom37_parallel_mode', 'none') == 'parallel'
            and getattr(c, 'atom37_main_branch', False)
        )
        prev = self.decoder.init_prev(data)

        count = int(c.num_recycle)
        for _ in range(count):
            result_i = self.decoder(data, prev, generator=generator)
            if atom37_main:
                prev = {"pos37": result_i["pos37"].detach(), "local": result_i["local"].detach()}
            else:
                prev = {"pos": result_i["pos"].detach(), "local": result_i["local"].detach()}

        result = self.decoder(data, prev, generator=generator)
        if return_trace:
            trace["dec/pos"] = result["pos"].detach()
            trace["dec/local"] = result["local"].detach()
            trace["dec/atom_pos"] = result["atom_pos"].detach()
            trace["dec/aa"] = result["aa"].detach()

        mask = data.get("mask_bool", data["mask"])
        if mask.dtype != torch.bool:
            mask = mask.bool()

        aa_logits_or_logprobs = result["aa"]
        aa_gt = data["aa_gt"]
        if aa_gt.dtype != torch.long:
            aa_gt = aa_gt.long()
        aa_onehot = F.one_hot(aa_gt.clamp(0, 19), 20).to(dtype=aa_logits_or_logprobs.dtype)
        aa_nll = -(aa_logits_or_logprobs * aa_onehot).sum(dim=-1)
        aa_nll = torch.where(mask, aa_nll, torch.zeros_like(aa_nll))
        mask_f = mask.to(dtype=aa_nll.dtype)
        aa_nll = aa_nll.sum() / torch.clamp(mask_f.sum().to(aa_nll.dtype), min=1.0)
        perplexity = torch.exp(aa_nll)

        aatype = torch.argmax(aa_logits_or_logprobs, dim=-1)
        recovery = ((aatype == aa_gt) & mask).to(dtype=aa_nll.dtype).sum() / torch.clamp(mask_f.sum(), min=1.0)

        # RMSD (CA)
        pos_gt = data["pos_gt"]
        pos = result["pos"]

        pos_gt_aligned = index_align(pos_gt, pos, data["batch_index"], mask)

        di2 = ((pos[:, 1] - pos_gt_aligned[:, 1]) ** 2).sum(dim=-1)
        rmsd_ca = torch.sqrt((di2 * mask_f.to(di2.dtype)).sum() / torch.clamp(mask_f.sum().to(di2.dtype), min=1.0))

        # TM score (CA)
        L = torch.clamp(mask_f.sum(), min=1.0)
        d02 = (1.24 * torch.pow(torch.clamp(L - 15.0, min=0.0), 1.0 / 3.0) - 1.8) ** 2
        inner = (1.0 / (1.0 + di2 / torch.clamp(d02, min=1e-8))) * mask_f.to(di2.dtype)
        tm = inner.sum() / torch.clamp(mask_f.sum().to(inner.dtype), min=1.0)

        # LDDT (CA)
        dca_gt = torch.linalg.norm(pos_gt_aligned[:, None, 1] - pos_gt_aligned[None, :, 1], dim=-1)
        dca = torch.linalg.norm(pos[:, None, 1] - pos[None, :, 1], dim=-1)
        pair_mask = (mask[:, None] & mask[None, :])

        derr = (dca_gt - dca).abs() * pair_mask.to(dtype=dca.dtype)

        threshold = torch.tensor([0.5, 1.0, 2.0, 4.0], device=dca.device, dtype=dca.dtype)
        Rinc = 15.0
        pair_mask = pair_mask & (dca_gt < Rinc)

        in_threshold = (derr[..., None] < threshold) & pair_mask[..., None]
        denom = torch.clamp(pair_mask[..., None].sum(dim=1).to(dtype=aa_nll.dtype), min=1.0)
        lddt_ca = (in_threshold.sum(dim=1).to(dtype=aa_nll.dtype) / denom).mean(dim=-1)
        lddt_ca = torch.where(mask, lddt_ca, torch.zeros_like(lddt_ca))

        atom_pos_gt = data["atom_pos"]
        atom_pos_pred = result["atom_pos"]
        if getattr(c, "is_rmsd_alinged", True):
            atom_pos_gt_eval = index_align(atom_pos_gt, atom_pos_pred, data["batch_index"], mask)
        else:
            atom_pos_gt_eval = atom_pos_gt
        rmsd_full_atom = _masked_rmsd_no_align(
            atom_pos_pred, atom_pos_gt_eval, data["atom_mask"], residue_mask=mask)

        rmsd_full_atom37 = rmsd_full_atom
        rmsd_valid_atom37 = rmsd_full_atom
        rmsd_nonvalid_atom37 = torch.zeros((), device=mask_f.device, dtype=mask_f.dtype)
        if "pos37" in result and "pos37_gt" in data:
            pos37_gt = data["pos37_gt"]
            pos37_pred = result["pos37"]
            if getattr(c, "is_rmsd_alinged", True):
                pos37_gt_eval = index_align(pos37_gt, pos37_pred, data["batch_index"], mask)
            else:
                pos37_gt_eval = pos37_gt
            physical_atom37_m = data.get("physical_all_atom_mask_37", data["all_atom_mask_37"])
            metric_atom37_m = data.get("metric_all_atom_mask_37", physical_atom37_m)
            rmsd_full_atom37 = _masked_rmsd_no_align(
                pos37_pred, pos37_gt_eval, metric_atom37_m, residue_mask=mask)
            rmsd_valid_atom37 = _masked_rmsd_no_align(
                pos37_pred, pos37_gt_eval, physical_atom37_m, residue_mask=mask)
            nonvalid_atom37_m = (
                1.0 - physical_atom37_m.to(dtype=pos37_pred.dtype)
            ) * mask[:, None].to(dtype=pos37_pred.dtype)
            rmsd_nonvalid_atom37 = _masked_rmsd_no_align(
                pos37_pred, pos37_gt_eval, nonvalid_atom37_m, residue_mask=mask)

        if getattr(c, "compute_violation", True):
            # AlphaFold-style violation loss (JAX-salad compatible API)
            res_mask = mask_f
            pred_mask = get_atom14_mask(aatype).to(device=res_mask.device, dtype=res_mask.dtype) * res_mask[:, None]
            violation_error, _ = violation_loss(
                aatype,
                data["residue_index"],
                result["atom_pos"],
                pred_mask,
                res_mask,
                clash_overlap_tolerance=1.5,
                violation_tolerance_factor=2.0,
                chain_index=data["chain_index"],
                batch_index=data["batch_index"],
                per_residue=False,
            )
        else:
            violation_error = torch.zeros((), device=mask_f.device, dtype=mask_f.dtype)

        latent_out = data["latent"]
        if "predicted_latent" in result:
            latent_out = result["predicted_latent"]

        out = dict(
            atom_pos=result["atom_pos"],
            aatype=aatype,
            latent=latent_out,
            local=result["local"],
            perplexity=perplexity,
            recovery=recovery,
            rmsd_ca=rmsd_ca,
            rmsd_full_atom=rmsd_full_atom,
            rmsd_full_atom37=rmsd_full_atom37,
            rmsd_valid_atom37=rmsd_valid_atom37,
            rmsd_nonvalid_atom37=rmsd_nonvalid_atom37,
            tm=tm,
            lddt=lddt_ca,
            time=data["time"],
            # dssp=data["dssp"],
            violation=violation_error,
        )
        if getattr(c, "codebook_size", 0):
            out["codebook_index"] = codebook_index

        if return_trace:
            return out, trace
        return out


class VQ(nn.Module):
    """Vector quantization module."""

    def __init__(
        self,
        codebook_size: int = 4096,
        affine=None,
        mapped_axes: Optional[List[str]] = None,  
        name: Optional[str] = "vq",
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.affine = affine
        self.mapped_axes = list(mapped_axes) if mapped_axes else []

        self.codebook_raw = nn.Parameter(torch.empty(0))  

        if self.affine:
            self.codebook_mean = nn.Parameter(torch.empty(0))  
            self.codebook_scale = nn.Parameter(torch.empty(0)) 
        else:
            self.codebook_mean = None
            self.codebook_scale = None

    @staticmethod
    def _replace_gradient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.detach() + y - y.detach()

    def _lazy_init_params(self, features: torch.Tensor):
        if self.codebook_raw.numel() != 0:
            return
        K = self.codebook_size
        D = int(features.shape[-1])
        device, dtype = features.device, features.dtype

        raw = torch.empty((K, D), device=device, dtype=dtype)
        nn.init.uniform_(raw, a=-0.1, b=0.1)
        self.codebook_raw = nn.Parameter(raw)

        if self.affine:
            mean = torch.zeros((1, D), device=device, dtype=dtype)
            scale = torch.zeros((1, D), device=device, dtype=dtype)
            self.codebook_mean = nn.Parameter(mean)
            self.codebook_scale = nn.Parameter(scale)

    def _maybe_psum(self, x: torch.Tensor) -> torch.Tensor:
        """JAX lax.psum over mapped_axes."""
        if not self.mapped_axes:
            return x
        if dist is None or (not dist.is_available()) or (not dist.is_initialized()):
            return x
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    def forward(
        self,
        features: torch.Tensor,   
        mask: torch.Tensor,       
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        self._lazy_init_params(features)

        codebook = 10.0 * self.codebook_raw

        if self.affine:
            codebook_scale = torch.exp(self.codebook_scale)
            codebook = self.codebook_mean + codebook_scale * codebook

        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        mask_loss = mask.to(dtype=features.dtype)

        distance = (features[:, None, :] - codebook[None, :, :]).pow(2).sum(dim=-1)
        distance = torch.where(mask_bool[:, None], distance, torch.full_like(distance, float("inf")))

        assign_fwd = torch.argmin(distance, dim=1)  # (N,)
        assign_rev = torch.argmin(distance, dim=0)  # (K,)

        features_fwd = codebook[assign_fwd]  # (N, D)
        features_rev = features[assign_rev]  # (K, D)

        out_features = self._replace_gradient(features_fwd, features)

        total = torch.clamp(mask_loss.sum(), min=1.0)

        codebook_loss_per = (features_fwd - features.detach()).pow(2).mean(dim=-1)
        codebook_loss = torch.where(mask_bool, codebook_loss_per, torch.zeros_like(codebook_loss_per)).sum() / total

        commitment_loss_per = (features_fwd.detach() - features).pow(2).mean(dim=-1)
        commitment_loss = torch.where(mask_bool, commitment_loss_per, torch.zeros_like(commitment_loss_per)).sum() / total
        # if we are mapping over one or more axes
        # sum assignment_count over all of them
        local_assignment_count = torch.zeros((self.codebook_size,), device=features.device, dtype=torch.float32)
        local_assignment_count.scatter_add_(0, assign_fwd, mask_loss.to(dtype=local_assignment_count.dtype))

        assignment_count = self._maybe_psum(local_assignment_count)

        assignment_mask = local_assignment_count < 1.0 

        unassigned_loss_per = (codebook - features_rev.detach()).pow(2).mean(dim=-1)
        unassigned_loss_per = unassigned_loss_per * assignment_mask.to(dtype=unassigned_loss_per.dtype)
        unassigned_loss = unassigned_loss_per.sum() / torch.clamp(
            assignment_mask.to(dtype=unassigned_loss_per.dtype).sum(),
            min=1.0,
        )

        losses = dict(
            codebook=codebook_loss,
            commitment=commitment_loss,
            unassigned=unassigned_loss,
            unassigned_percent=(assignment_count > 0).to(dtype=assignment_count.dtype).mean(),
        )
        return out_features, assign_fwd, losses

class FSQ(nn.Module):
    """Finite scalar quantization module.
    
    Not used in the manuscript.
    """

    def __init__(self, name: Optional[str] = "fsq"):
        super().__init__()
        self.downcast = Linear(7, bias=False)

        self.upcast: Optional[Linear] = None
        self._out_dim: Optional[int] = None

    def _ensure_upcast(self, out_dim: int, device: torch.device, dtype: torch.dtype):
        if self.upcast is not None:
            if self._out_dim != out_dim:
                raise ValueError(f"FSQ initialized with out_dim={self._out_dim}, got out_dim={out_dim}")
            return
        self.upcast = Linear(int(out_dim), bias=False).to(device=device, dtype=dtype)
        self._out_dim = int(out_dim)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        downcast = self.downcast(features)  

        half_l = (3.0 * (1.0 - 1e-3)) / 2.0
        offset = 0.5

        half_l_t = torch.as_tensor(half_l, device=downcast.device, dtype=downcast.dtype)
        offset_t = torch.as_tensor(offset, device=downcast.device, dtype=downcast.dtype)
        shift = torch.tan(offset_t / half_l_t)

        downcast = torch.tanh(downcast + shift) * half_l_t - offset_t

        rounded = torch.round(downcast.detach())
        rounded = rounded.detach() + (downcast - downcast.detach())  

        D = int(features.shape[-1])
        self._ensure_upcast(D, device=features.device, dtype=features.dtype)

        upcast = self.upcast(rounded / 2.0) 
        return upcast, rounded

class VQState(nn.Module):
    """Vector quantization module with state."""

    def __init__(
        self,
        codebook_size: int = 4096,
        gamma: float = 0.99,
        mapped_axes: Optional[List[str]] = None,  
        name: Optional[str] = "vq",
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.gamma = float(gamma)
        self.mapped_axes = list(mapped_axes) if mapped_axes else []

        self.register_buffer("codebook", torch.empty(0), persistent=True)  
        self.register_buffer("count", torch.empty(0), persistent=True)    
        self.register_buffer("avg", torch.empty(0), persistent=True)      

    def _maybe_psum_(self, x: torch.Tensor) -> torch.Tensor:
        """JAX lax.psum analogue over replicas."""
        if not self.mapped_axes:
            return x
        if dist is None or (not dist.is_available()) or (not dist.is_initialized()):
            return x
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    def _maybe_pmean_(self, x: torch.Tensor) -> torch.Tensor:
        """JAX lax.pmean analogue over replicas."""
        if not self.mapped_axes:
            return x
        if dist is None or (not dist.is_available()) or (not dist.is_initialized()):
            return x
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y /= float(dist.get_world_size())
        return y

    def _lazy_init_state(self, features: torch.Tensor):
        if self.codebook.numel() != 0:
            return
        K = self.codebook_size
        D = int(features.shape[-1])
        device = features.device
        dtype = features.dtype

        self.codebook = torch.randn((K, D), device=device, dtype=dtype)
        self.count = torch.zeros((K,), device=device, dtype=torch.float32)
        self.avg = torch.zeros((K,), device=device, dtype=torch.float32)

    @staticmethod
    def _replace_gradient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.detach() + y - y.detach()

    def forward(
        self,
        features: torch.Tensor,         
        mask: torch.Tensor,             
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Returns:
            out_features: (N, D) straight-through output
            assign_fwd: (N,) indices into codebook
            losses: dict(commitment=..., unassigned_percent=...)
            state_update: dict(count=..., avg=..., codebook=...)  (new state)
        """
        self._lazy_init_state(features)

        codebook = self.codebook
        prev_codebook = codebook
        count = self.count
        avg = self.avg

        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        mask_loss = mask.to(dtype=features.dtype)

        distance = (features[:, None, :] - codebook[None, :, :]).pow(2).sum(dim=-1)
        distance = torch.where(mask_bool[:, None], distance, torch.full_like(distance, float("inf")))

        assign_fwd = torch.argmin(distance, dim=1)  
        assign_rev = torch.argmin(distance, dim=0)  

        features_fwd = codebook[assign_fwd]         
        features_rev = features[assign_rev]         

        out_features = self._replace_gradient(features_fwd, features)

        total = torch.clamp(mask_loss.sum(), min=1.0)

        codebook_loss_per = (features_fwd - features.detach()).pow(2).mean(dim=-1)
        codebook_loss = torch.where(mask_bool, codebook_loss_per, torch.zeros_like(codebook_loss_per)).sum() / total

        commitment_loss_per = (features_fwd.detach() - features).pow(2).mean(dim=-1)
        commitment_loss = torch.where(mask_bool, commitment_loss_per, torch.zeros_like(commitment_loss_per)).sum() / total

        assignment_count = torch.zeros((self.codebook_size,), device=features.device, dtype=torch.float32)
        assignment_count.scatter_add_(0, assign_fwd, mask_loss.to(dtype=assignment_count.dtype))

        assignment_count = self._maybe_psum_(assignment_count)

        assignment_mask = assignment_count < 1.0

        gamma = self.gamma
        count_new = (1.0 - gamma) * assignment_count + gamma * count
        avg_new = (1.0 - gamma) * (assignment_count / total) + gamma * avg

        alpha = torch.exp(-avg_new * codebook.shape[0] * 10.0 / (1.0 - gamma) - 1e-3)  # (K,)

        accum = torch.zeros_like(codebook)
        accum.scatter_add_(0, assign_fwd[:, None].expand(-1, codebook.shape[-1]), features)
        assigned_update = gamma * codebook + (1.0 - gamma) * accum
        assigned_update = assigned_update / torch.clamp(count_new[:, None], min=1.0)

        unassigned_update = (1.0 - alpha)[:, None] * codebook + alpha[:, None] * features_rev

        codebook_new = torch.where(assignment_mask[:, None], assigned_update, unassigned_update)

        codebook_delta = prev_codebook - codebook_new
        codebook_delta = self._maybe_pmean_(codebook_delta)
        codebook_new = prev_codebook + codebook_delta

        losses = dict(
            commitment=commitment_loss,
            unassigned_percent=(assignment_count > 0).to(dtype=assignment_count.dtype).mean(),
        )
        state_update = dict(
            count=count_new,
            avg=avg_new,
            codebook=codebook_new,
        )

        self.count = count_new.detach()
        self.avg = avg_new.detach()
        self.codebook = codebook_new.detach()

        return out_features, assign_fwd, losses, state_update
