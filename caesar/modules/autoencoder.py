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
from caesar.modules.encoder import Encoder
from caesar.modules.decoder import Decoder
from caesar.modules.utils.dssp import assign_dssp
from caesar.utils.all_atom_multimer import get_atom14_mask
from caesar.utils.loss import violation_loss

DEBUG = True

def _stat(t: torch.Tensor):
    t = t.detach()
    return dict(mean=float(t.mean()), std=float(t.std()), max=float(t.abs().max()))


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
            return_trace=False
        ):
            c = self.config

            data = dict(data)
            
            # convert coordinates, center & add encoded features
            data.update(prepare_data(data, generator=generator))
            if DEBUG:
                trace = {}
                trace["prep/mask_sum"] = float(data["mask"].detach().sum())
                trace["prep/pos_gt"] = _stat(data["pos_gt"])
                trace["prep/pos_init"] = _stat(data["pos"])
                trace["prep/dmap"] = _stat(data["dmap"])
                
            # optionally apply noise to the inputs
            if getattr(c, "input_diffusion", False):
                clean_latent = self.encoder(data)
                data["clean_latent"] = clean_latent
                data.update(self.prepare_input_diffusion(data))

            latent = self.encoder(data)
            if DEBUG:
                trace["enc/latent"] = _stat(latent)
                trace["enc/latent_slice"] = latent[:2, :5].detach().cpu()
            # optionally apply noise to the latents (not used in the manuscript)
            if getattr(c, "latent_diffusion", False):
                # NOTE(from authors): constraining latents is necessary for diffusion to work
                # with a trainable encoder. Otherwise, the model learns to cheat.
                # we do this by applying a parameter-less LayerNorm to the latent
                # vectors. This fixes the variance of the latent vectors to 1 and
                # bounds the achievable signal-to-noise ratio during the diffusion
                # process. Thus, the model has to actively learn to denoise.
                latent = F.layer_norm(latent, (latent.shape[-1],), weight=None, bias=None)
                data["clean_latent"] = latent
                latent, time = self.prepare_latent_diffusion(latent, data)
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
            prev = dict(
                pos=data["pos"],
                local=torch.zeros((data["pos"].shape[0], c.local_size), device=latent.device, dtype=torch.float32),
            )

            if not running_init:
                if getattr(c, "eval", False):
                    count = int(c.num_recycle)
                else:
                    count = int(torch.randint(0, 4, (), device=latent.device, generator=generator).item())

                for _ in range(count):
                    result_i = self.decoder(data, prev, generator=generator)
                    prev = {
                        "pos": result_i["pos"].detach(),
                        "local": result_i["local"].detach(),
                    }

            local0, pos0, resi, chain, batch, mask = self.decoder.prepare_features(data, prev)
            trace["dec/local0"] = _stat(local0)
            
            result = self.decoder(data, prev, generator=generator)

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
        batch = batch.to(torch.long)

        noise_level = torch.randn(batch.shape, device=latent.device, dtype=latent.dtype, generator=generator)[batch]
        noise_level = torch.exp(noise_level)

        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator) * noise_level
        return latent + noise

    def prepare_input_diffusion(self, data: dict, *, generator: torch.Generator | None = None):
        """When training as a structure diffusion model, prepare model input.
        
        Not used in the manuscript.
        """
        c = self.config

        batch = data["batch_index"].to(torch.long)
        pos = data["pos_input"]

        pos = pos - index_mean(pos[:, 1], batch, data["mask"][:, None])[:, None]  # FIXME (ported as-is)

        time = torch.rand(batch.shape, device=pos.device, dtype=pos.dtype, generator=generator)[batch]
        if "time" in data:
            time = data["time"].to(device=pos.device, dtype=pos.dtype) * torch.ones_like(time)

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

        batch = data["batch_index"].to(torch.long)
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator)

        if bool(c.vp_diffusion):
            time = torch.rand(batch.shape, device=latent.device, dtype=latent.dtype, generator=generator)[batch]
        else:
            time = torch.randn(batch.shape, device=latent.device, dtype=latent.dtype, generator=generator)[batch]
            time = torch.exp(torch.tensor(1.0, device=latent.device, dtype=latent.dtype) + 1.2 * time)

        if "time" in data:
            time = data["time"].to(device=latent.device, dtype=latent.dtype) * torch.ones_like(time)

        if bool(c.vp_diffusion):
            s = 0.01
            denom = math.cos(s / (1.0 + s) * math.pi / 2.0)
            time = torch.cos((time + s) / (1.0 + s) * math.pi / 2.0) / denom
            time = torch.sqrt(torch.clamp(1.0 - time ** 2, 0.0, 1.0))
            latent = torch.sqrt(1.0 - time[:, None] ** 2) * latent + time[:, None] * noise * 5.0
        else:
            latent = latent + time[:, None] * noise

        return latent, time

def prepare_data(data, generator=None):
    """Prepare model inputs from a batch of data."""
    pos = data["all_atom_positions"]   # [N, M, 3]
    atom_mask = data["all_atom_mask"]  # [N, M]
    chain = data["chain_index"]        # [N]
    batch = data["batch_index"]        # [N]

    if not torch.is_tensor(pos):
        pos = torch.as_tensor(pos)
    if not torch.is_tensor(atom_mask):
        atom_mask = torch.as_tensor(atom_mask)
    if not torch.is_tensor(chain):
        chain = torch.as_tensor(chain)
    if not torch.is_tensor(batch):
        batch = torch.as_tensor(batch)

    device = pos.device
    dtype = pos.dtype

    # recast positions into atom14:
    # first, truncate to atom14 format
    pos = pos[:, :14]
    atom_mask = atom_mask[:, :14]

    # boolean atom mask for logic
    atom_mask_bool = atom_mask.to(torch.bool)

    # uniquify chain IDs across batches:
    chain = unique_chain(chain.long(), batch.long())

    # mask = seq_mask * residue_mask * atom_mask[:,:3].all(axis=-1)
    seq_mask = torch.as_tensor(data["seq_mask"], device=device, dtype=dtype)
    residue_mask = torch.as_tensor(data["residue_mask"], device=device, dtype=dtype)
    mask = seq_mask * residue_mask * atom_mask_bool[:, :3].all(dim=-1).to(dtype)

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
    noise = 0.3 * torch.randn(list(cb.shape) + [3], device=pos_ncacocb.device, dtype=pos_ncacocb.dtype, generator=generator)
    cb = cb + Vec3Array.from_array(noise)                
    dmap = (cb[:, None] - cb[None, :]).norm()            

    dmap_mask = batch[:, None].long() == batch[None, :].long()        

    # set all-atom-position target (GT targets)
    atom_pos = pos_14
    atom_mask_out = atom_mask_14.to(dtype)

    # assign dssp
    dssp, _, _ = assign_dssp(atom_pos, batch.long(), mask)

    # set initial backbone positions (decoder init)
    pos_init = torch.randn(pos_ncacocb.shape, device=pos_ncacocb.device, dtype=pos_ncacocb.dtype, generator=generator) 

    return dict(
        pos=pos_init,
        pos_gt=pos_ncacocb,
        pos_input=pos_ncacocb,
        dssp=dssp,
        dmap=dmap,
        dmap_mask=dmap_mask,
        chain_index=chain,
        mask=mask,
        atom_pos=atom_pos,
        atom_mask=atom_mask_out,
        all_atom_positions=pos_14,
        all_atom_mask=atom_mask_out,
    )
        
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
    def forward(self, data: Dict[str, Any], *, generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
        c = self.config
        data = dict(data)

        # data.update(self.prepare_data(data))
        
        latent, codebook_index = self.encoder_apply(data, generator=generator)
        latent = latent.detach()
        data["latent"] = latent

        prev = dict(
            pos=data["pos"],
            local=torch.zeros(
                (data["pos"].shape[0], c.local_size),
                device=data["pos"].device,
                dtype=torch.float32,
            ),
        )

        count = int(c.num_recycle)
        for _ in range(count):
            result_i = self.decoder(data, prev, generator=generator)
            prev = {
                "pos": result_i["pos"].detach(),
                "local": result_i["local"].detach(),
            }

        result = self.decoder(data, prev, generator=generator)

        mask = data["mask"].to(torch.bool)
        aa_gt = data["aa_gt"].to(torch.long)

        aa_pred = result["aa"]  
        aa_onehot = F.one_hot(aa_gt, 20).to(dtype=aa_pred.dtype)

        aa_nll = -(aa_pred * aa_onehot).sum(dim=-1)
        aa_nll = torch.where(mask, aa_nll, torch.zeros_like(aa_nll))
        aa_nll = aa_nll.sum() / torch.clamp(mask.sum().to(aa_nll.dtype), min=1.0)

        perplexity = torch.exp(aa_nll)
        aatype = torch.argmax(aa_pred, dim=-1)
        recovery = ((aa_gt == aatype) & mask).sum().to(torch.float32) / torch.clamp(mask.sum().to(torch.float32), min=1.0)

        pos_gt = data["pos_gt"]
        pos = result["pos"]

        pos_gt = index_align(pos_gt, pos, data["batch_index"], mask)

        di2 = ((pos[:, 1] - pos_gt[:, 1]) ** 2).sum(dim=-1)
        rmsd_ca = torch.sqrt((di2 * mask.to(di2.dtype)).sum() / torch.clamp(mask.sum().to(di2.dtype), min=1.0))

        L = torch.clamp(mask.sum().to(torch.float32), min=1.0)
        d02 = (1.24 * torch.pow(torch.clamp(L - 15.0, min=0.0), 1.0 / 3.0) - 1.8) ** 2
        inner = (1.0 / (1.0 + di2 / torch.clamp(d02, min=1e-8))) * mask.to(di2.dtype)
        tm = inner.sum() / torch.clamp(mask.sum().to(inner.dtype), min=1.0)

        dca_gt = torch.linalg.norm(pos_gt[:, None, 1] - pos_gt[None, :, 1], dim=-1)
        dca = torch.linalg.norm(pos[:, None, 1] - pos[None, :, 1], dim=-1)

        pair_mask = (mask[:, None] & mask[None, :])
        derr = (dca_gt - dca).abs() * pair_mask.to(dca.dtype)

        threshold = torch.tensor([0.5, 1.0, 2.0, 4.0], device=derr.device, dtype=derr.dtype)
        Rinc = 15.0
        pair_mask = pair_mask & (dca_gt < Rinc)

        in_threshold = (derr[..., None] < threshold) & pair_mask[..., None]
        denom = torch.clamp(pair_mask[..., None].sum(dim=1).to(torch.float32), min=1.0)
        lddt_ca = (in_threshold.sum(dim=1).to(torch.float32) / denom).mean(dim=-1)
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
            dssp=data["dssp"],
        )
        if getattr(c, "codebook_size", 0):
            out["codebook_index"] = codebook_index

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

        latent = self.encoder(data)

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
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        c = self.config
        data = dict(data)

        data.update(self.prepare_data_fn(data) if generator is None else self.prepare_data_fn(data, generator=generator))

        latent, codebook_index = self.encoder_apply(data, generator=generator)

        latent = latent.detach()
        data["latent"] = latent

        prev = dict(
            pos=data["pos"],
            local=torch.zeros(
                (data["pos"].shape[0], c.local_size),
                device=data["pos"].device,
                dtype=torch.float32,
            ),
        )

        if not running_init:
            if getattr(c, "eval", False):
                count = int(c.num_recycle)
            else:
                if generator is None:
                    count = int(torch.randint(0, 4, (1,), device="cpu").item())
                else:
                    count = int(torch.randint(0, 4, (1,), generator=generator, device="cpu").item())

            for _ in range(count):
                result_i = self.decoder(data, prev, generator=generator)
                prev = dict(
                    pos=result_i["pos"].detach(),
                    local=result_i["local"].detach(),
                )

        result = self.decoder(data, prev, generator=generator)

        total, losses = self.decoder.loss(data, result, generator=generator)

        out_dict = dict(
            results=result,
            losses=losses,
        )
        return total, out_dict

    def add_noise(
        self,
        latent: torch.Tensor,
        batch: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Add noise to latents."""
        batch = batch.to(torch.long)

        if generator is None:
            noise_level_table = torch.randn(batch.shape, device=latent.device, dtype=latent.dtype)
            noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype)
        else:
            noise_level_table = torch.randn(batch.shape, generator=generator, device=latent.device, dtype=latent.dtype)
            noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)

        noise_level = noise_level_table[batch]
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
    def forward(self, data: Dict[str, Any], *, generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
        c = self.config

        data = dict(data)
        data.update(self.prepare_data(data, generator=generator))

        # optionally apply noise to inputs
        if getattr(c, "input_diffusion", False):
            clean_latent = self.encoder(data)
            data["clean_latent"] = clean_latent
            data.update(self.prepare_input_diffusion(data, generator=generator))

        latent = self.encoder(data)

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
            print("time", float(data["time"][0].item()))
        else:
            data.setdefault("time", torch.zeros((data["mask"].shape[0],), device=latent.device, dtype=latent.dtype))

        codebook_index = None
        if getattr(c, "codebook_size", 0):
            latent, codebook_index, _ = self.quantize(latent, data["mask"])

        data["latent"] = latent

        prev = dict(
            pos=data["pos"],
            local=torch.zeros((data["pos"].shape[0], c.local_size), device=latent.device, dtype=torch.float32),
        )

        count = int(c.num_recycle)
        for _ in range(count):
            result_i = self.decoder(data, prev, generator=generator)
            prev = {
                "pos": result_i["pos"].detach(),
                "local": result_i["local"].detach(),
            }

        result = self.decoder(data, prev, generator=generator)

        mask = data["mask"].to(torch.bool)

        aa_logits_or_logprobs = result["aa"]
        aa_gt = data["aa_gt"].to(torch.long)
        aa_onehot = F.one_hot(aa_gt, 20).to(dtype=aa_logits_or_logprobs.dtype)
        aa_nll = -(aa_logits_or_logprobs * aa_onehot).sum(dim=-1)
        aa_nll = torch.where(mask, aa_nll, torch.zeros_like(aa_nll))
        aa_nll = aa_nll.sum() / torch.clamp(mask.sum().to(aa_nll.dtype), min=1.0)
        perplexity = torch.exp(aa_nll)

        aatype = torch.argmax(aa_logits_or_logprobs, dim=-1)
        recovery = ((aatype == aa_gt) & mask).sum().to(torch.float32) / torch.clamp(mask.sum().to(torch.float32), min=1.0)

        # RMSD (CA)
        pos_gt = data["pos_gt"]
        pos = result["pos"]

        pos_gt_aligned = index_align(pos_gt, pos, data["batch_index"], mask)

        di2 = ((pos[:, 1] - pos_gt_aligned[:, 1]) ** 2).sum(dim=-1)
        rmsd_ca = torch.sqrt((di2 * mask.to(di2.dtype)).sum() / torch.clamp(mask.sum().to(di2.dtype), min=1.0))

        # TM score (CA)
        L = torch.clamp(mask.sum().to(torch.float32), min=1.0)
        d02 = (1.24 * torch.pow(torch.clamp(L - 15.0, min=0.0), 1.0 / 3.0) - 1.8) ** 2
        inner = (1.0 / (1.0 + di2 / torch.clamp(d02, min=1e-8))) * mask.to(di2.dtype)
        tm = inner.sum() / torch.clamp(mask.sum().to(inner.dtype), min=1.0)

        # LDDT (CA)
        dca_gt = torch.linalg.norm(pos_gt_aligned[:, None, 1] - pos_gt_aligned[None, :, 1], dim=-1)
        dca = torch.linalg.norm(pos[:, None, 1] - pos[None, :, 1], dim=-1)
        pair_mask = (mask[:, None] & mask[None, :])

        derr = (dca_gt - dca).abs() * pair_mask.to(dca.dtype)

        threshold = torch.tensor([0.5, 1.0, 2.0, 4.0], device=dca.device, dtype=dca.dtype)
        Rinc = 15.0
        pair_mask = pair_mask & (dca_gt < Rinc)

        in_threshold = (derr[..., None] < threshold) & pair_mask[..., None]
        denom = torch.clamp(pair_mask[..., None].sum(dim=1).to(torch.float32), min=1.0)
        lddt_ca = (in_threshold.sum(dim=1).to(torch.float32) / denom).mean(dim=-1)
        lddt_ca = torch.where(mask, lddt_ca.to(mask.dtype), torch.zeros_like(lddt_ca).to(mask.dtype))

        # AlphaFold violation loss
        res_mask = mask.to(torch.float32)
        pred_mask = get_atom14_mask(aatype).to(res_mask.device).to(res_mask.dtype) * res_mask[:, None]

        violation, _ = violation_loss(
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
        violation_error = violation.mean()

        latent_out = data["latent"]
        if "predicted_latent" in result:
            print("returning predicted latent")
            latent_out = result["predicted_latent"]

        out = dict(
            atom_pos=result["atom_pos"],
            aatype=aatype,
            latent=latent_out,
            local=result["local"],
            perplexity=perplexity,
            recovery=recovery,
            rmsd_ca=rmsd_ca,
            tm=tm,
            lddt=lddt_ca,
            time=data["time"],
            dssp=data["dssp"],
            violation=violation_error,
        )
        if getattr(c, "codebook_size", 0):
            out["codebook_index"] = codebook_index

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

        mask_bool = mask.to(torch.bool)
        mask_f = mask.to(torch.float32)

        distance = (features[:, None, :] - codebook[None, :, :]).pow(2).sum(dim=-1)
        distance = torch.where(mask_bool[:, None], distance, torch.full_like(distance, float("inf")))

        assign_fwd = torch.argmin(distance, dim=1)  # (N,)
        assign_rev = torch.argmin(distance, dim=0)  # (K,)

        features_fwd = codebook[assign_fwd]  # (N, D)
        features_rev = features[assign_rev]  # (K, D)

        out_features = self._replace_gradient(features_fwd, features)

        total = torch.clamp(mask_f.sum(), min=1.0)

        codebook_loss_per = (features_fwd - features.detach()).pow(2).mean(dim=-1)
        codebook_loss = torch.where(mask_bool, codebook_loss_per, torch.zeros_like(codebook_loss_per)).sum() / total

        commitment_loss_per = (features_fwd.detach() - features).pow(2).mean(dim=-1)
        commitment_loss = torch.where(mask_bool, commitment_loss_per, torch.zeros_like(commitment_loss_per)).sum() / total
        # if we are mapping over one or more axes
        # sum assignment_count over all of them
        local_assignment_count = torch.zeros((self.codebook_size,), device=features.device, dtype=torch.float32)
        local_assignment_count.scatter_add_(0, assign_fwd, mask_f)

        assignment_count = self._maybe_psum(local_assignment_count)

        assignment_mask = local_assignment_count < 1.0 

        unassigned_loss_per = (codebook - features_rev.detach()).pow(2).mean(dim=-1)
        unassigned_loss_per = unassigned_loss_per * assignment_mask.to(unassigned_loss_per.dtype)
        unassigned_loss = unassigned_loss_per.sum() / torch.clamp(assignment_mask.to(torch.float32).sum(), min=1.0)

        losses = dict(
            codebook=codebook_loss,
            commitment=commitment_loss,
            unassigned=unassigned_loss,
            unassigned_percent=(assignment_count > 0).to(torch.float32).mean(),
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

        mask_bool = mask.to(torch.bool)
        mask_f = mask.to(torch.float32)

        distance = (features[:, None, :] - codebook[None, :, :]).pow(2).sum(dim=-1)
        distance = torch.where(mask_bool[:, None], distance, torch.full_like(distance, float("inf")))

        assign_fwd = torch.argmin(distance, dim=1)  
        assign_rev = torch.argmin(distance, dim=0)  

        features_fwd = codebook[assign_fwd]         
        features_rev = features[assign_rev]         

        out_features = self._replace_gradient(features_fwd, features)

        total = torch.clamp(mask_f.sum(), min=1.0)

        codebook_loss_per = (features_fwd - features.detach()).pow(2).mean(dim=-1)
        codebook_loss = torch.where(mask_bool, codebook_loss_per, torch.zeros_like(codebook_loss_per)).sum() / total

        commitment_loss_per = (features_fwd.detach() - features).pow(2).mean(dim=-1)
        commitment_loss = torch.where(mask_bool, commitment_loss_per, torch.zeros_like(commitment_loss_per)).sum() / total

        assignment_count = torch.zeros((self.codebook_size,), device=features.device, dtype=torch.float32)
        assignment_count.scatter_add_(0, assign_fwd, mask_f)

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
            unassigned_percent=(assignment_count > 0).to(torch.float32).mean(),
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