"""Autoencoder combining Encoder and Decoder."""

import math
import torch
import torch.nn as nn
from typing import Any, Optional, Dict, Tuple

import torch.nn.functional as F
from caesar.utils.geometry import Vec3Array 

from caesar.modules.utils.geometry import (
    unique_chain, compute_pseudo_cb, positions_to_ncacocb, 
    index_mean, index_align)
from caesar.modules.encoder import Encoder
from caesar.modules.decoder import Decoder
from caesar.modules.utils.dssp import assign_dssp
from caesar.utils.all_atom_multimer import get_atom14_mask
from caesar.utils.loss import violation_loss

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
            # mapped_axes is Haiku-specific; ignored here
            self.quantize = vq_cls(c.codebook_size, c.affine)
            
        # or FSQ (not used in the manuscript)
        self.fsq = FSQ() if getattr(c, "fsq", False) else None
        
    def forward(
            self,
            data: Dict[str, Any],
            *,
            generator: Optional[torch.Generator] = None,
            running_init: bool = False,
        ):
            c = self.config

            data = dict(data)
            
            # convert coordinates, center & add encoded features
            data.update(prepare_data(data, generator=generator))

            # optionally apply noise to the inputs
            if getattr(c, "input_diffusion", False):
                clean_latent = self.encoder(data)
                data["clean_latent"] = clean_latent
                data.update(self.prepare_input_diffusion(data))

            latent = self.encoder(data)
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
                    result_i = self.decoder(data, prev)
                    prev = {
                        "pos": result_i["pos"].detach(),
                        "local": result_i["local"].detach(),
                    }

            result = self.decoder(data, prev)

            if getattr(c, "codebook_size", 0):
                result["codebook_losses"] = codebook_losses

            total, losses = self.decoder.loss(data, result)

            out_dict = dict(results=result, losses=losses)
            if getattr(c, "codebook_size", 0) and getattr(c, "state", False):
                out_dict["_state_update"] = state_update

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

def prepare_data(data):
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
    noise = 0.3 * torch.randn(list(cb.shape) + [3], device=pos_ncacocb.device, dtype=pos_ncacocb.dtype)
    cb = cb + Vec3Array.from_array(noise)                
    dmap = (cb[:, None] - cb[None, :]).norm()            

    dmap_mask = batch[:, None].long() == batch[None, :].long()        

    # set all-atom-position target (GT targets)
    atom_pos = pos_14
    atom_mask_out = atom_mask_14.to(dtype)

    # assign dssp
    dssp, _, _ = assign_dssp(atom_pos, batch.long(), mask)

    # set initial backbone positions (decoder init)
    pos_init = torch.randn_like(pos_ncacocb)

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
        
class StructureDecoder(nn.Module):
    """Wrapper for training a structure autoencoder with a fixed encoder."""

    def __init__(self, config, name: Optional[str] = "structure_decoder"):
        super().__init__()
        self.config = config
        c = self.config

        # fixed encoder (loaded from params)
        self.encoder = assign_state_torch(c, c.param_path)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # learnable decoder
        self.decoder = Decoder(c)

    def add_noise(self, latent: torch.Tensor, batch: torch.Tensor, *, generator: Optional[torch.Generator] = None):
        """Add noise to latents."""
        batch = batch.to(torch.long)
        noise_level = torch.randn(batch.shape, device=latent.device, dtype=latent.dtype, generator=generator)[batch]
        noise_level = torch.exp(noise_level)
        noise = torch.randn(latent.shape, device=latent.device, dtype=latent.dtype, generator=generator) * noise_level
        return latent + noise

    def forward(
        self,
        data: Dict[str, Any],
        *,
        generator: Optional[torch.Generator] = None,
        running_init: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        c = self.config

        data = dict(data)
        data.update(prepare_data(data, generator=generator))

        with torch.no_grad():
            enc_out = self.encoder(data)

        if isinstance(enc_out, (tuple, list)) and len(enc_out) >= 1:
            latent = enc_out[0]
            codebook_index = enc_out[1] if len(enc_out) > 1 else None
        else:
            latent = enc_out
            codebook_index = None

        latent = latent.detach()
        data["latent"] = latent
        if codebook_index is not None:
            data["codebook_index"] = codebook_index

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
                result_i = self.decoder(data, prev)
                prev = {
                    "pos": result_i["pos"].detach(),
                    "local": result_i["local"].detach(),
                }

        result = self.decoder(data, prev)

        total, losses = self.decoder.loss(data, result)

        out_dict = dict(results=result, losses=losses)
        return total, out_dict

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
            result_i = self.decoder(data, prev)
            prev = {
                "pos": result_i["pos"].detach(),
                "local": result_i["local"].detach(),
            }

        result = self.decoder(data, prev)

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
