"""Autoencoder combining Encoder and Decoder."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from caesar.config import AutoencoderConfig, EncoderConfig, DecoderConfig
from caesar.utils.geometry import Vec3Array 

from caesar.modules.utils.geometry import (
    unique_chain, compute_pseudo_cb, positions_to_ncacocb, 
    index_mean)
from caesar.modules.encoder import Encoder
from caesar.modules.decoder import Decoder
from caesar.modules.utils.dssp import assign_dssp

class StructureAutoencoder(nn.Module):
    """Wrapper class for protein structure autoencoder training"""
    def __init__(self, 
                 config: Optional[AutoencoderConfig] = None,
                 encoder_config: Optional[EncoderConfig] = None,
                 decoder_config: Optional[DecoderConfig] = None):
        super().__init__()
        
        if config is None:
            config = AutoencoderConfig()
        
        if encoder_config is None:
            encoder_config = EncoderConfig(**vars(config))
        if decoder_config is None:
            decoder_config = DecoderConfig(**vars(config))
        
        self.config = config
        # self.encoder = Encoder(encoder_config)
        # self.decoder = Decoder(decoder_config)
    
    def prepare_data(self, data):
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
