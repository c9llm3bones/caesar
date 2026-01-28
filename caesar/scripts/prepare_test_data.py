import os
import sys
import pickle
import json
import time
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, "salad")

import numpy as np

import jax
import jax.numpy as jnp
import haiku as hk

from salad.aflib.common.protein import from_pdb_string
from salad.aflib.model.all_atom_multimer import atom37_to_atom14
from salad.aflib.model.geometry import Vec3Array
from salad.modules.config import distance_to_structure_decoder as config_choices

# use configs that have distogram_block set
CONFIG_NAME = "small"
config = deepcopy(getattr(config_choices, CONFIG_NAME))

PDB_PATH = "salad/data/afdb1024/test.cif"
SEED = 42
NUM_RECYCLE = 0
DEBUG = False
config.eval = True
config.num_recycle = NUM_RECYCLE

# add needed fields to config if missing
if not hasattr(config, 'time_embedding'):
    config.time_embedding = False
if not hasattr(config, 'latent_diffusion'):
    config.latent_diffusion = False
if not hasattr(config, 'input_diffusion'):
    config.input_diffusion = False

print(f"Config: {CONFIG_NAME}")
print(f"  - encoder_depth: {config.encoder_depth}")
print(f"  - aa_decoder_depth: {config.aa_decoder_depth}")
print(f"  - latent_size: {config.latent_size}")
print(f"  - local_size: {config.local_size}")
print(f"  - distogram_block: {config.distogram_block}")
print(f"  - num_recycle: {config.num_recycle}")
print(f"  - eval: {config.eval}")

def load_pdb(pdb_path):
    """Load PDB/CIF and convert to data dict for autoencoder."""
    from Bio.PDB import MMCIFParser, PDBParser
    
    with open(pdb_path, "rt") as f:
        pdb_str = f.read()
    
    try:
        if pdb_path.endswith('.cif'):
            from Bio.PDB.MMCIFParser import MMCIFParser
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as tmp:
                tmp.write(pdb_str)
                tmp_path = tmp.name
            
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('protein', tmp_path)
            
            model = structure[0]
            chain = list(model)[0]
            
            from Bio.PDB import PDBIO
            io = PDBIO()
            io.set_structure(structure)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb:
                io.save(tmp_pdb.name)
                with open(tmp_pdb.name, 'r') as f:
                    pdb_str = f.read()
            
            import os
            os.unlink(tmp_path)
            os.unlink(tmp_pdb.name)
    except Exception as e:
        print(f"Note: CIF parsing issue: {e}, trying direct parsing...")
    
    # SALAD's parser
    protein = from_pdb_string(pdb_str)
    
    aatype = protein.aatype
    atom_pos_37 = protein.atom_positions
    atom_mask_37 = protein.atom_mask
    residue_index = protein.residue_index
    chain_index = protein.chain_index
    
    num_aa = len(aatype)
    batch_index = np.zeros(num_aa, dtype=np.int32)
    
    atom_pos_14, atom_mask_14 = atom37_to_atom14(
        aatype, Vec3Array.from_array(atom_pos_37), atom_mask_37
    )
    atom_pos_14_arr = np.array(atom_pos_14.to_array(), dtype=np.float32)  
    atom_mask_14_arr = np.array(atom_mask_14, dtype=np.float32)           

    N_pos  = atom_pos_14_arr[:, 0, :]
    CA_pos = atom_pos_14_arr[:, 1, :]
    C_pos  = atom_pos_14_arr[:, 2, :]
    O_pos  = atom_pos_14_arr[:, 3, :]
    CB_pos = atom_pos_14_arr[:, 4, :]

    CA_mask = atom_mask_14_arr[:, 1].astype(np.float32)  
    CB_mask = atom_mask_14_arr[:, 4].astype(np.float32)

    CB_pos_fixed = np.where(CB_mask[:, None] > 0.0, CB_pos, CA_pos)

    pos_gt = np.stack([N_pos, CA_pos, C_pos, O_pos, CB_pos_fixed], axis=1).astype(np.float32)

    mask = (CA_mask > 0.0).astype(np.float32)

    cb = pos_gt[:, 4, :]  
    diff = cb[:, None, :] - cb[None, :, :]
    dmap = np.sqrt((diff * diff).sum(axis=-1)).astype(np.float32)

    pos = pos_gt.copy()
    latent_size = int(getattr(config, "latent_size", 20))
    
    data = {
        "all_atom_positions": atom_pos_14_arr,        
        "all_atom_mask": atom_mask_14_arr,            
        "atom_pos": atom_pos_14_arr,   
        "atom_mask": atom_mask_14_arr, 
        "mask": mask,                           
        "pos_gt": pos_gt,                       
        "pos": pos,                             
        "dmap": dmap,  
        "aa_gt": aatype,
        "residue_index": residue_index,
        "chain_index": chain_index,
        "batch_index": batch_index,
        "seq_mask": (aatype != 20).astype(np.float32),
        "residue_mask": atom_mask_37[:, 1].astype(np.float32),  
        # Add required fields for autoencoder 
        "time": np.array([0.0], dtype=np.float32),  
        "dssp": np.zeros(num_aa, dtype=np.int32), 
        "latent": np.zeros(20, dtype=np.float32), 
    }
    if DEBUG:
        print(data['all_atom_positions'].shape)
        print(data['all_atom_mask'].shape)
        print(data['aa_gt'])
        print(data['aa_gt'].shape)
    return data, num_aa

import os
pdb_abs_path = os.path.abspath(PDB_PATH)
try:
    data, num_aa = load_pdb(pdb_abs_path)
    print(f"Structure loaded: {num_aa} residues")
    print(f"  File: {pdb_abs_path}")
    print(f"  Keys: {list(data.keys())}")
except FileNotFoundError as e:
    print(f"File not found: {pdb_abs_path}")
    print(f"Error: {e}")
    data = None
except Exception as e:
    print(f"Error loading structure: {e}")
    import traceback
    traceback.print_exc()
    data = None
    
    
OUT = "tests/data/test_structure_extended.npz"

np.savez(
    OUT,
    **data,
    num_aa=num_aa
)

print(f"Saved test data to {OUT}")
