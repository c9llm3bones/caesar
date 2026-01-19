# Torch port of JAX version used in SALAD code

# Adapted from PyDSSP

# MIT License

# Copyright (c) 2022 Shintaro Minami

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn.functional as F


CONST_Q1Q2 = 0.084
CONST_F = 332
DEFAULT_CUTOFF = -0.5
DEFAULT_MARGIN = 1.0
SHEET_ADJACENCY = 8.0
OTHER_ADJACENCY = 8.0

N_INDEX = 0
CA_INDEX = 1
CO_INDEX = 2
O_INDEX = 3

import torch

def compute_dssp(atom_pos: torch.Tensor,
                 batch_index: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    dssp, _, _ = assign_dssp(atom_pos, batch_index, mask)
    return dssp

def _unfold(a: torch.Tensor, window: int, axis: int) -> torch.Tensor:
    if axis < 0:
        axis = a.dim() + axis
    assert 0 <= axis < a.dim()

    # torch.unfold returns shape:
    #   dims < axis, new_len, window, dims > axis
    unfolded = a.unfold(dimension=axis, size=window, step=1)

    # Move the "window" dimension (axis+1) to the last, matching the JAX moveaxis behavior.
    win_dim = axis + 1
    if win_dim != unfolded.dim() - 1:
        perm = list(range(unfolded.dim()))
        perm.pop(win_dim)
        perm.append(win_dim)
        unfolded = unfolded.permute(*perm)
    return unfolded

def _check_input(coord: torch.Tensor):
    org_shape = coord.shape
    # salad code asserts len==3 though docstring mentions batch means len can be 4. 
    # we implement as in original.
    assert len(org_shape) == 3, (
        "Shape of input tensor should be [L, atom, xyz] or [batch, L, atom, xyz]"
    )
    return coord, org_shape


def _get_peptide_bond_h_position(coord: torch.Tensor) -> torch.Tensor:
    """
    coord: [L, 4, 3] containing atoms [N, CA, C(=CO_INDEX here), O]
    returns H positions for residues 1..L-1: [L-1, 3]
    """
    # vec_cn = N_{i+1} - C_i
    vec_cn = coord[1:, N_INDEX] - coord[:-1, CO_INDEX]
    vec_cn = vec_cn / torch.clamp(torch.linalg.norm(vec_cn, dim=-1, keepdim=True), min=1e-3)

    # vec_can = N_{i+1} - CA_{i+1}
    vec_can = coord[1:, N_INDEX] - coord[1:, CA_INDEX]
    vec_can = vec_can / torch.clamp(torch.linalg.norm(vec_can, dim=-1, keepdim=True), min=1e-3)

    vec_nh = vec_cn + vec_can
    vec_nh = vec_nh / torch.clamp(torch.linalg.norm(vec_nh, dim=-1, keepdim=True), min=1e-3)

    # H approx at N + 1.01 * direction
    return coord[1:, N_INDEX] + 1.01 * vec_nh


def get_hbond_map(
    coord: torch.Tensor,
    mask: torch.Tensor,
    cutoff: float = DEFAULT_CUTOFF,
    margin: float = DEFAULT_MARGIN,
) -> torch.Tensor:
    """
    coord: [L, atom, 3] (atom >= 4, expects N,CA,C,O in first 4 slots)
    mask:  [L, L] pair mask (boolean or 0/1 float)
    returns: hbond_map [L, L] in [0,1]
    """
    coord, _ = _check_input(coord)
    num_aa, num_atoms, _ = coord.shape
    assert num_atoms >= 4, "Number of atoms should be at least 4 (N,CA,C,O)"

    device = coord.device
    dtype = coord.dtype

    coord = coord[:, :4]  # [L,4,3]
    h = _get_peptide_bond_h_position(coord)  # [L-1,3]

    # distance matrix components
    # nmap: N_{i} for i=1..L-1  -> [L-1,1,3]
    nmap = coord[1:, None, N_INDEX]
    # hmap: H_{i} for i=1..L-1 -> [L-1,1,3]
    hmap = h[:, None]
    # cmap: C_{j} for j=0..L-2  -> [1,L-1,3]
    cmap = coord[None, :-1, CO_INDEX]
    # omap: O_{j} for j=0..L-2  -> [1,L-1,3]
    omap = coord[None, :-1, O_INDEX]

    d_on = torch.linalg.norm(omap - nmap, dim=-1) + 1e-3
    d_ch = torch.linalg.norm(cmap - hmap, dim=-1) + 1e-3
    d_oh = torch.linalg.norm(omap - hmap, dim=-1) + 1e-3
    d_cn = torch.linalg.norm(cmap - nmap, dim=-1) + 1e-3

    # electrostatic interaction energy
    e_small = CONST_Q1Q2 * (1.0 / d_on + 1.0 / d_ch - 1.0 / d_oh - 1.0 / d_cn) * CONST_F  # [L-1,L-1]

    # jnp.pad(e_small, [[1,0],[0,1]]) => add 1 row at top, 1 col at right
    e = F.pad(e_small, pad=(0, 1, 1, 0), mode="constant", value=0.0)  # [L,L]

    # local mask: exclude (i,i), (i,i-1), (i,i-2)  (same as JAX code)
    local_mask = ~torch.eye(num_aa, dtype=torch.bool, device=device)
    if num_aa >= 2:
        local_mask = local_mask & ~torch.diag(torch.ones(num_aa - 1, dtype=torch.bool, device=device), diagonal=-1)
    if num_aa >= 3:
        local_mask = local_mask & ~torch.diag(torch.ones(num_aa - 2, dtype=torch.bool, device=device), diagonal=-2)

    # hydrogen bond map (continuous value extension)
    hbond_map = torch.clamp(torch.tensor(cutoff - margin, device=device, dtype=dtype) - e, min=-margin, max=margin)
    hbond_map = (torch.sin(hbond_map / margin * torch.pi / 2.0) + 1.0) / 2.0
    hbond_map = torch.where(local_mask, hbond_map, torch.zeros_like(hbond_map))

    # apply pair mask
    if mask.dtype != torch.bool:
        mask_bool = mask != 0
    else:
        mask_bool = mask
    hbond_map = torch.where(mask_bool, hbond_map, torch.zeros_like(hbond_map))
    return hbond_map


def _compute_blocks(index: torch.Tensor) -> torch.Tensor:
    """
    JAX lax.scan:
      increment block when index changes.
    index: [L] int
    returns blocks: [L] int starting from 0
    """
    L = index.shape[0]
    blocks = torch.zeros(L, dtype=torch.long, device=index.device)
    if L == 0:
        return blocks
    for i in range(1, L):
        blocks[i] = blocks[i - 1] + (index[i] != index[i - 1]).long()
    return blocks


def _block_min_distance(distance: torch.Tensor, blocks: torch.Tensor, big: float = 1e6) -> torch.Tensor:
    """
    distance: [L,L] float
    blocks:   [L] long in [0..B-1]
    returns:  Dblock_expanded [L,L] where each entry is min distance between the two blocks.
    """
    device = distance.device
    dtype = distance.dtype
    L = blocks.shape[0]
    B = int(blocks.max().item()) + 1 if L > 0 else 0
    if B == 0:
        return distance.new_zeros((L, L))

    # Build min-distance matrix between blocks: [B,B]
    D = torch.full((B, B), big, device=device, dtype=dtype)

    bi = blocks[:, None].expand(L, L)
    bj = blocks[None, :].expand(L, L)
    flat_idx = (bi * B + bj).reshape(-1)                 # [L*L]
    flat_val = distance.reshape(-1)                      # [L*L]

    # Preferred: scatter_reduce amin (PyTorch >= 1.12/2.0 depending)
    if hasattr(D, "scatter_reduce_"):
        D_flat = D.reshape(-1)
        D_flat.scatter_reduce_(0, flat_idx, flat_val, reduce="amin", include_self=True)
        D = D_flat.reshape(B, B)
    else:
        # Fallback (slower): loop over unique indices
        # Still OK for tests / CPU, but you may want to require newer PyTorch.
        for k, v in zip(flat_idx.tolist(), flat_val.tolist()):
            if v < D.view(-1)[k].item():
                D.view(-1)[k] = torch.tensor(v, device=device, dtype=dtype)

    return D[blocks[:, None], blocks[None, :]]


def assign_dssp(
    coord: torch.Tensor,
    batch_index: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    coord: [L, atom>=4, 3]
    batch_index: [L] int
    mask: [L] 0/1 or bool residue mask

    returns:
      secondary_structure: [L] int (0 loop, 1 helix, 2 strand) or 3 unknown in short case
      blocks: [L] int contiguous blocks of same ss label
      block_adjacency: [L,L] bool adjacency between non-loop blocks (computed via CA distances)
    """
    # Short sequence case: match JAX behavior
    if coord.shape[0] < 5:
        dssp = 3 * torch.ones_like(batch_index, dtype=torch.int32)
        blocks = torch.zeros_like(batch_index, dtype=torch.int32)
        block_adjacency = torch.zeros((batch_index.shape[0], batch_index.shape[0]), dtype=torch.bool, device=coord.device)
        return dssp, blocks, block_adjacency

    coord, _ = _check_input(coord)

    L = coord.shape[0]
    device = coord.device

    if batch_index.dtype != torch.long:
        batch_index = batch_index.long()
    if mask.dtype != torch.bool:
        mask_bool = mask != 0
    else:
        mask_bool = mask

    pair_mask = (batch_index[:, None] == batch_index[None, :]) & (mask_bool[:, None] & mask_bool[None, :])

    hbmap = get_hbond_map(coord, pair_mask)
    hbmap = hbmap.transpose(-1, -2)

    # identify turn 3,4,5 (diagonals with offsets)
    turn3 = torch.diagonal(hbmap, offset=3, dim1=-2, dim2=-1) > 0.0  # [L-3]
    turn4 = torch.diagonal(hbmap, offset=4, dim1=-2, dim2=-1) > 0.0  # [L-4]
    turn5 = torch.diagonal(hbmap, offset=5, dim1=-2, dim2=-1) > 0.0  # [L-5]

    # assignment of helical sses
    # h3 = pad(turn3[:-1] * turn3[1:], [[1,3]])
    # 1D pad: left=1, right=3
    h3 = F.pad((turn3[:-1] & turn3[1:]).to(torch.bool), (1, 3))
    h4 = F.pad((turn4[:-1] & turn4[1:]).to(torch.bool), (1, 4))
    h5 = F.pad((turn5[:-1] & turn5[1:]).to(torch.bool), (1, 5))

    # helix4 prioritized
    helix4 = h4 | torch.roll(h4, 1, 0) | torch.roll(h4, 2, 0) | torch.roll(h4, 3, 0)
    h3 = h3 & (~torch.roll(helix4, -1, 0)) & (~helix4)
    h5 = h5 & (~torch.roll(helix4, -1, 0)) & (~helix4)

    helix3 = h3 | torch.roll(h3, 1, 0) | torch.roll(h3, 2, 0)
    helix5 = h5 | torch.roll(h5, 1, 0) | torch.roll(h5, 2, 0) | torch.roll(h5, 3, 0) | torch.roll(h5, 4, 0)

    # identify bridge
    unfoldmap = _unfold(_unfold(hbmap, 3, -2), 3, -2) > 0.0  # [L-2, L-2, 3, 3]
    unfoldmap_rev = unfoldmap.transpose(0, 1)

    p_bridge = (unfoldmap[:, :, 0, 1] & unfoldmap_rev[:, :, 1, 2]) | (unfoldmap_rev[:, :, 0, 1] & unfoldmap[:, :, 1, 2])
    p_bridge = F.pad(p_bridge, (1, 1, 1, 1))  # to [L,L]
    p_bridge = torch.where(pair_mask, p_bridge, torch.zeros_like(p_bridge))

    a_bridge = (unfoldmap[:, :, 1, 1] & unfoldmap_rev[:, :, 1, 1]) | (unfoldmap[:, :, 0, 2] & unfoldmap_rev[:, :, 0, 2])
    a_bridge = F.pad(a_bridge, (1, 1, 1, 1))
    a_bridge = torch.where(pair_mask, a_bridge, torch.zeros_like(a_bridge))

    ladder = (p_bridge.to(torch.int32) + a_bridge.to(torch.int32)).sum(dim=-1) > 0  # [L]

    # strand (the JAX code computes more but then overwrites strand=ladder)
    strand = ladder
    helix = (helix3 | helix4 | helix5)

    loop = (~helix) & (~strand)

    # index = argmax(stack([loop, helix, strand]), axis=-1)
    # Since they are mutually exclusive-ish, argmax works. We'll mimic.
    stacked = torch.stack([loop, helix, strand], dim=-1).to(torch.int64)  # [L,3]
    index = torch.argmax(stacked, dim=-1).to(torch.long)  # 0/1/2

    blocks = _compute_blocks(index)  # [L]
    same_block = blocks[:, None] == blocks[None, :]

    sheet_sheet = (index[:, None] == 2) & (index[None, :] == 2)

    # CA distance matrix
    distance = torch.linalg.norm(coord[:, None, CA_INDEX] - coord[None, :, CA_INDEX], dim=-1)  # [L,L]
    block_distance = _block_min_distance(distance, blocks, big=1e6)

    block_adjacency = torch.where(
        sheet_sheet,
        block_distance <= SHEET_ADJACENCY,
        block_distance <= OTHER_ADJACENCY,
    )

    # remove adjacency within same block, and anything involving loops
    block_adjacency = torch.where(
        same_block | loop[:, None] | loop[None, :],
        torch.zeros_like(block_adjacency),
        block_adjacency,
    )

    secondary_structure = index.to(torch.int32)
    return secondary_structure, blocks.to(torch.int32), block_adjacency


def drop_dssp(
    generator: Optional[torch.Generator],
    secondary_structure: torch.Tensor,
    blocks: torch.Tensor,
    block_adjacency: torch.Tensor,
    p_drop: float = 0.2,
):
    """
    Port of:
      drop_mask = bernoulli(key, p_drop, shape)[blocks]
      drop_pair = drop_mask[:,None] + drop_mask[None,:] > 0
      secondary_structure = where(drop_mask, 0, secondary_structure)
      block_adjacency = where(drop_pair, 0, block_adjacency)
    """
    device = secondary_structure.device
    L = secondary_structure.shape[0]

    if blocks.dtype != torch.long:
        blocks_long = blocks.long()
    else:
        blocks_long = blocks

    # Sample length-L Bernoulli then index by blocks, exactly like JAX code.
    probs = torch.full((L,), p_drop, device=device, dtype=torch.float32)
    if generator is None:
        drop_mask_raw = torch.bernoulli(probs)
    else:
        drop_mask_raw = torch.bernoulli(probs, generator=generator)
    drop_mask = drop_mask_raw[blocks_long].to(torch.bool)

    drop_pair = drop_mask[:, None] | drop_mask[None, :]

    secondary_structure = torch.where(drop_mask, torch.zeros_like(secondary_structure), secondary_structure)
    block_adjacency = torch.where(drop_pair, torch.zeros_like(block_adjacency), block_adjacency)
    return secondary_structure, block_adjacency
