"""Centralized neighbour preparation for the experimental native path."""

from __future__ import annotations

import torch

from caesar.experimental.torch_native.types import NeighbourSet, PreparedBatch


def prepare_neighbours(
    prepared: PreparedBatch,
    dmap: torch.Tensor,
    *,
    fape_count: int,
    local_count: int,
) -> NeighbourSet:
    """Build deterministic fixed-shape neighbours for native losses."""
    batch = prepared.raw.batch_index.long()
    mask = prepared.mask
    fape = _topk_by_distance(dmap, batch, mask, int(fape_count), exclude_self=False)
    cb = prepared.pos_gt[:, -1]
    local_distance = torch.linalg.norm(cb[:, None] - cb[None, :], dim=-1)
    local_atom = _topk_by_distance(local_distance, batch, mask, int(local_count), exclude_self=False)
    return NeighbourSet(fape=fape, local_atom=local_atom)


def _topk_by_distance(
    distance: torch.Tensor,
    batch: torch.Tensor,
    mask: torch.Tensor,
    count: int,
    *,
    exclude_self: bool,
) -> torch.Tensor:
    n = int(distance.shape[0])
    if count <= 0:
        return torch.empty((n, 0), device=distance.device, dtype=torch.long)

    pair_mask = (batch[:, None] == batch[None, :]) & mask[:, None].bool() & mask[None, :].bool()
    dist = torch.where(pair_mask, distance, torch.full_like(distance, float("inf")))
    if exclude_self and n > 0:
        dist = dist.clone()
        dist.fill_diagonal_(float("inf"))

    tie = torch.arange(n, device=dist.device, dtype=dist.dtype)[None, :] * 1e-6
    sortable = torch.where(torch.isfinite(dist), dist + tie, dist)
    k = min(count, n)
    idx = torch.topk(sortable, k=k, dim=-1, largest=False, sorted=True).indices
    valid = torch.gather(dist, dim=1, index=idx) < float("inf")
    idx = torch.where(valid, idx, torch.full_like(idx, -1))
    if k < count:
        pad = torch.full((n, count - k), -1, device=distance.device, dtype=torch.long)
        idx = torch.cat((idx, pad), dim=-1)
    return idx

