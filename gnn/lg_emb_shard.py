"""Embedding row-sharding + NCCL all-gather (--emb-shard, building blocks for N>10M graphs)."""
from __future__ import annotations

import os
import sys
from typing import List

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import qwen3_emb_path  # noqa: E402


def init_process_group(rank: int, world_size: int, master_addr="127.0.0.1",
                       master_port: str = "29501"):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_local_shard(dataset: str, rank: int, world_size: int,
                     fp16: bool = True, device=None):
    """Load only this rank's row-shard ``[start:end]`` of the embedding to GPU."""
    obj = torch.load(qwen3_emb_path(dataset), weights_only=False, map_location="cpu")
    emb_full = obj["emb"] if isinstance(obj, dict) else obj
    N = emb_full.shape[0]
    rows_per = (N + world_size - 1) // world_size
    start = rank * rows_per
    end = min(start + rows_per, N)
    shard = emb_full[start:end].clone()
    if fp16: shard = shard.half()
    else:    shard = shard.float()
    if device is not None: shard = shard.to(device)
    return shard, start, end, N


def owner_rank(global_ids: torch.Tensor, world_size: int, N: int) -> torch.Tensor:
    """Compute the owner rank for each global node id."""
    rows_per = (N + world_size - 1) // world_size
    return torch.clamp(global_ids // rows_per, max=world_size - 1)


def all_gather_rows(global_ids: torch.Tensor,
                    local_shard: torch.Tensor,
                    local_start: int, local_end: int,
                    world_size: int, N: int) -> torch.Tensor:
    """Two all-to-all collectives: route requests -> owners -> requested rows back."""
    device = local_shard.device
    D = local_shard.shape[1]
    owner = owner_rank(global_ids, world_size, N)

    # Sort by owner for an efficient bucketed all-to-all.
    order = torch.argsort(owner)
    sorted_ids = global_ids[order]
    sorted_owner = owner[order]

    out_bucket_sizes = torch.bincount(sorted_owner, minlength=world_size).to(device)

    # Exchange bucket sizes so each rank knows how much it will receive.
    in_bucket_sizes = torch.empty(world_size, dtype=torch.int64, device=device)
    dist.all_to_all_single(in_bucket_sizes, out_bucket_sizes)

    # First all-to-all: send the requested IDs to their owner ranks.
    incoming = torch.empty(int(in_bucket_sizes.sum().item()),
                           dtype=torch.int64, device=device)
    dist.all_to_all_single(
        incoming, sorted_ids,
        output_split_sizes=in_bucket_sizes.tolist(),
        input_split_sizes=out_bucket_sizes.tolist(),
    )

    # Each rank looks up its locally-owned rows.
    local_rows = local_shard[(incoming - local_start)]  # [sum_in, D]

    # Second all-to-all: send the requested rows back.
    out_rows = torch.empty((int(out_bucket_sizes.sum().item()), D),
                           dtype=local_shard.dtype, device=device)
    dist.all_to_all_single(
        out_rows, local_rows.contiguous(),
        output_split_sizes=(out_bucket_sizes * D).tolist(),
        input_split_sizes=(in_bucket_sizes * D).tolist(),
    )
    out_rows = out_rows.view(-1, D)

    # Restore the original (pre-sort) ordering.
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=device)
    return out_rows[inv]
