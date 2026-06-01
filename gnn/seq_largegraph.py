"""Large-graph adaptation for OCS subgraph sampling (opt-in via --large-graph)."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import qwen3_emb_path  # noqa: E402


# Build a GPU-resident undirected CSR adjacency from a [2, E] edge_index.
def build_csr_gpu(edge_index: torch.Tensor, num_nodes: int, device):
    src = torch.cat([edge_index[0], edge_index[1]]).to(device)
    dst = torch.cat([edge_index[1], edge_index[0]]).to(device)
    order = torch.argsort(src)
    src, dst = src[order], dst[order]
    deg = torch.bincount(src, minlength=num_nodes)
    ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    ptr[1:] = torch.cumsum(deg, dim=0)
    return dst.contiguous(), ptr.contiguous(), deg


# Per-step random K-neighbour gather; cand_2d=[B,C], returns [B,C,K] indices.
@torch.no_grad()
def _gather_nb_matrix_2d(cand_2d, valid_2d, nb_flat, ptr, deg, max_k, gen):
    B, C = cand_2d.shape
    deg_c = deg[cand_2d]
    take = torch.clamp(deg_c, max=max_k)
    rand = torch.rand((B, C, max_k), device=cand_2d.device, generator=gen)
    denom = torch.clamp(deg_c, min=1).unsqueeze(-1).float()
    offs = (rand * denom).long()
    flat = ptr[cand_2d].unsqueeze(-1) + offs
    nb_idx = nb_flat[flat]
    arange_k = torch.arange(max_k, device=cand_2d.device).view(1, 1, max_k)
    valid_k = (arange_k < take.unsqueeze(-1)) & valid_2d.unsqueeze(-1)
    nb_idx = torch.where(valid_k, nb_idx,
                         cand_2d.unsqueeze(-1).expand(-1, -1, max_k))
    return nb_idx, valid_k, take * valid_2d.long()


# Sample one OCS sequence for a single center (B=9 walks fused per step).
@torch.no_grad()
def _generate_ocs_sequence_impl(
    center_node, embeddings, nb_flat, ptr, deg, N,
    threshold, max_steps, num_neighbors, max_elements,
    beta, max_k, c_max, device, gen, buf_visited, buf_cand,
):
    B = num_neighbors
    seq = [int(center_node)]
    lo, hi = ptr[center_node], ptr[center_node + 1]
    row = nb_flat[lo:hi]
    if row.numel() == 0:
        seq.extend([-500] * (max_elements - 1))
        return seq
    if row.numel() > num_neighbors:
        idx = torch.randperm(row.numel(), device=device, generator=gen)[:num_neighbors]
        row = row[idx]
    nbs = row
    n_walks = nbs.shape[0]
    seq.extend(nbs.tolist())
    seq.extend([-500] * (11 - len(seq)))

    buf_visited.zero_()
    buf_cand.zero_()
    cv = embeddings[center_node].unsqueeze(0)

    cur = torch.full((B,), int(center_node), dtype=torch.long, device=device)
    cur[:n_walks] = nbs
    active = torch.zeros(B, dtype=torch.bool, device=device)
    active[:n_walks] = True

    walks = torch.full((B, max_steps), -500, dtype=torch.long, device=device)
    arange_B = torch.arange(B, device=device)

    for step in range(max_steps):
        walks[:, step] = torch.where(active, cur,
                                     torch.tensor(-500, device=device))
        buf_visited[arange_B, cur] = True
        for b in range(B):
            if not bool(active[b]):
                continue
            c = int(cur[b].item())
            buf_cand[b, nb_flat[ptr[c]:ptr[c + 1]]] = True

        cand_eff = buf_cand & ~buf_visited
        if not cand_eff.any():
            break

        # Top-c_max random-priority cap on candidate set per step.
        prio = torch.where(cand_eff,
                           torch.rand(B, N, device=device, generator=gen),
                           torch.full((1, 1), -1.0, device=device).expand(B, N))
        topv, topi = torch.topk(prio, k=min(c_max, N), dim=1)
        valid = (topv > 0) & active.unsqueeze(1)
        if not valid.any():
            break
        cand_2d = topi

        # rel(v,c) = cosine(emb[cand], emb[center]).
        cand_vecs = embeddings[cand_2d]
        rel = torch.clamp(F.cosine_similarity(
            cand_vecs, cv.expand_as(cand_vecs), dim=-1), min=0)

        # struct(v) = mean cosine to K random neighbours (paper definition).
        nb_idx, vmask, take = _gather_nb_matrix_2d(
            cand_2d, valid, nb_flat, ptr, deg, max_k, gen)
        nb_vecs = embeddings[nb_idx]
        cos = F.cosine_similarity(
            cand_vecs.unsqueeze(2).expand(-1, -1, max_k, -1),
            nb_vecs, dim=-1)
        cos = torch.clamp(cos, min=0) * vmask.float()
        struct = cos.sum(dim=-1) * (take.float() / max_k)
        ctx = beta * struct + (1 - beta) * rel
        NEG = torch.finfo(ctx.dtype).min / 4
        ctx = torch.where(valid, ctx,
                          torch.tensor(NEG, device=device, dtype=ctx.dtype))
        eta = torch.softmax(ctx, dim=-1)
        valid_eta = eta > threshold
        valid_any = valid_eta.any(dim=1) & valid.any(dim=1)
        new_active = active & valid_any
        if not new_active.any():
            break
        best = torch.argmax(eta * valid_eta.float(), dim=1)
        next_cur = cand_2d.gather(1, best.unsqueeze(1)).squeeze(1)
        cur = torch.where(new_active, next_cur, cur)
        active = new_active

    walks_cpu = walks.cpu().tolist()
    for b in range(n_walks):
        seq.extend(walks_cpu[b])
    for _ in range(num_neighbors - n_walks):
        seq.extend([-500] * max_steps)

    seq.extend([-500] * (max_elements - len(seq)))
    return seq[:max_elements]


_COMPILED_IMPL = None


def _maybe_compile(fn, enabled: bool):
    if not enabled:
        return fn
    try:
        return torch.compile(fn, dynamic=True, fullgraph=False)
    except Exception as e:
        print(f"[seq-lg] torch.compile disabled ({e})", flush=True)
        return fn


def generate_ocs_sequence(
    center_node: int, embeddings, nb_flat, ptr, deg, N,
    *, threshold=0.0, max_steps=10, num_neighbors=9, max_elements=111,
    beta=0.55, max_k=10, c_max=256, device=None, gen=None,
    buf_visited=None, buf_cand=None, compile_kernel: bool = False,
) -> List[int]:
    """Public single-center entry point used by the seq.py dispatcher."""
    global _COMPILED_IMPL
    if compile_kernel:
        if _COMPILED_IMPL is None:
            _COMPILED_IMPL = _maybe_compile(_generate_ocs_sequence_impl, True)
        impl = _COMPILED_IMPL
    else:
        impl = _generate_ocs_sequence_impl
    return impl(
        center_node, embeddings, nb_flat, ptr, deg, N,
        threshold, max_steps, num_neighbors, max_elements,
        beta, max_k, c_max, device or embeddings.device, gen,
        buf_visited, buf_cand,
    )


# Per-GPU worker for the multi-GPU driver.
def _worker(gpu_id, dataset, edge_index_cpu, centers, fp16, kwargs, q,
            shared_emb_handle=None, center_batch=1, use_compile=False,
            progress_every=1000):
    import torch as _torch
    _torch.cuda.set_device(gpu_id)
    device = _torch.device(f"cuda:{gpu_id}")

    # Mode A: attach to a single shared embedding via CUDA IPC.
    if shared_emb_handle is not None:
        from lg_shared_emb import attach_shared_embedding
        emb = attach_shared_embedding(shared_emb_handle, gpu_id)
        owner_id = shared_emb_handle["owner_id"]
        print(f"[GPU{gpu_id}] attached to shared embedding owned by GPU{owner_id} "
              f"shape={tuple(emb.shape)} dtype={emb.dtype}", flush=True)
    else:
        # Mode B: each worker loads its own full embedding replica.
        obj = _torch.load(qwen3_emb_path(dataset), weights_only=False)
        emb = obj["emb"] if isinstance(obj, dict) else obj
        emb = (emb.half() if fp16 else emb.float()).to(device)
    N = emb.shape[0]

    nb_flat, ptr, deg = build_csr_gpu(edge_index_cpu, N, device)
    _torch.cuda.synchronize(device)

    seed = kwargs.get("seed", 42) + gpu_id
    gen = _torch.Generator(device=device).manual_seed(seed)
    B = kwargs.get("num_neighbors", 9)

    out = []
    t0 = time.time()
    if center_batch > 1:
        # Super-batch BC centers per forward pass (shape grows by leading BC dim).
        from lg_center_batch import generate_ocs_super_batch
        bv = _torch.zeros(center_batch, B, N, dtype=_torch.bool, device=device)
        bc = _torch.zeros(center_batch, B, N, dtype=_torch.bool, device=device)
        for i in range(0, len(centers), center_batch):
            chunk = centers[i:i + center_batch]
            actual = len(chunk)
            if actual < center_batch:
                bv_eff = bv[:actual]; bc_eff = bc[:actual]
            else:
                bv_eff = bv; bc_eff = bc
            seqs = generate_ocs_super_batch(
                [int(c) for c in chunk], emb, nb_flat, ptr, deg, N,
                device=device, gen=gen,
                buf_visited=bv_eff, buf_cand=bc_eff,
                **{k: v for k, v in kwargs.items() if k != "seed"},
            )
            for cid, seq in zip(chunk, seqs):
                out.append((int(cid), seq))
            if (len(out)) % progress_every == 0:
                rate = len(out) / (time.time() - t0)
                eta = (len(centers) - len(out)) / max(rate, 1e-6) / 60
                print(f"[GPU{gpu_id}] {len(out)}/{len(centers)}  "
                      f"rate={rate:.1f}/s  ETA={eta:.1f}min  (BC={center_batch})",
                      flush=True)
    else:
        bv = _torch.zeros(B, N, dtype=_torch.bool, device=device)
        bc = _torch.zeros(B, N, dtype=_torch.bool, device=device)
        for i, c in enumerate(centers):
            seq = generate_ocs_sequence(
                int(c), emb, nb_flat, ptr, deg, N,
                device=device, gen=gen, buf_visited=bv, buf_cand=bc,
                compile_kernel=use_compile,
                **{k: v for k, v in kwargs.items() if k != "seed"},
            )
            out.append((int(c), seq))
            if (i + 1) % progress_every == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(centers) - i - 1) / max(rate, 1e-6) / 60
                print(f"[GPU{gpu_id}] {i+1}/{len(centers)}  "
                      f"rate={rate:.1f}/s  ETA={eta:.1f}min", flush=True)
    q.put((gpu_id, out))


def compute_ocs_sequences_multi_gpu(
    dataset: str,
    edge_index: torch.Tensor,
    centers: List[int],
    *,
    gpus: List[int],
    fp16: bool = True,
    shared_emb: bool = False,
    emb_shard: bool = False,
    center_batch: int = 1,
    use_compile: bool = False,
    **kwargs,
) -> dict:
    """Multi-GPU OCS sampling with optional advanced features (shared_emb / emb_shard / center_batch / compile)."""
    if shared_emb and emb_shard:
        raise ValueError("--shared-emb and --emb-shard are mutually exclusive")

    centers = list(centers)
    chunks = [centers[i::len(gpus)] for i in range(len(gpus))]
    edge_index_cpu = edge_index.detach().cpu()

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()

    handle_ctx = None
    handle = None
    if shared_emb:
        from lg_shared_emb import SharedEmbeddingHandle
        handle_ctx = SharedEmbeddingHandle(dataset, gpus, fp16=fp16)
        handle = handle_ctx.__enter__()

    if emb_shard:
        # Building blocks live in lg_emb_shard.py; per-step NCCL all-gather not yet wired in.
        raise NotImplementedError(
            "--emb-shard provides the building blocks (lg_emb_shard.py) but "
            "the per-step NCCL all-gather is not yet wired into the inner loop. "
            "For graphs <= ~5M nodes, prefer --shared-emb."
        )

    procs = []
    for k, gid in enumerate(gpus):
        p = mp.Process(
            target=_worker,
            args=(gid, dataset, edge_index_cpu, chunks[k], fp16, kwargs, q,
                  handle, center_batch, use_compile),
        )
        p.start(); procs.append(p)

    results = {}
    for _ in procs:
        gpu_id, out = q.get()
        for cid, seq in out:
            results[cid] = seq
    for p in procs:
        p.join()

    if handle_ctx is not None:
        handle_ctx.__exit__(None, None, None)
    return results
