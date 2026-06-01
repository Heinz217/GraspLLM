"""Center-level batching: process BC centers concurrently per GPU (--center-batch K)."""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


# 3D variant of the per-step neighbour gather; cand_3d=[BC,B,C], output [BC,B,C,K].
@torch.no_grad()
def _gather_nb_matrix_3d(cand_3d: torch.Tensor, valid_3d: torch.Tensor,
                        nb_flat: torch.Tensor, ptr: torch.Tensor,
                        deg: torch.Tensor, max_k: int, gen):
    BC, B, C = cand_3d.shape
    deg_c = deg[cand_3d]
    take = torch.clamp(deg_c, max=max_k)
    rand = torch.rand((BC, B, C, max_k), device=cand_3d.device, generator=gen)
    denom = torch.clamp(deg_c, min=1).unsqueeze(-1).float()
    offs = (rand * denom).long()
    flat = ptr[cand_3d].unsqueeze(-1) + offs
    nb_idx = nb_flat[flat]
    arange_k = torch.arange(max_k, device=cand_3d.device).view(1, 1, 1, max_k)
    valid_k = (arange_k < take.unsqueeze(-1)) & valid_3d.unsqueeze(-1)
    nb_idx = torch.where(valid_k, nb_idx,
                         cand_3d.unsqueeze(-1).expand(-1, -1, -1, max_k))
    return nb_idx, valid_k, take * valid_3d.long()


@torch.no_grad()
def generate_ocs_super_batch(
    centers: List[int],
    embeddings: torch.Tensor,
    nb_flat: torch.Tensor,
    ptr: torch.Tensor,
    deg: torch.Tensor,
    N: int,
    *,
    threshold: float = 0.0,
    max_steps: int = 10,
    num_neighbors: int = 9,
    max_elements: int = 111,
    beta: float = 0.55,
    max_k: int = 10,
    c_max: int = 256,
    device=None,
    gen=None,
    buf_visited=None,
    buf_cand=None,
):
    """Sample BC centers in a single batched forward pass; returns one sequence per center."""
    if device is None: device = embeddings.device
    BC = len(centers)
    B = num_neighbors

    if buf_visited is None:
        buf_visited = torch.zeros(BC, B, N, dtype=torch.bool, device=device)
    if buf_cand is None:
        buf_cand = torch.zeros(BC, B, N, dtype=torch.bool, device=device)

    sequences = [[int(c)] for c in centers]
    centers_t = torch.tensor(centers, dtype=torch.long, device=device)

    # 9 starting neighbours per center (padded to B).
    cur = torch.full((BC, B), 0, dtype=torch.long, device=device)
    active = torch.zeros(BC, B, dtype=torch.bool, device=device)
    n_walks_per = torch.zeros(BC, dtype=torch.long, device=device)

    for ci, c in enumerate(centers):
        lo, hi = ptr[c], ptr[c + 1]
        row = nb_flat[lo:hi]
        if row.numel() == 0:
            sequences[ci].extend([-500] * (max_elements - 1))
            continue
        if row.numel() > num_neighbors:
            idx = torch.randperm(row.numel(), device=device, generator=gen)[:num_neighbors]
            row = row[idx]
        nbs = row
        nw = nbs.shape[0]
        n_walks_per[ci] = nw
        sequences[ci].extend(nbs.tolist())
        sequences[ci].extend([-500] * (11 - len(sequences[ci])))
        cur[ci, :nw] = nbs
        cur[ci, nw:] = c
        active[ci, :nw] = True

    buf_visited.zero_()
    buf_cand.zero_()
    cv = embeddings[centers_t]  # [BC, D]

    walks = torch.full((BC, B, max_steps), -500, dtype=torch.long, device=device)
    arange_BC = torch.arange(BC, device=device).unsqueeze(1).expand(BC, B)
    arange_B  = torch.arange(B, device=device).unsqueeze(0).expand(BC, B)

    for step in range(max_steps):
        walks[:, :, step] = torch.where(
            active, cur, torch.tensor(-500, device=device))

        buf_visited[arange_BC.flatten(),
                    arange_B.flatten(),
                    cur.flatten()] = True

        # Python loop over BC*B (=36 default) is cheap relative to the matmuls.
        for ci in range(BC):
            for b in range(B):
                if not bool(active[ci, b]):
                    continue
                v = int(cur[ci, b].item())
                buf_cand[ci, b, nb_flat[ptr[v]:ptr[v + 1]]] = True

        cand_eff = buf_cand & ~buf_visited
        if not cand_eff.any():
            break

        # Top-c_max via random-priority topk on the flattened [BC*B, N] tensor.
        flat_eff = cand_eff.view(BC * B, N)
        prio = torch.where(flat_eff,
                           torch.rand(BC * B, N, device=device, generator=gen),
                           torch.full((1, 1), -1.0, device=device).expand(BC * B, N))
        topv, topi = torch.topk(prio, k=min(c_max, N), dim=1)
        valid = (topv > 0).view(BC, B, -1) & active.unsqueeze(-1)
        cand_3d = topi.view(BC, B, -1)
        if not valid.any():
            break

        cand_vecs = embeddings[cand_3d]  # [BC, B, c_max, D]
        cv_exp = cv.unsqueeze(1).unsqueeze(2).expand_as(cand_vecs)
        rel = torch.clamp(F.cosine_similarity(cand_vecs, cv_exp, dim=-1), min=0)

        nb_idx, vmask, take = _gather_nb_matrix_3d(
            cand_3d, valid, nb_flat, ptr, deg, max_k, gen)
        nb_vecs = embeddings[nb_idx]  # [BC, B, c_max, K, D]
        cos = F.cosine_similarity(
            cand_vecs.unsqueeze(3).expand_as(nb_vecs), nb_vecs, dim=-1)
        cos = torch.clamp(cos, min=0) * vmask.float()
        struct = cos.sum(dim=-1) * (take.float() / max_k)

        ctx = beta * struct + (1 - beta) * rel
        NEG = torch.finfo(ctx.dtype).min / 4
        ctx = torch.where(valid, ctx,
                          torch.tensor(NEG, device=device, dtype=ctx.dtype))
        eta = torch.softmax(ctx, dim=-1)
        valid_eta = eta > threshold
        valid_any = valid_eta.any(dim=-1) & valid.any(dim=-1)
        new_active = active & valid_any
        if not new_active.any():
            break
        best = torch.argmax(eta * valid_eta.float(), dim=-1)
        next_cur = cand_3d.gather(2, best.unsqueeze(-1)).squeeze(-1)
        cur = torch.where(new_active, next_cur, cur)
        active = new_active

    walks_cpu = walks.cpu().tolist()
    n_walks_cpu = n_walks_per.cpu().tolist()
    for ci in range(BC):
        nw = n_walks_cpu[ci]
        for b in range(nw):
            sequences[ci].extend(walks_cpu[ci][b])
        for _ in range(num_neighbors - nw):
            sequences[ci].extend([-500] * max_steps)
        sequences[ci].extend([-500] * (max_elements - len(sequences[ci])))
        sequences[ci] = sequences[ci][:max_elements]
    return sequences
