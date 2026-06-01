from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class MotifGNN(nn.Module):
    def __init__(self,
                 in_dim: int = 4096,
                 shared_dim: int = 256,
                 hidden_channels: int = 256,
                 out_channels: int = 128,
                 motif_names=None,
                 tau: float = 0.4):
        super().__init__()

        if motif_names is None:
            motif_names = ["edge", "triangle", "4-cycle", "4-clique"]
        self.motif_names = list(motif_names)
        self.motif_count = len(self.motif_names)

        self.shared_proj = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.conv1s = nn.ModuleList([
            SAGEConv(shared_dim, hidden_channels)
            for _ in range(self.motif_count)
        ])
        self.conv2s = nn.ModuleList([
            SAGEConv(hidden_channels, out_channels)
            for _ in range(self.motif_count)
        ])
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(self.motif_count)
        ])

        # learnable motif fusion weights
        self.motif_weights = nn.Parameter(torch.randn(self.motif_count))

        # GRACE projection head (operates on `out_channels`)
        self.fc1 = nn.Linear(out_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.tau = tau

    def forward(self, data, motif_adj_matrices):
        x = self.shared_proj(data.x)

        out = torch.zeros(
            x.size(0),
            self.conv2s[0].out_channels,
            device=x.device,
            dtype=x.dtype,
        )

        for i, motif in enumerate(self.motif_names):
            edge_index = motif_adj_matrices[motif]
            h = self.conv1s[i](x, edge_index)
            h = F.relu(h)
            h = self.batch_norm[i](h)
            h = self.conv2s[i](h, edge_index)
            out = out + self.motif_weights[i] * h

        return out

    @staticmethod
    def sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        return z1 @ z2.t()

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        f = lambda x: torch.exp(x / self.tau)
        refl    = f(self.sim(z1, z1))
        between = f(self.sim(z1, z2))
        return -torch.log(
            between.diag() /
            (refl.sum(1) + between.sum(1) - refl.diag())
        )

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int) -> torch.Tensor:
        device = z1.device
        n = z1.size(0)
        n_batches = (n - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(n, device=device)
        out = []
        for i in range(n_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl    = f(self.sim(z1[mask], z1))            # [B, N]
            between = f(self.sim(z1[mask], z2))            # [B, N]
            sl = slice(i * batch_size, (i + 1) * batch_size)
            out.append(
                -torch.log(
                    between[:, sl].diag() /
                    (refl.sum(1) + between.sum(1) - refl[:, sl].diag())
                )
            )
        return torch.cat(out)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0) -> torch.Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        ret = (l1 + l2) * 0.5
        return ret.mean() if mean else ret.sum()


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    drop_mask = torch.empty(
        (x.size(1),), dtype=torch.float32, device=x.device,
    ).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x
