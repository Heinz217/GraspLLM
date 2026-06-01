from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge

# allow `python gnn/train.py` to find utils/paths.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn import MotifGNN, drop_feature  # local import
from get_matrix import (load_data, convert_edge_index_to_adj_list,
                        compute_motifs_torch, compute_motifs_subgraph)
from utils.paths import CHECKPOINT_ROOT  # noqa: E402


def graph_sampling(data, num_samples, method="n-hop", n_hop=2, walk_length=10):
    device = data.x.device
    num_nodes = data.num_nodes

    if method == "probability":
        probs = torch.rand(num_nodes, device=device)
        sampled_nodes = torch.topk(probs, num_samples).indices

    elif method == "n-hop":
        start_nodes = torch.randint(0, num_nodes, (num_samples // n_hop,), device=device)
        visited = set(start_nodes.tolist())
        queue = list(start_nodes.tolist())
        while queue and len(visited) < num_samples:
            node = queue.pop(0)
            neighbors = data.edge_index[1][data.edge_index[0] == node]
            for nb in neighbors:
                if nb.item() not in visited and len(visited) < num_samples:
                    visited.add(nb.item())
                    queue.append(nb.item())
        sampled_nodes = torch.tensor(list(visited), device=device)

    elif method == "random-walk":
        sampled_nodes = set()
        for _ in range(num_samples):
            current = torch.randint(0, num_nodes, (1,), device=device).item()
            for _ in range(walk_length):
                neighbors = data.edge_index[1][data.edge_index[0] == current]
                if len(neighbors) == 0:
                    break
                current = neighbors[torch.randint(0, len(neighbors), (1,)).item()].item()
                sampled_nodes.add(current)
        sampled_nodes = torch.tensor(list(sampled_nodes), device=device)
    else:
        raise ValueError(f"Invalid sampling method: {method}")

    sampled_nodes = sampled_nodes[:num_samples]
    id_map = {old.item(): new for new, old in enumerate(sampled_nodes)}

    mask = (torch.isin(data.edge_index[0], sampled_nodes)
            & torch.isin(data.edge_index[1], sampled_nodes))
    sub_ei = data.edge_index[:, mask]
    new_ei = torch.stack([
        torch.tensor([id_map[i.item()] for i in sub_ei[0]]),
        torch.tensor([id_map[i.item()] for i in sub_ei[1]]),
    ], dim=0).to(device)

    return Data(x=data.x[sampled_nodes], edge_index=new_ei), sampled_nodes, id_map


def generate_motif_views(motif_adj, drop_rate_1: float, drop_rate_2: float):
    motif_adj_1, motif_adj_2 = {}, {}
    for motif, edge_index in motif_adj.items():
        ei1, _ = dropout_edge(edge_index, p=drop_rate_1)
        ei2, _ = dropout_edge(edge_index, p=drop_rate_2)
        motif_adj_1[motif] = ei1
        motif_adj_2[motif] = ei2
    return motif_adj_1, motif_adj_2


def train_model_multi_dataset(datasets,
                              num_epochs: int,
                              lr: float,
                              device,
                              num_samples: int,
                              sampling_method: str,
                              n_hop: int,
                              shared_dim: int,
                              hidden_channels: int,
                              out_channels: int,
                              tau: float):
    all_motifs = ["edge", "triangle", "4-cycle", "4-clique"]

    # Load all sources once.  All features are 4096-d Qwen3-Embedding-8B vectors.
    dataset_data = {}
    in_dim = None
    for name in datasets.keys():
        x, edge_index = load_data(name)
        # Qwen3 embeddings are already L2-normed unit vectors — do NOT re-normalize.
        x = x.to(device)
        edge_index = edge_index.to(device)
        dataset_data[name] = (x, edge_index)
        if in_dim is None:
            in_dim = x.size(1)
        else:
            assert in_dim == x.size(1), (
                f"feature dim mismatch: {name}={x.size(1)} vs {in_dim}")
        print(f"  [load] {name:<12s}  N={x.size(0):>7d}  "
              f"E={edge_index.size(1):>8d}  dim={x.size(1)}")
    print(f"[init] shared feature dim = {in_dim}")
    print(f"[init] datasets = {list(datasets.keys())}")

    model = MotifGNN(
        in_dim=in_dim,
        shared_dim=shared_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        motif_names=all_motifs,
        tau=tau,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for name in datasets.keys():
            optimizer.zero_grad()
            x, edge_index = dataset_data[name]
            data = Data(x=x, edge_index=edge_index).to(device)

            sampled_data, _, _ = graph_sampling(
                data, num_samples, method=sampling_method, n_hop=n_hop)
            sampled_data = sampled_data.to(device)

            motif_adj = compute_motifs_torch(
                sampled_data.edge_index, num_samples)
            for m in all_motifs:
                if m not in motif_adj:
                    motif_adj[m] = torch.zeros(
                        (2, 0), device=device, dtype=torch.long)
                else:
                    motif_adj[m] = motif_adj[m].to(device)

            motif_adj_1, motif_adj_2 = generate_motif_views(
                motif_adj, drop_rate_1=0.4, drop_rate_2=0.2)
            x1 = drop_feature(sampled_data.x, 0.3)
            x2 = drop_feature(sampled_data.x, 0.4)
            data1 = Data(x=x1, edge_index=sampled_data.edge_index).to(device)
            data2 = Data(x=x2, edge_index=sampled_data.edge_index).to(device)

            z1 = model(data1, motif_adj_1)
            z2 = model(data2, motif_adj_2)
            loss = model.loss(z1, z2, batch_size=0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(1, len(datasets))
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch + 1:>4d}/{num_epochs}  "
                  f"avg_loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs}  avg_loss={avg:.4f}")

    return model


def process_new_dataset(model, new_dataset_name, device):
    x, edge_index = load_data(new_dataset_name)
    x = x.to(device)
    edge_index = edge_index.to(device)

    num_nodes = x.size(0)
    num_partitions = max(10, num_nodes // 20000)
    motif_adj = compute_motifs_subgraph(edge_index, num_nodes,
                                        num_partitions=num_partitions,
                                        device=device)
    for m in model.motif_names:
        if m not in motif_adj:
            motif_adj[m] = torch.zeros((2, 0), device=device, dtype=torch.long)
        else:
            motif_adj[m] = motif_adj[m].to(device)

    data = Data(x=x, edge_index=edge_index).to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model(data, motif_adj)
    return embeddings


def parse_args():
    p = argparse.ArgumentParser(description="Train cross-dataset motif GNN (Stage 1)")
    p.add_argument('--datasets', nargs='+',
                   default=['arxiv', 'pubmed', 'computer', 'history', 'reddit'],
                   help='Stage-1 source datasets (default: revision-set; see gnn/train.sh).')
    p.add_argument('--samples-per-dataset', type=int, default=60)
    p.add_argument('--num-epochs',  type=int, default=300)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--num-samples', type=int, default=2000,
                   help='Number of nodes to subsample per dataset per step.')
    p.add_argument('--sampling-method', type=str, default='n-hop',
                   choices=['n-hop', 'probability', 'random-walk'])
    p.add_argument('--n-hop',          type=int, default=2)
    p.add_argument('--shared-dim',      type=int, default=256)
    p.add_argument('--hidden-channels', type=int, default=256)
    p.add_argument('--out-channels',    type=int, default=128)
    p.add_argument('--tau',  type=float, default=0.4)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--model-save-path', type=str,
                   default=os.path.join(CHECKPOINT_ROOT, 'structure_learner_qwen3.pth'))
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device if (torch.cuda.is_available() and args.device == 'cuda') else 'cpu'
    print(f"[init] device={device}")

    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    datasets = {name: args.samples_per_dataset for name in args.datasets}
    model = train_model_multi_dataset(
        datasets=datasets,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=device,
        num_samples=args.num_samples,
        sampling_method=args.sampling_method,
        n_hop=args.n_hop,
        shared_dim=args.shared_dim,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        tau=args.tau,
    )

    torch.save({'model_state_dict': model.state_dict(),
                'args': vars(args)}, args.model_save_path)
    print(f"[save] -> {args.model_save_path}")
