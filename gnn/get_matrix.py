import argparse
import os
import random
import sys
import numpy as np
import networkx as nx
from collections import defaultdict

import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph

# allow `python gnn/train.py` (which imports get_matrix) to find utils/paths.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import processed_data_path, qwen3_emb_path  # noqa: E402


def load_features_and_labels(dataset_name):
    data = torch.load(processed_data_path(dataset_name), weights_only=False)
    emb_obj = torch.load(qwen3_emb_path(dataset_name), weights_only=False)
    features = emb_obj["emb"] if isinstance(emb_obj, dict) else emb_obj
    features = features.float()  # fp16 → fp32 for downstream math
    labels = data.y
    return features, labels, data.edge_index


def load_data(dataset):
    features, labels, edge_index = load_features_and_labels(dataset)
    return features, edge_index


def load_labels(dataset):
    _, labels, _ = load_features_and_labels(dataset)
    return labels


import torch
import networkx as nx

def compute_motifs_for_subgraph(adj_list):
    G = nx.Graph()
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    motif_adj = {}

    # Edge motif
    edge_list = []
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            edge_list.append([node, neighbor])
    motif_adj["edge"] = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Triangle motif
    triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
    triangle_edges = []
    for triangle in triangles:
        for i in triangle:
            for j in triangle:
                if i != j:
                    triangle_edges.append([i, j])
    motif_adj["triangle"] = torch.tensor(triangle_edges, dtype=torch.long).t().contiguous() if triangle_edges else torch.zeros((2, 0), dtype=torch.long)

    # 4-cycle motif
    cycles = [cycle for cycle in nx.cycle_basis(G) if len(cycle) == 4]
    cycle_edges = []
    for cycle in cycles:
        for i in cycle:
            for j in cycle:
                if i != j:
                    cycle_edges.append([i, j])
    motif_adj["4-cycle"] = torch.tensor(cycle_edges, dtype=torch.long).t().contiguous() if cycle_edges else torch.zeros((2, 0), dtype=torch.long)

    # 4-clique motif
    cliques = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 4]
    clique_edges = []
    for clique in cliques:
        for i in clique:
            for j in clique:
                if i != j:
                    clique_edges.append([i, j])
    motif_adj["4-clique"] = torch.tensor(clique_edges, dtype=torch.long).t().contiguous() if clique_edges else torch.zeros((2, 0), dtype=torch.long)

    return motif_adj

def convert_edge_index_to_adj_list(edge_index):
    adj_list = defaultdict(set)
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        adj_list[src].add(dst)
        adj_list[dst].add(src)
    return adj_list


def compute_motifs_torch(edge_index, num_nodes, device='cuda'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(torch.device(device))


    motif_adj = {}

    # Edge Motif 
    motif_adj["edge"] = edge_index.to(device)

    # Triangle Motif
    A2 = torch.matmul(A, A)  
    triangle_mask = (A2 > 0) & (A > 0)  
    triangle_edges = torch.nonzero(triangle_mask, as_tuple=False).t()
    motif_adj["triangle"] = triangle_edges if triangle_edges.size(1) > 0 else torch.zeros((2, 0), dtype=torch.long,
                                                                                          device=device)

    # 4-Cycle Motif
    A3 = torch.matmul(A2, A)  
    A4_cycle = (A3 > 0) & (A > 0)  
    cycle_edges = torch.nonzero(A4_cycle, as_tuple=False).t()
    motif_adj["4-cycle"] = cycle_edges if cycle_edges.size(1) > 0 else torch.zeros((2, 0), dtype=torch.long,
                                                                                   device=device)

    # 4-Clique Motif
    cliques_mask = (A2 > 0) & (torch.matmul(A, A2) > 0) 
    clique_edges = torch.nonzero(cliques_mask, as_tuple=False).t()
    motif_adj["4-clique"] = clique_edges if clique_edges.size(1) > 0 else torch.zeros((2, 0), dtype=torch.long,
                                                                                      device=device)

    return motif_adj

def compute_motifs_sparse(edge_index, num_nodes, device='cuda'):
    from torch_sparse import spmm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    motif_adj = {}

    row, col = edge_index
    A_sparse = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1), device=device),
                                       (num_nodes, num_nodes)).to(device)

    # Edge Motif
    motif_adj["edge"] = edge_index

    # Triangle Motif
    A2 = spmm(edge_index, torch.ones(edge_index.size(1), device=device), num_nodes, num_nodes, A_sparse)
    triangle_mask = (A2.to_dense() > 0) & (A_sparse.to_dense() > 0)
    triangle_edges = triangle_mask.nonzero(as_tuple=False).t()
    motif_adj["triangle"] = triangle_edges

    # 4-Cycle Motif
    A3 = spmm(edge_index, torch.ones(edge_index.size(1), device=device), num_nodes, num_nodes, A2)
    cycle_mask = (A3.to_dense() > 0) & (A_sparse.to_dense() > 0)
    cycle_edges = cycle_mask.nonzero(as_tuple=False).t()
    motif_adj["4-cycle"] = cycle_edges

    # 4-Clique Motif
    clique_mask = (A2.to_dense() > 0) & (A3.to_dense() > 0)
    clique_edges = clique_mask.nonzero(as_tuple=False).t()
    motif_adj["4-clique"] = clique_edges

    return motif_adj


def compute_motifs_subgraph(edge_index, num_nodes, num_partitions=10, device='cuda'):
    partition_size = num_nodes // num_partitions
    motif_adj = {"edge": [], "triangle": [], "4-cycle": [], "4-clique": []}

    for i in range(num_partitions):
        start_node = i * partition_size
        end_node = (i + 1) * partition_size if i < num_partitions - 1 else num_nodes

        node_mask = torch.arange(start_node, end_node, device=device)
        sub_edge_index, _ = subgraph(node_mask, edge_index)

        sub_motif_adj = compute_motifs_torch(sub_edge_index, len(node_mask), device=device)

        for motif, edges in sub_motif_adj.items():
            motif_adj[motif].append(edges)

    for motif in motif_adj:
        motif_adj[motif] = torch.cat(motif_adj[motif], dim=1) if len(motif_adj[motif]) > 0 else torch.zeros((2, 0),
                                                                                                            device=device)

    return motif_adj

