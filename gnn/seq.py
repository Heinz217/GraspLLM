import numpy as np
import json
import pyhocon
import argparse
import os
import random
import sys
import time
import ot
from collections import Counter
from collections import defaultdict

import torch
import torch.nn.functional as F


from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn import MotifGNN  
from get_matrix import load_data, compute_motifs_for_subgraph, convert_edge_index_to_adj_list, load_labels, \
    compute_motifs_torch, compute_motifs_sparse, compute_motifs_subgraph
from gen import * 
from utils.paths import (dataset_dir, processed_data_path, 
                         CHECKPOINT_ROOT)

def cosine_similarity(vec1, vec2):
    dot_product = torch.dot(vec1, vec2)
    norm_a = torch.norm(vec1)
    norm_b = torch.norm(vec2)
    return dot_product / (norm_a * norm_b + 1e-8) 


def sigmoid(x):
    return 1 / (1 + torch.exp(-x)) 


def cosine_similarity_batch(vec1, vec2):
    vec1_norm = vec1 / vec1.norm(dim=-1, keepdim=True)  
    vec2_norm = vec2 / vec2.norm(dim=-1, keepdim=True) 
    return torch.matmul(vec1_norm, vec2_norm.T)

def greedy_search_no_revisit(
        start_node, embeddings, adjacency_dict, threshold, max_steps, device, center_node, beta=0.55
):
    sequence = []
    visited = set()
    S = set()

    center_vec = embeddings[center_node].unsqueeze(0)  

    current_node = start_node
    steps = 0

    while steps < max_steps:
        sequence.append(current_node)
        visited.add(current_node)
        S.add(current_node)
        steps += 1

        candidate_nodes = set()
        for s in S:
            if s in adjacency_dict:
                candidate_nodes.update(adjacency_dict[s])
        candidate_nodes.difference_update(visited)

        if not candidate_nodes:
            break

        candidate_nodes = list(candidate_nodes)
        candidate_tensor = torch.tensor(candidate_nodes, device=device)
        candidate_vecs = embeddings[candidate_tensor]  # [num_cand, dim]

        # ΔRel
        rel_scores = torch.clamp(
            torch.nn.functional.cosine_similarity(candidate_vecs, center_vec),
            min=0
        )  # [num_cand]

        # ΔCoh
        max_neighbors = 10
        struct_scores = torch.zeros(len(candidate_nodes), device=device)

        for i, node in enumerate(candidate_nodes):
            neighbors = list(adjacency_dict.get(node, set()))

            if len(neighbors) > max_neighbors:
                neighbors = random.sample(neighbors, max_neighbors) 

            if len(neighbors) == 0:
                continue

            neighbor_vecs = embeddings[torch.tensor(neighbors, device=device)]  # [s, dim]
            node_vec = candidate_vecs[i].unsqueeze(0).expand(len(neighbors), -1)  # [s, dim]

            cos_sim = torch.clamp(
                torch.nn.functional.cosine_similarity(node_vec, neighbor_vecs),
                min=0
            )
            struct_scores[i] = torch.sum(cos_sim) * (len(neighbors) / max_neighbors if max_neighbors > 0 else 1)

        # -------- η(v|S) --------
        contextual_scores = beta * struct_scores + (1 - beta) * rel_scores
        shifted_scores = contextual_scores - torch.max(contextual_scores)
        exp_scores = torch.exp(shifted_scores)
        eta_scores = exp_scores / (torch.sum(exp_scores) + 1e-10)

        valid_mask = eta_scores > threshold
        if not valid_mask.any():
            break

        best_index = torch.argmax(eta_scores * valid_mask.float())
        current_node = candidate_nodes[best_index.item()]

    sequence.extend([-500] * (max_steps - len(sequence)))

    return sequence

def build_adjacency_dict(edge_list):
    adjacency_dict = {}

    if isinstance(edge_list, torch.Tensor):
        u_list = edge_list[0].tolist()
        v_list = edge_list[1].tolist()
        edges = zip(u_list, v_list)
    else:
        edges = edge_list

    for u, v in edges:
        if u not in adjacency_dict:
            adjacency_dict[u] = set()
        if v not in adjacency_dict:
            adjacency_dict[v] = set()
        adjacency_dict[u].add(v)
        adjacency_dict[v].add(u)

    return adjacency_dict


def get_neighbors(node, adjacency_dict):
    return list(adjacency_dict.get(node, set()))


def get_neighbors_from_edge_list(node, edge_list, device):
    neighbors = []

    if isinstance(edge_list, torch.Tensor):
        src_mask = edge_list[0] == node
        tgt_mask = edge_list[1] == node

        neighbors_from_src = edge_list[1][src_mask].tolist()
        neighbors_from_tgt = edge_list[0][tgt_mask].tolist()

        neighbors = neighbors_from_src + neighbors_from_tgt
    else:
        for u, v in edge_list:
            if u == node:
                neighbors.append(v)
            elif v == node:
                neighbors.append(u)

    return neighbors

def generate_final_sequence(center_node, embeddings, adjacency_dict, threshold=0.3, max_steps=10, num_neighbors=9,
                           max_elements=111, device="cuda", beta=0.55):
    sequence = [center_node]

    neighbors = get_neighbors(center_node, adjacency_dict)
    neighbors_tensor = torch.tensor(neighbors, device=device)

    if len(neighbors_tensor) > num_neighbors:
        selected_indices = torch.randperm(len(neighbors_tensor))[:num_neighbors]
        neighbors = neighbors_tensor[selected_indices].tolist()
    else:
        neighbors = neighbors_tensor.tolist()

    sequence.extend(neighbors)
    sequence.extend([-500] * (11 - len(sequence)))  
    for neighbor in neighbors:
        neighbor_sequence = greedy_search_no_revisit(neighbor, embeddings, adjacency_dict, threshold, max_steps, device, center_node, beta=0.55)
        sequence.extend(neighbor_sequence)

    sequence.extend([-500] * (max_elements - len(sequence)))

    return sequence



def merge_motif_adjacency(motif_adjs, num_nodes):
    merged_adj = defaultdict(set)

    for motif, adj_matrix in motif_adjs.items():
        for src, tgt in zip(adj_matrix[0], adj_matrix[1]):
            src = src.item()
            tgt = tgt.item()
            merged_adj[src].add(tgt)
            merged_adj[tgt].add(src)

    final_adj = defaultdict(set, {i: merged_adj[i] for i in range(num_nodes)})

    return final_adj


def compute_wasserstein_distance(dist1, dist2):
    dist1 = dist1.cpu().numpy()
    dist2 = dist2.cpu().numpy()

    M = ot.dist(dist1, dist2)

    a = np.ones(len(dist1)) / len(dist1)
    b = np.ones(len(dist2)) / len(dist2)

    return ot.emd2(a, b, M)


def find_closest_dataset(new_features, reference_features):
    distances = {}
    for dataset_name, features in reference_features.items():
        dist = compute_wasserstein_distance(new_features, features)
        distances[dataset_name] = dist

    return min(distances.items(), key=lambda x: x[1])[0]


def find_closest_dataset_with_sampling(new_features, reference_features, sample_size, num_samples):
    votes = []

    for _ in range(num_samples):
        sampled_indices = torch.randperm(len(new_features))[:sample_size]
        sampled_new_features = new_features[sampled_indices]

        distances = {}
        for dataset_name, features in reference_features.items():
            sampled_ref_indices = torch.randperm(len(features))[:sample_size]
            sampled_ref_features = features[sampled_ref_indices]
            dist = compute_wasserstein_distance(sampled_new_features, sampled_ref_features)
            distances[dataset_name] = dist

        closest_dataset = min(distances.items(), key=lambda x: x[1])[0]
        votes.append(closest_dataset)

    most_common_dataset = Counter(votes).most_common(1)[0][0]
    return most_common_dataset

config = {
    # Stage-1 sources (see gnn/train.sh for rationale).
    "datasets": ["arxiv", "pubmed", "computer", "history", "reddit"],
    "samples_per_dataset": 60,
    "num_epochs": 1,
    "learning_rate": 0.0001,
    "num_samples": 2000,
    "sampling_method": "n-hop",
    "n_hop": 2,
    "shared_dim": 256,
    "hidden_channels": 256,
    "out_channels": 128,
    "tau": 0.4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": os.path.join(CHECKPOINT_ROOT,
                                    "structure_learner_qwen3.pth"),
}


def process_new_dataset(model, new_dataset_name, device):
    """Run the trained shared GNN on a new dataset.

    Qwen3-Embedding-8B gives a unified feature space across datasets, so we
    simply apply the trained model directly.
    """
    node_features, edge_index = load_data(new_dataset_name)  # qwen3_emb fp32, edge_index
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)

    num_nodes = node_features.size(0)
    if new_dataset_name == "cora":
        motif_adj = compute_motifs_torch(edge_index, num_nodes)
    else:
        num_partitions = max(10, num_nodes // 20000)
        motif_adj = compute_motifs_subgraph(edge_index, num_nodes,
                                            num_partitions=num_partitions,
                                            device=device)
    motif_adj = {k: v.to(device) for k, v in motif_adj.items()}
    for motif in model.motif_names:
        if motif not in motif_adj:
            motif_adj[motif] = torch.zeros((2, 0), device=device, dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index).to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model(data, motif_adj)
    return embeddings, motif_adj


def load_model(cfg):
    """Load the shared MotifGNN checkpoint trained in Stage 1."""
    motif_names = ["edge", "triangle", "4-cycle", "4-clique"]
    model = MotifGNN(
        in_dim=4096,                # Qwen3-Embedding-8B
        shared_dim=cfg["shared_dim"],
        hidden_channels=cfg["hidden_channels"],
        out_channels=cfg["out_channels"],
        motif_names=motif_names,
        tau=cfg["tau"],
    )
    ckpt = torch.load(cfg["model_save_path"],
                      map_location=cfg["device"], weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(cfg["device"])
    model.eval()
    return model


def merge_motif_adjacency(motif_adjs, num_nodes):
    merged_adj = defaultdict(set)

    for motif, adj_matrix in motif_adjs.items():
        for src, tgt in zip(adj_matrix[0], adj_matrix[1]):
            src = src.item()
            tgt = tgt.item()
            merged_adj[src].add(tgt)
            merged_adj[tgt].add(src)

    final_adj = defaultdict(set, {i: merged_adj[i] for i in range(num_nodes)})

    return final_adj

TRAIN_DATASETS = {"arxiv", "computer", "reddit"}

def generate_data(new_dataset, threshold=0.1, beta=0.55):
    model = load_model(config)
    embeddings, motif_adj = process_new_dataset(
        model, new_dataset, device=config["device"])

    print(f"New dataset {new_dataset} processed.")
    print(f"Generated embeddings shape: {embeddings.shape}")

    node_embeddings = embeddings

    node_features, edge_index = load_data(new_dataset)

    adj_lists = edge_index
    adj_lists_dropped, _ = dropout_edge(edge_index, p=0.2)

    ds_dir = dataset_dir(new_dataset)
    train_path = os.path.join(ds_dir, 'ocs_train.jsonl')
    test_path  = os.path.join(ds_dir, 'ocs_test.jsonl')

    tens = torch.load(processed_data_path(new_dataset), weights_only=False)
    labels = tens.y
    train_indices = torch.where(tens.train_mask)[0].numpy()
    test_indices  = torch.where(tens.test_mask)[0].numpy()

    # Dispatch table: dataset name -> *_input function (defined in gen.py).
    DISPATCH = {
        "cora":       cora_input,
        "citeseer":   citeseer_input,
        "pubmed":     pubmed_input,
        "arxiv":      arxiv_input,
        "history":    history_input,
        "computer":   computer_input,
        "photo":      photo_input,
        "wikics":     wikics_input,
        "instagram":  instagram_input,
        "reddit":     reddit_input,
        "cornell":    cornell_input,
        "texas":      texas_input,
        "wisconsin":  wisconsin_input,
        "washington": washington_input,
        "bookchild":  bookchild_input,
        "sportsfit":  sportsfit_input,
    }
    if new_dataset not in DISPATCH:
        raise ValueError(f"Unknown dataset: {new_dataset!r}.  "
                         f"Add a *_input() function in gen.py and a row in the "
                         f"DISPATCH table here.")
    fn = DISPATCH[new_dataset]
    common = (node_embeddings, adj_lists, adj_lists_dropped, labels)

    if new_dataset in TRAIN_DATASETS:
        print(f"  -> writing train ({len(train_indices)} samples) -> {train_path}")
        fn(train_indices, *common, train_path, threshold=threshold, beta=beta)
    else:
        print(f"  -> {new_dataset} is a test-only dataset; skipping train split")

    print(f"  -> writing test  ({len(test_indices)} samples) -> {test_path}")
    fn(test_indices,  *common, test_path,  threshold=threshold, beta=beta)

    print(f"  done: {ds_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate sequences for graph datasets')
    parser.add_argument('--dataset', type=str, default='history', help='Dataset name')
    parser.add_argument('--threshold', type=float, default=0.1, help='Threshold value')
    parser.add_argument('--beta', type=float, default=0.55, help='Beta value')
    parser.add_argument('--force-train', action='store_true',
                        help='Force generating ocs_train.jsonl even if dataset is not in TRAIN_DATASETS '
                             '(useful for smoke tests on cora etc.)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.force_train:
        TRAIN_DATASETS.add(args.dataset)
        print(f"[force-train] {args.dataset} added to TRAIN_DATASETS for this run")
    generate_data(args.dataset, threshold=args.threshold, beta=args.beta)

    

