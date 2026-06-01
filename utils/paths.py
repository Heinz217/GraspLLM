from __future__ import annotations

import os
from typing import Final

REPO_ROOT: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_ROOT: Final[str] = os.environ.get(
    "GRASPLLM_DATASET_ROOT", os.path.join(REPO_ROOT, "dataset")
)
MODELS_ROOT: Final[str] = os.environ.get(
    "GRASPLLM_MODELS_ROOT", os.path.join(REPO_ROOT, "models")
)
CHECKPOINT_ROOT: Final[str] = os.environ.get(
    "GRASPLLM_CHECKPOINT_ROOT", os.path.join(REPO_ROOT, "checkpoints")
)

_DATASET_ALIAS: Final[dict] = {
    "arxiv":      "ogbn-arxiv",
    "ogbn-arxiv": "ogbn-arxiv",
}

# Canonical list of supported datasets.
DATASETS: Final[tuple] = (
    "cora", "citeseer", "pubmed", "arxiv",
    "history", "computer", "photo", "wikics",
    "instagram", "reddit",
    "cornell", "texas", "washington", "wisconsin",
    "bookchild", "sportsfit",
)

# Default Stage-1 GNN source datasets. Modify as needed for your setup.
STAGE1_SOURCE_DATASETS: Final[tuple] = (
    "arxiv", "pubmed", "computer", "history", "reddit",
)


def _subdir(name: str) -> str:
    """Map a code-side dataset name to its on-disk subdirectory name."""
    return _DATASET_ALIAS.get(name, name)


def dataset_dir(name: str) -> str:
    """Return absolute path of the dataset folder, e.g. dataset_dir('arxiv')."""
    return os.path.join(DATASET_ROOT, _subdir(name))


def processed_data_path(name: str) -> str:
    return os.path.join(dataset_dir(name), "processed_data.pt")


def qwen3_emb_path(name: str) -> str:
    return os.path.join(dataset_dir(name), "qwen3_emb_x.pt")


_MODEL_ALIAS: Final[dict] = {
    "qwen3-embedding":    "Qwen3-Embedding-8B",
    "qwen3-embedding-8b": "Qwen3-Embedding-8B",
    "qwen3":              "Qwen3-8B",
    "qwen3-8b":           "Qwen3-8B",
    "qwen3-moe":          "Qwen3-30B-A3B-Instruct-2507",
    "qwen3-30b":          "Qwen3-30B-A3B-Instruct-2507",
    "mistral":            "Mistral-7B-Instruct-v0.3",
    "mistral-7b":         "Mistral-7B-Instruct-v0.3",
    "vicuna":             "vicuna-7b-v1.5",
    "vicuna-7b":          "vicuna-7b-v1.5",
    "llama3":             "Meta-Llama-3.1-8B-Instruct",
    "llama3.1":           "Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-8b":       "Meta-Llama-3.1-8B-Instruct",
}


def model_dir(name: str) -> str:
    """Resolve a model alias / path to an absolute model directory.

    If `name` is an absolute existing path, return as-is.
    Otherwise look up the alias map (case-insensitive); fall back to verbatim.
    """
    if os.path.isabs(name) and os.path.isdir(name):
        return name
    key = name.lower()
    sub = _MODEL_ALIAS.get(key, name)
    return os.path.join(MODELS_ROOT, sub)


if __name__ == "__main__":
    print("REPO_ROOT      :", REPO_ROOT)
    print("DATASET_ROOT   :", DATASET_ROOT)
    print("MODELS_ROOT    :", MODELS_ROOT)
    print("CHECKPOINT_ROOT:", CHECKPOINT_ROOT)
    print()
    print("--- dataset paths ---")
    for d in DATASETS:
        p = dataset_dir(d)
        ok = "OK" if os.path.isdir(p) else "MISSING"
        print(f"  {d:<12}  {p}  [{ok}]")
    print()
    print("--- model paths ---")
    for m in ["vicuna", "mistral", "llama3", "qwen3", "qwen3-embedding"]:
        p = model_dir(m)
        ok = "OK" if os.path.isdir(p) else "MISSING"
        print(f"  {m:<16}  {p}  [{ok}]")
