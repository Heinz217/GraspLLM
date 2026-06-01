
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# Some datasets contain torch_sparse.SparseTensor (arxiv).  Make sure the
# package is importable; we already source-built it in this env.
try:
    import torch_sparse  # noqa: F401
except Exception:
    pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from utils.paths import DATASET_ROOT, model_dir as _model_dir  

QWEN3_EMB_PATH = os.environ.get("GRASPLLM_QWEN3_EMB_MODEL", _model_dir("qwen3-embedding"))

DATASETS: "OrderedDict[str, str]" = OrderedDict([
    ("cora",       "cora"),
    ("citeseer",   "citeseer"),
    ("pubmed",     "pubmed"),
    ("arxiv",      "ogbn-arxiv"),
    ("history",    "history"),
    ("computer",   "computer"),
    ("photo",      "photo"),
    ("wikics",     "wikics"),
    ("instagram",  "instagram"),
    ("reddit",     "reddit"),
    ("cornell",    "cornell"),
    ("texas",      "texas"),
    ("washington", "washington"),
    ("wisconsin",  "wisconsin"),
    ("bookchild",  "bookchild"),
    ("sportsfit",  "sportsfit"),
])

DEFAULT_MAX_LENGTH = {
    "cora":        4096,   # p99=544,  max=1256
    "citeseer":    4096,   # p99=376,  max=636
    "pubmed":      4096,   # p99=761,  max=1386
    "arxiv":       4096,   # p99=445,  max=2019
    "computer":    4096,   # p99=373,  max=806
    "reddit":      4096,   # p99=1022, max=6157
    "history":     4096,   # p99=1463, max=22550 
    "photo":       4096,   # p99=1508, max=8241
    "bookchild":   4096,   # p99=1399, max=27610 
    "washington":  4096,   # p99=1538, max=2268
    "wikics":      4096,   # p99=4241, max=22091
    "cornell":     4096,   # p99=2268, max=2953
    "texas":       4096,   # p99=2427, max=3311
    "wisconsin":   4096,   # p99=2942, max=8098
    "instagram":    256,   # p99=128, max=524
    "sportsfit":    256,   # p99=69,  max=230
}

def assemble_text(dataset: str, data) -> List[str]:
    """Return one string per node, in node-id order, using `raw_texts`.

    Falls back to `title + " " + abs` if `raw_texts` is absent.
    """
    keys = set(getattr(data, "keys", lambda: [])() if callable(getattr(data, "keys", None)) else list(data.keys()))
    # PyG Data exposes attribute access, but `in data` and `data.keys()` work.
    if "raw_texts" in keys:
        rts = data["raw_texts"] if hasattr(data, "__getitem__") else getattr(data, "raw_texts")
        return [str(t) if t is not None else "" for t in rts]
    if "title" in keys and "abs" in keys:
        titles = data["title"]; abss = data["abs"]
        return [f"Title: {t}\tAbstract: {a}" for t, a in zip(titles, abss)]
    raise ValueError(f"[{dataset}] no usable text field; keys={sorted(keys)}")

def _safe_torch_load(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def last_token_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last-token pooling that is robust to both left- and right-padding.

    With left-padding (which we use), the real last token is at index `-1`,
    so this is equivalent to `last_hidden[:, -1]`. We still implement the
    generic version that supports both, matching the Qwen3-Embedding repo.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    bsz = last_hidden.size(0)
    return last_hidden[torch.arange(bsz, device=last_hidden.device), seq_lens]


def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    out_dtype: torch.dtype = torch.float16,
    log_every_pct: float = 1.0,
    log_prefix: str = "",
) -> torch.Tensor:
    n = len(texts)
    out = torch.empty((n, model.config.hidden_size), dtype=out_dtype)

    n_batches = math.ceil(n / batch_size)
    log_step = max(1, int(n_batches * log_every_pct / 100.0))
    started = time.time()

    def _bar(pct: float, w: int = 20) -> str:
        f = int(round(pct * w / 100.0))
        f = max(0, min(w, f))
        return "█" * f + "░" * (w - f)

    model.eval()
    with torch.inference_mode():
        for bi in range(n_batches):
            s = bi * batch_size
            e = min(n, s + batch_size)
            batch_texts = texts[s:e]
            # Replace empty strings with a single space (otherwise tokenizer may emit 0 tokens).
            batch_texts = [t if (t and t.strip()) else " " for t in batch_texts]

            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**batch)
            emb = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            out[s:e] = emb.to(out_dtype).cpu()

            if bi % log_step == 0 or bi == n_batches - 1:
                now = time.time()
                done = e
                pct = 100.0 * done / n
                elapsed = now - started
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(f"{log_prefix} [{_bar(pct)}] {pct:5.1f}%  "
                      f"batch {bi+1:>5d}/{n_batches}  "
                      f"({done:>7d}/{n})  "
                      f"rate={rate:6.1f} samp/s  "
                      f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s",
                      flush=True)
    return out


def shard_indices(n: int, shard_id: int, num_shards: int) -> List[int]:
    """Return the global indices this shard owns (block-shard, contiguous)."""
    if num_shards <= 1:
        return list(range(n))
    sz = (n + num_shards - 1) // num_shards
    s = shard_id * sz
    e = min(n, s + sz)
    return list(range(s, e))


def auto_batch_size(max_length: int, user_bs: int) -> int:
    table = {256: 128, 512: 64, 1024: 32, 2048: 16, 4096: 8}
    suggested = table.get(max_length, 16)
    if user_bs > 0 and user_bs < suggested:
        return user_bs
    return suggested


def process_dataset(
    name: str,
    subdir: str,
    *,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    shard_id: int,
    num_shards: int,
    overwrite: bool,
):
    base = os.path.join(DATASET_ROOT, subdir)
    final_out = os.path.join(base, "qwen3_emb_x.pt")
    pt_path   = os.path.join(base, "processed_data.pt")
    if not os.path.isfile(pt_path):
        print(f"[{name}] SKIP — {pt_path} not found"); return

    if num_shards > 1:
        out_path = os.path.join(base, f"qwen3_emb_x.shard{shard_id}of{num_shards}.pt")
    else:
        out_path = final_out

    if os.path.isfile(out_path) and not overwrite:
        try:
            existing = _safe_torch_load(out_path)
            if isinstance(existing, dict):
                existing_emb = existing.get("emb", None)
            else:
                existing_emb = existing
            if existing_emb is not None:
                print(f"[{name}] EXISTS  shape={tuple(existing_emb.shape)}  "
                      f"dtype={existing_emb.dtype}  ->  skip (use --overwrite to redo)")
                return
        except Exception:
            pass

    # per-dataset max_length & batch size
    per_ds_ml = DEFAULT_MAX_LENGTH.get(name, 1024)
    if max_length and max_length > 0:
        ml = max_length      # explicit CLI override
    else:
        ml = per_ds_ml       # per-dataset table (recommended)
    bs = auto_batch_size(ml, batch_size)

    t0 = time.time()
    print(f"[{name}] loading  {pt_path}  (max_length={ml}, batch_size={bs})")
    data = _safe_torch_load(pt_path)
    texts_full = assemble_text(name, data)
    n_full = len(texts_full)

    indices = shard_indices(n_full, shard_id, num_shards)
    texts = [texts_full[i] for i in indices]
    print(f"[{name}] N_full={n_full}  shard={shard_id}/{num_shards}  this_shard={len(texts)}")
    if len(texts) == 0:
        print(f"[{name}] empty shard, nothing to do."); return

    log_prefix = f"[{name}{'#'+str(shard_id) if num_shards>1 else ''}]"
    emb = encode_texts(
        model, tokenizer, texts,
        batch_size=bs, max_length=ml,
        device=device, out_dtype=torch.float16,
        log_prefix=log_prefix,
    )
    payload = {
        "emb":          emb,                       # (n_shard, 4096) fp16
        "indices":      torch.tensor(indices, dtype=torch.long),
        "n_full":       n_full,
        "shard_id":     shard_id,
        "num_shards":   num_shards,
        "encoder":      "Qwen3-Embedding-8B",
        "hidden_size":  int(model.config.hidden_size),
        "max_length":   int(ml),
        "dtype":        "float16",
    }
    torch.save(payload, out_path)
    dt = time.time() - t0
    print(f"[{name}] DONE  -> {out_path}   "
          f"emb={tuple(emb.shape)} fp16   in {dt/60:.1f} min")


def merge_shards(name: str, subdir: str):
    base = os.path.join(DATASET_ROOT, subdir)
    files = sorted([f for f in os.listdir(base)
                    if f.startswith("qwen3_emb_x.shard") and f.endswith(".pt")])
    if not files:
        print(f"[{name}] no shard files to merge."); return
    print(f"[{name}] merging {len(files)} shards: {files}")
    parts = [_safe_torch_load(os.path.join(base, f)) for f in files]
    n_full = parts[0]["n_full"]
    hsz    = parts[0]["hidden_size"]
    out = torch.empty((n_full, hsz), dtype=torch.float16)
    seen = torch.zeros(n_full, dtype=torch.bool)
    for p in parts:
        idx = p["indices"]
        out[idx] = p["emb"]
        seen[idx] = True
    if not seen.all():
        missing = (~seen).nonzero(as_tuple=False).squeeze().tolist()
        if not isinstance(missing, list): missing = [missing]
        raise RuntimeError(f"[{name}] merge incomplete; missing {len(missing)} indices, e.g. {missing[:10]}")
    final = os.path.join(base, "qwen3_emb_x.pt")
    torch.save(out, final)
    print(f"[{name}] merged -> {final}  shape={tuple(out.shape)} fp16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="logical names (default: all 16 unless --all is set).")
    ap.add_argument("--all", action="store_true", help="run all datasets")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 (default) = use auto table per max_length; >0 = upper bound override.")
    ap.add_argument("--max-length", type=int, default=0,
                    help="0 (default) = use per-dataset table from p99 stats; >0 = override (rarely needed).")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge-only", action="store_true",
                    help="skip encoding; only merge existing shard files into qwen3_emb_x.pt")
    ap.add_argument("--model-path", default=QWEN3_EMB_PATH)
    args = ap.parse_args()

    if args.all:
        names = list(DATASETS.keys())
    elif args.datasets:
        names = args.datasets
    else:
        names = list(DATASETS.keys())
    bad = [n for n in names if n not in DATASETS]
    if bad:
        print(f"[FATAL] unknown datasets: {bad}", file=sys.stderr); sys.exit(2)

    if args.merge_only:
        for n in names:
            merge_shards(n, DATASETS[n])
        return

    from transformers import AutoTokenizer, AutoModel
    print(f"[init] loading {args.model_path}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    device = next(model.parameters()).device
    print(f"[init] loaded in {time.time()-t0:.1f}s  device={device}  "
          f"hidden={model.config.hidden_size}  pad_side=left  pad_id={tokenizer.pad_token_id}")

    for n in names:
        try:
            process_dataset(
                n, DATASETS[n],
                model=model, tokenizer=tokenizer, device=device,
                batch_size=args.batch_size, max_length=args.max_length,
                shard_id=args.shard_id, num_shards=args.num_shards,
                overwrite=args.overwrite,
            )
            gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"[{n}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
