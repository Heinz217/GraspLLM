"""Shared embedding via CUDA IPC for multi-GPU OCS sampling (--shared-emb)."""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from typing import List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import qwen3_emb_path  # noqa: E402


# Owner process: load the full embedding once, expose it via an IPC handle.
def _owner_proc(gpu_id, dataset, fp16, ready_evt, stop_evt, handle_q):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    obj = torch.load(qwen3_emb_path(dataset), weights_only=False)
    emb = obj["emb"] if isinstance(obj, dict) else obj
    emb = (emb.half() if fp16 else emb.float()).to(device)
    torch.cuda.synchronize(device)

    storage = emb.untyped_storage()
    handle = storage._share_cuda_()  # (device, handle, size, ...)
    handle_q.put({
        "handle":   handle,
        "shape":    tuple(emb.shape),
        "dtype":    str(emb.dtype),
        "owner_id": gpu_id,
    })
    print(f"[shared-emb owner GPU{gpu_id}] embedding loaded "
          f"shape={tuple(emb.shape)} dtype={emb.dtype}", flush=True)
    ready_evt.set()
    stop_evt.wait()


def attach_shared_embedding(handle_dict, local_gpu_id):
    """Attach to the IPC-shared storage and rebuild a tensor view (lives on owner GPU)."""
    handle = handle_dict["handle"]
    shape  = handle_dict["shape"]
    dtype  = {"torch.float16": torch.float16,
              "torch.float32": torch.float32,
              "torch.bfloat16": torch.bfloat16}[handle_dict["dtype"]]
    storage = torch.UntypedStorage._new_shared_cuda(*handle)
    emb = torch.tensor([], dtype=dtype, device=storage.device)
    emb.set_(storage, 0, shape, None)
    return emb


class SharedEmbeddingHandle:
    """Context manager: spawns the owner process and yields the IPC handle dict."""

    def __init__(self, dataset: str, gpus: List[int], fp16: bool = True):
        self.dataset = dataset
        self.gpus = gpus
        self.fp16 = fp16
        self._proc = None
        self._stop = None
        self._handle = None

    def __enter__(self):
        ctx = mp.get_context("spawn")
        ready = ctx.Event()
        self._stop = ctx.Event()
        q = ctx.Queue()
        owner_gpu = self.gpus[0]
        self._proc = ctx.Process(target=_owner_proc,
                                 args=(owner_gpu, self.dataset, self.fp16,
                                       ready, self._stop, q))
        self._proc.start()
        ready.wait(timeout=600)  # allow up to 10 min for big embeddings
        self._handle = q.get(timeout=10)
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stop is not None:
            self._stop.set()
        if self._proc is not None:
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                self._proc.terminate()
        return False
