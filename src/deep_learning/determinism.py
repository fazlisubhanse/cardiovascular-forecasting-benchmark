"""Strict deterministic CPU controls for PyTorch pilot execution."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_deterministic_seed(seed: int, threads: int = 1) -> dict[str, Any]:
    """Set Python, NumPy, and PyTorch seeds plus deterministic CPU controls."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    return {"seed": seed, "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_available": torch.cuda.is_available(), "device": "cpu"}
