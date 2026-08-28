"""Global determinism helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> int:
    """Seed ``random``, ``numpy`` and (if importable) ``torch``.

    Returns the seed so callers can log it. ``deterministic_torch`` also pins
    cuDNN into deterministic mode, which matters for reproducible training runs
    at the cost of some throughput.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed set for 3rd-party libs
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ModuleNotFoundError:
        pass
    return seed
