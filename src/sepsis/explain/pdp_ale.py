"""Partial dependence and Accumulated Local Effects for the GBM baseline.

PDP (Friedman, 2001) shows the marginal effect of a feature but is biased when
features are correlated -- rife in physiology (MAP vs SBP vs DBP). ALE
(Apley & Zhu, 2020) fixes this by averaging *local* differences within quantile
bins, so both are reported side by side.
"""

from __future__ import annotations

import numpy as np


def partial_dependence(
    predict_fn, X: np.ndarray, feature_idx: int, grid_size: int = 25
) -> dict[str, np.ndarray]:
    col = X[:, feature_idx]
    grid = np.quantile(col, np.linspace(0.02, 0.98, grid_size))
    grid = np.unique(grid)
    pd = np.empty(grid.size)
    Xc = X.copy()
    for i, g in enumerate(grid):
        Xc[:, feature_idx] = g
        pd[i] = float(np.mean(predict_fn(Xc)))
    return {"grid": grid, "partial_dependence": pd}


def accumulated_local_effects(
    predict_fn, X: np.ndarray, feature_idx: int, n_bins: int = 20
) -> dict[str, np.ndarray]:
    col = X[:, feature_idx]
    edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return {"centers": edges, "ale": np.zeros_like(edges)}
    idx = np.clip(np.digitize(col, edges) - 1, 0, edges.size - 2)
    local = np.zeros(edges.size - 1)
    counts = np.zeros(edges.size - 1)
    for b in range(edges.size - 1):
        m = idx == b
        if not np.any(m):
            continue
        lo = X[m].copy()
        hi = X[m].copy()
        lo[:, feature_idx] = edges[b]
        hi[:, feature_idx] = edges[b + 1]
        local[b] = float(np.mean(predict_fn(hi) - predict_fn(lo)))
        counts[b] = m.sum()
    ale = np.concatenate([[0.0], np.cumsum(local)])
    ale = ale - np.sum((counts / counts.sum()) * (ale[:-1] + ale[1:]) / 2.0)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {"centers": np.concatenate([[edges[0]], centers]), "ale": ale}
