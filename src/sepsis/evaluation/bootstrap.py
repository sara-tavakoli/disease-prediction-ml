"""Patient-clustered bootstrap confidence intervals.

Hour-level rows within a stay are correlated, so resampling rows would give
dishonestly tight intervals. We resample **stays** with replacement (the
cluster bootstrap) and recompute the metric on the pooled hours each time.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np


@dataclasses.dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    se: float
    n_resamples: int
    level: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.point, self.lo, self.hi

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 20190804,
) -> BootstrapCI:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    groups = np.asarray(groups).ravel()
    rng = np.random.default_rng(seed)

    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    point = float(metric_fn(y_true, y_score))

    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        rows = np.concatenate([idx_by_group[g] for g in picked])
        try:
            stats[b] = metric_fn(y_true[rows], y_score[rows])
        except ValueError:
            stats[b] = np.nan

    stats = stats[np.isfinite(stats)]
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return BootstrapCI(
        point=point,
        lo=float(lo),
        hi=float(hi),
        se=float(np.std(stats, ddof=1)),
        n_resamples=int(stats.size),
        level=level,
    )
