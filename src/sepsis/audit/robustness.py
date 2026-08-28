"""Perturbation and missing-data stress tests.

ICU monitoring streams are noisy and intermittently dropped. A model that is
only accurate on clean data is not deployable. We re-score the test set under:

* **Gaussian sensor noise** added to the standardised value channels at a range
  of sigmas (mask / delta / static channels untouched);
* **extra missingness** -- randomly blanking an additional fraction of the
  observed measurements and re-deriving carry-forward values and recency.

Both report the AUROC / AUPRC degradation curve; a robust model degrades
gracefully and monotonically.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sepsis.constants import DYNAMIC_COLS
from sepsis.data.preprocess import TensorDataset
from sepsis.evaluation.metrics import auprc, auroc

ScoreFn = Callable[[TensorDataset], np.ndarray]  # -> flat hour-level risk


def _flat_labels(td: TensorDataset) -> np.ndarray:
    _, y = td.flat_valid()
    return y


def _perturbed_copy(td: TensorDataset) -> TensorDataset:
    import dataclasses

    return dataclasses.replace(td, X=td.X.copy())


def noise_robustness_curve(
    td: TensorDataset,
    score_fn: ScoreFn,
    sigmas: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5),
    seed: int = 20190804,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_dyn = len(DYNAMIC_COLS)
    y = _flat_labels(td)
    rows = []
    for sigma in sigmas:
        pert = _perturbed_copy(td)
        if sigma > 0:
            noise = rng.normal(0.0, sigma, size=pert.X[..., :n_dyn].shape).astype(np.float32)
            for i, n in enumerate(pert.lengths):
                pert.X[i, :n, :n_dyn] += noise[i, :n]
        s = score_fn(pert)
        rows.append({"sigma": float(sigma), "auroc": auroc(y, s), "auprc": auprc(y, s)})
    return rows


def missingness_stress_test(
    td: TensorDataset,
    score_fn: ScoreFn,
    drop_fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5),
    seed: int = 20190804,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    n_dyn = len(DYNAMIC_COLS)
    mask_sl = slice(n_dyn, 2 * n_dyn)
    delta_sl = slice(2 * n_dyn, 3 * n_dyn)
    y = _flat_labels(td)
    rows = []

    for frac in drop_fractions:
        pert = _perturbed_copy(td)
        if frac > 0:
            for i, n in enumerate(pert.lengths):
                seq = pert.X[i, :n]
                obs = seq[:, mask_sl] > 0.5
                drop = obs & (rng.random(obs.shape) < frac)
                for j in range(n_dyn):
                    col_val = seq[:, j].copy()
                    col_obs = obs[:, j].copy()
                    col_obs[drop[:, j]] = False
                    last, since = 0.0, 0.0
                    seen_any = False
                    for t in range(n):
                        if col_obs[t]:
                            last = col_val[t]
                            since = 0.0
                            seen_any = True
                        else:
                            since += 1.0
                            if not seen_any:
                                last = col_val[t]  # keep the train-mean backfill
                        seq[t, j] = last
                        seq[t, delta_sl.start + j] = np.log1p(since) / max(np.log1p(n), 1e-6)
                    seq[:, mask_sl.start + j] = col_obs.astype(np.float32)
                pert.X[i, :n] = seq
        s = score_fn(pert)
        rows.append({"extra_missing_frac": float(frac), "auroc": auroc(y, s), "auprc": auprc(y, s)})
    return rows
