"""Post-hoc probability calibration and calibration diagnostics.

Discrimination (AUROC/AUPRC) says nothing about whether a "0.3" really means a
30% chance of impending sepsis. We fit a calibrator on a held-out split and
report the expected / maximum calibration error and a reliability curve before
and after.

Methods
-------
platt        1-D logistic regression on the logit (Platt, 1999).
isotonic     monotonic piecewise-constant fit (Zadrozny & Elkan, 2002).
temperature  single scalar T dividing the logits (Guo et al., 2017); preserves
             the ranking, so AUROC is unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclasses.dataclass
class Calibrator:
    method: str
    _platt: LogisticRegression | None = None
    _iso: IsotonicRegression | None = None
    _temperature: float | None = None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).ravel()
        if self.method == "none":
            return p
        if self.method == "platt":
            return self._platt.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self._iso.predict(p)
        if self.method == "temperature":
            return 1.0 / (1.0 + np.exp(-_logit(p) / self._temperature))
        raise ValueError(self.method)

    __call__ = transform


def fit_calibrator(y_true: np.ndarray, p: np.ndarray, method: str = "isotonic") -> Calibrator:
    y_true = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(p, dtype=np.float64).ravel()
    keep = (y_true >= 0) & np.isfinite(p)
    y_true, p = y_true[keep], p[keep]

    if method == "none":
        return Calibrator("none")
    if method == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(_logit(p).reshape(-1, 1), y_true)
        return Calibrator("platt", _platt=lr)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y_true)
        return Calibrator("isotonic", _iso=iso)
    if method == "temperature":
        z = _logit(p)
        best_t, best_nll = 1.0, np.inf
        for t in np.linspace(0.25, 6.0, 231):
            q = 1.0 / (1.0 + np.exp(-z / t))
            q = np.clip(q, 1e-7, 1 - 1e-7)
            nll = -np.mean(y_true * np.log(q) + (1 - y_true) * np.log(1 - q))
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        return Calibrator("temperature", _temperature=best_t)
    raise ValueError(f"unknown calibration method {method!r}")


def reliability_curve(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 15, strategy: str = "quantile"
) -> dict[str, np.ndarray]:
    y_true = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(p, dtype=np.float64).ravel()
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    idx = np.digitize(p, edges) - 1
    conf, acc, weight = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not np.any(m):
            continue
        conf.append(p[m].mean())
        acc.append(y_true[m].mean())
        weight.append(m.mean())
    return {
        "confidence": np.asarray(conf),
        "accuracy": np.asarray(acc),
        "weight": np.asarray(weight),
    }


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 15
) -> dict[str, float]:
    rc = reliability_curve(y_true, p, n_bins=n_bins, strategy="uniform")
    gap = np.abs(rc["confidence"] - rc["accuracy"])
    return {
        "ece": float(np.sum(gap * rc["weight"])),
        "mce": float(np.max(gap)) if gap.size else float("nan"),
    }
