"""Split-conformal prediction sets for the per-hour sepsis label.

Given a calibration set the model never trained on, split conformal
(Vovk et al.; Angelopoulos & Bates, 2023) turns a probability into a
**set-valued** prediction with a finite-sample coverage guarantee: for a fresh
hour, ``P(Y in C(X)) >= 1 - alpha``.

Nonconformity score:  ``s(x, y) = 1 - p_hat_y(x)``  (one minus the predicted
probability of the realised label).

Threshold:  ``q_hat`` = the ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile
of calibration scores. The set for a new point is
``{y : 1 - p_hat_y(x) <= q_hat}``, i.e. it may be ``{0}``, ``{1}`` or the
uninformative ``{0, 1}`` -- and the fraction of ``{0,1}`` sets is a direct,
honest readout of model uncertainty.

``mondrian=True`` computes a separate ``q_hat`` per true class so coverage holds
*within* the septic and non-septic hours, not just on average -- important under
2% prevalence.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def _quantile_level(n: int, alpha: float) -> float:
    return min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / max(n, 1))


@dataclasses.dataclass
class ConformalRiskClassifier:
    alpha: float = 0.1
    mondrian: bool = True
    q_hat_: dict[int, float] | None = None
    q_hat_marginal_: float | None = None

    def fit(self, y_cal: np.ndarray, p_cal: np.ndarray) -> ConformalRiskClassifier:
        y_cal = np.asarray(y_cal).ravel().astype(int)
        p1 = np.clip(np.asarray(p_cal, dtype=np.float64).ravel(), 0.0, 1.0)
        keep = y_cal >= 0
        y_cal, p1 = y_cal[keep], p1[keep]
        p_true = np.where(y_cal == 1, p1, 1.0 - p1)
        scores = 1.0 - p_true

        self.q_hat_marginal_ = float(
            np.quantile(scores, _quantile_level(scores.size, self.alpha),
                        method="higher")
        )
        self.q_hat_ = {}
        for cls in (0, 1):
            s = scores[y_cal == cls]
            if s.size == 0:
                self.q_hat_[cls] = self.q_hat_marginal_
            else:
                self.q_hat_[cls] = float(
                    np.quantile(s, _quantile_level(s.size, self.alpha),
                                method="higher")
                )
        return self

    # ------------------------------------------------------------------ sets --
    def predict_set(self, p1: np.ndarray, allow_empty: bool = False) -> np.ndarray:
        """Return an ``(n, 2)`` boolean array: column ``c`` is True if class ``c``
        is in the conformal set.

        Class ``c`` is included when its nonconformity ``1 - p_c`` does not exceed
        the calibration threshold. With ``mondrian=True`` a *class-conditional*
        threshold ``q_hat_[c]`` is used, which is a valid test-time construction
        (we take the union over the two label hypotheses) and targets coverage
        *within* each class. Unless ``allow_empty`` we fall back to the argmax
        class when both are excluded -- this only raises coverage, so the
        ``1 - alpha`` guarantee is preserved.
        """
        if self.q_hat_ is None:
            raise RuntimeError("fit() first")
        p1 = np.clip(np.asarray(p1, dtype=np.float64).ravel(), 0.0, 1.0)
        p0 = 1.0 - p1
        if self.mondrian:
            in_0 = p0 >= 1.0 - self.q_hat_[0]
            in_1 = p1 >= 1.0 - self.q_hat_[1]
        else:
            q = self.q_hat_marginal_
            in_0 = (1.0 - p0) <= q
            in_1 = (1.0 - p1) <= q
        sets = np.stack([in_0, in_1], axis=1)
        if not allow_empty:
            empty = ~sets.any(axis=1)
            if np.any(empty):
                top = (p1[empty] >= 0.5).astype(int)
                sets[np.flatnonzero(empty), top] = True
        return sets

    def evaluate(self, y_true: np.ndarray, p1: np.ndarray) -> dict[str, float]:
        y_true = np.asarray(y_true).ravel().astype(int)
        keep = y_true >= 0
        y_true = y_true[keep]
        sets = self.predict_set(np.asarray(p1).ravel()[keep])
        covered = sets[np.arange(len(y_true)), y_true]
        size = sets.sum(axis=1)
        out = {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": float(covered.mean()),
            "avg_set_size": float(size.mean()),
            "frac_uncertain": float(np.mean(size == 2)),
            "frac_singleton": float(np.mean(size == 1)),
            "frac_empty": float(np.mean(size == 0)),
        }
        for cls in (0, 1):
            m = y_true == cls
            out[f"coverage_class{cls}"] = (
                float(covered[m].mean()) if np.any(m) else float("nan")
            )
        return out
