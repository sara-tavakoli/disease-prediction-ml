"""Decision-curve analysis (Vickers & Elkin, 2006).

Net benefit at a threshold probability ``p_t`` is

    NB = TP/N - FP/N * (p_t / (1 - p_t))

which puts a model's clinical value on the same scale as the default policies
"treat everyone" and "treat no one". A model is useful over the range of ``p_t``
where its curve sits above both defaults.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class DecisionCurve:
    thresholds: np.ndarray
    net_benefit_model: np.ndarray
    net_benefit_all: np.ndarray
    net_benefit_none: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "threshold": self.thresholds.tolist(),
            "model": self.net_benefit_model.tolist(),
            "treat_all": self.net_benefit_all.tolist(),
            "treat_none": self.net_benefit_none.tolist(),
        }


def decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> DecisionCurve:
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    keep = (y_true >= 0) & np.isfinite(y_prob)
    y_true, y_prob = y_true[keep], y_prob[keep]
    if thresholds is None:
        thresholds = np.round(np.linspace(0.01, 0.6, 60), 4)

    n = y_true.size
    prev = y_true.mean()
    nb_model = np.empty_like(thresholds, dtype=np.float64)
    for i, pt in enumerate(thresholds):
        pred = y_prob >= pt
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        w = pt / (1.0 - pt)
        nb_model[i] = tp / n - fp / n * w

    nb_all = prev - (1.0 - prev) * (thresholds / (1.0 - thresholds))
    nb_none = np.zeros_like(thresholds)
    return DecisionCurve(thresholds, nb_model, nb_all, nb_none)
