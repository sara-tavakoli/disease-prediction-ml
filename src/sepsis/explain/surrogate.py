"""A global surrogate decision tree.

Fit a shallow, readable tree to the *predictions* of any model and measure how
faithfully it reproduces them (R^2 on the logit, and agreement of the binarised
alarm). High fidelity means the tree's rules are a defensible summary of what
the black box does; low fidelity is itself a finding.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor, export_text


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@dataclasses.dataclass
class SurrogateFit:
    fidelity_r2: float
    alarm_agreement: float
    rules: str
    max_depth: int


class GlobalSurrogateTree:
    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 50, seed: int = 0):
        self.tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed
        )
        self.feature_names: list[str] | None = None

    def fit(
        self,
        X: np.ndarray,
        model_probs: np.ndarray,
        feature_names: list[str],
        alarm_threshold: float = 0.5,
    ) -> SurrogateFit:
        self.feature_names = list(feature_names)
        y = _logit(np.asarray(model_probs, dtype=np.float64).ravel())
        self.tree.fit(X, y)
        pred_logit = self.tree.predict(X)
        surro_p = 1.0 / (1.0 + np.exp(-pred_logit))
        agree = float(np.mean((surro_p >= alarm_threshold) == (model_probs >= alarm_threshold)))
        return SurrogateFit(
            fidelity_r2=float(r2_score(y, pred_logit)),
            alarm_agreement=agree,
            rules=export_text(self.tree, feature_names=self.feature_names, max_depth=4),
            max_depth=int(self.tree.get_depth()),
        )
