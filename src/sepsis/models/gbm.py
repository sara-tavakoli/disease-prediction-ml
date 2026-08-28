"""LightGBM baseline over sliding-window tabular features.

Gradient-boosted trees remain a strong, fast baseline for tabular clinical
prediction and give an honest yardstick for the sequence models. The wrapper
keeps a stable feature order, trains with early stopping on a grouped
validation fold, and exposes ``predict_sequences`` so its per-hour scores can be
fed into the same evaluation + utility-score machinery as the neural models.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from sepsis.config import ModelConfig
from sepsis.features.tabular import WindowedTable
from sepsis.utils.logging import get_logger

log = get_logger("models.gbm")


@dataclasses.dataclass
class GBMResult:
    val_auprc: float
    best_iteration: int
    feature_importance: dict[str, float]


class GBMRiskModel:
    def __init__(self, cfg: ModelConfig, seed: int = 20190804):
        self.cfg = cfg
        self.seed = seed
        self.booster = None
        self.feature_names: list[str] | None = None

    def fit(self, train: WindowedTable, val: WindowedTable) -> GBMResult:
        import lightgbm as lgb

        self.feature_names = list(train.feature_names)
        pos = float(train.y.sum())
        neg = float(len(train.y) - pos)
        params = {
            "objective": "binary",
            "metric": ["auc", "average_precision"],
            "num_leaves": self.cfg.gbm_num_leaves,
            "learning_rate": self.cfg.gbm_learning_rate,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_data_in_leaf": 64,
            "scale_pos_weight": max(neg / max(pos, 1.0), 1.0),
            "seed": self.seed,
            "verbosity": -1,
            "force_col_wise": True,
        }
        dtrain = lgb.Dataset(train.X, label=train.y, feature_name=self.feature_names)
        dval = lgb.Dataset(val.X, label=val.y, reference=dtrain)
        self.booster = lgb.train(
            params,
            dtrain,
            num_boost_round=self.cfg.gbm_n_estimators,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        gain = self.booster.feature_importance(importance_type="gain")
        imp = dict(
            sorted(
                zip(self.feature_names, (gain / max(gain.sum(), 1e-9)).tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
        from sepsis.evaluation.metrics import auprc

        vp = self.predict(val)
        res = GBMResult(
            val_auprc=float(auprc(val.y, vp)),
            best_iteration=int(self.booster.best_iteration or self.cfg.gbm_n_estimators),
            feature_importance=imp,
        )
        log.info("GBM trained: val AUPRC=%.4f best_iter=%d",
                 res.val_auprc, res.best_iteration)
        return res

    def predict(self, table: WindowedTable) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("fit() first")
        return self.booster.predict(
            table.X, num_iteration=self.booster.best_iteration
        ).astype(np.float64)

    def predict_sequences(
        self, table: WindowedTable, n_sequences: int, max_len: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scatter row-level scores back into ``(N, T)`` risk / label grids."""
        scores = np.zeros((n_sequences, max_len), dtype=np.float64)
        labels = np.full((n_sequences, max_len), -1, dtype=np.int8)
        p = self.predict(table)
        for row in range(len(table)):
            g = int(table.groups[row])
            t = int(table.times[row])
            if t < max_len:
                scores[g, t] = p[row]
                labels[g, t] = table.y[row]
        return scores, labels

    def save(self, path: str) -> None:
        if self.booster is None:
            raise RuntimeError("fit() first")
        self.booster.save_model(path)

    def load(self, path: str) -> GBMRiskModel:
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=path)
        self.feature_names = self.booster.feature_name()
        return self
