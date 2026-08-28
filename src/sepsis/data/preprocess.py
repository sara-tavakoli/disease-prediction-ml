"""Turn a list of :class:`PatientRecord` into leak-free model tensors.

For every ICU hour and every dynamic channel we emit three quantities, following
the missing-data representation of Che et al. (GRU-D, 2018):

* ``value``       -- last observed measurement, carried forward, z-scored with
  **train-only** statistics; back-filled with the train median before the first
  observation.
* ``mask``        -- 1 if the channel was *actually measured* this hour, else 0.
* ``delta``       -- hours since the channel was last measured, ``log1p``-scaled.

Static covariates (age, sex, ICU unit, admission offset) are median-imputed,
z-scored and broadcast across time. The final feature vector has length
``3 * len(DYNAMIC_COLS) + len(STATIC_COLS)`` and its column order is frozen in
:class:`PreprocessArtifacts` so training, evaluation and the serving API all
agree.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from sepsis.constants import (
    DYNAMIC_COLS,
    PHYSIOLOGIC_RANGES,
    POPULATION_MEDIANS,
    STATIC_COLS,
)
from sepsis.data.psv import PatientRecord
from sepsis.utils.logging import get_logger

log = get_logger("data.preprocess")


@dataclasses.dataclass
class PreprocessArtifacts:
    """Everything needed to reproduce the transform at inference time."""

    feature_names: list[str]
    dynamic_mean: dict[str, float]
    dynamic_std: dict[str, float]
    static_mean: dict[str, float]
    static_std: dict[str, float]
    delta_scale: float
    max_seq_len: int
    clip_ranges: dict[str, list[float]]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PreprocessArtifacts:
        return cls(**json.loads(Path(path).read_text()))

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


@dataclasses.dataclass
class TensorDataset:
    """Right-padded arrays. ``X[i, :lengths[i]]`` is the real sequence."""

    X: np.ndarray             # (N, T, F) float32
    y: np.ndarray             # (N, T)    int8
    lengths: np.ndarray       # (N,)      int32
    pids: list[str]
    static_raw: np.ndarray    # (N, len(STATIC_COLS)) float32, pre-standardisation
    feature_names: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    def flat_valid(self) -> tuple[np.ndarray, np.ndarray]:
        """Flatten to ``(rows, F)`` / ``(rows,)`` keeping only real timesteps."""
        keep = np.zeros((len(self), self.X.shape[1]), dtype=bool)
        for i, n in enumerate(self.lengths):
            keep[i, :n] = True
        return self.X[keep], self.y[keep]


class Preprocessor:
    def __init__(self, max_seq_len: int = 336):
        self.max_seq_len = int(max_seq_len)
        self.artifacts: PreprocessArtifacts | None = None

    @classmethod
    def from_artifacts(cls, artifacts: PreprocessArtifacts) -> Preprocessor:
        obj = cls(max_seq_len=artifacts.max_seq_len)
        obj.artifacts = artifacts
        return obj

    # ------------------------------------------------------------------ fit --
    def fit(self, records: list[PatientRecord]) -> Preprocessor:
        sums: dict[str, float] = dict.fromkeys(DYNAMIC_COLS, 0.0)
        sqs: dict[str, float] = dict.fromkeys(DYNAMIC_COLS, 0.0)
        cnts: dict[str, int] = dict.fromkeys(DYNAMIC_COLS, 0)
        for rec in records:
            for c in DYNAMIC_COLS:
                col = rec.frame[c].to_numpy(dtype=np.float64)
                lo, hi = PHYSIOLOGIC_RANGES.get(c, (-np.inf, np.inf))
                col = np.clip(col, lo, hi)
                obs = col[~np.isnan(col)]
                sums[c] += obs.sum()
                sqs[c] += np.square(obs).sum()
                cnts[c] += obs.size

        d_mean, d_std = {}, {}
        for c in DYNAMIC_COLS:
            if cnts[c] > 1:
                mu = sums[c] / cnts[c]
                var = max(sqs[c] / cnts[c] - mu * mu, 1e-8)
            else:
                mu, var = POPULATION_MEDIANS.get(c, 0.0), 1.0
            d_mean[c], d_std[c] = float(mu), float(np.sqrt(var))

        static_stack = np.array(
            [[rec.static_vector()[c] for c in STATIC_COLS] for rec in records],
            dtype=np.float64,
        )
        s_mean, s_std = {}, {}
        for j, c in enumerate(STATIC_COLS):
            col = static_stack[:, j]
            col = col[~np.isnan(col)]
            mu = float(col.mean()) if col.size else POPULATION_MEDIANS.get(c, 0.0)
            sd = float(col.std()) if col.size else 1.0
            s_mean[c], s_std[c] = mu, max(sd, 1e-6)

        feat_names = (
            [f"{c}__value" for c in DYNAMIC_COLS]
            + [f"{c}__mask" for c in DYNAMIC_COLS]
            + [f"{c}__delta" for c in DYNAMIC_COLS]
            + [f"{c}__static" for c in STATIC_COLS]
        )
        self.artifacts = PreprocessArtifacts(
            feature_names=feat_names,
            dynamic_mean=d_mean,
            dynamic_std=d_std,
            static_mean=s_mean,
            static_std=s_std,
            delta_scale=float(np.log1p(self.max_seq_len)),
            max_seq_len=self.max_seq_len,
            clip_ranges={k: list(v) for k, v in PHYSIOLOGIC_RANGES.items()},
        )
        log.info("fitted preprocessor: %d features, %d train stays",
                 len(feat_names), len(records))
        return self

    # -------------------------------------------------------------- transform --
    def _encode_one(self, rec: PatientRecord) -> tuple[np.ndarray, np.ndarray, int]:
        assert self.artifacts is not None
        a = self.artifacts
        T = min(rec.n_hours, self.max_seq_len)
        frame = rec.frame.iloc[-T:] if rec.n_hours > T else rec.frame
        D = len(DYNAMIC_COLS)
        values = np.zeros((T, D), dtype=np.float32)
        mask = np.zeros((T, D), dtype=np.float32)
        delta = np.zeros((T, D), dtype=np.float32)

        for j, c in enumerate(DYNAMIC_COLS):
            raw = frame[c].to_numpy(dtype=np.float64)
            lo, hi = a.clip_ranges.get(c, [-np.inf, np.inf])
            raw = np.clip(raw, lo, hi)
            seen = ~np.isnan(raw)
            mask[:, j] = seen.astype(np.float32)

            last = a.dynamic_mean[c]
            since = 0.0
            for t in range(T):
                if seen[t]:
                    last = raw[t]
                    since = 0.0
                else:
                    since += 1.0
                values[t, j] = last
                delta[t, j] = since
            values[:, j] = (values[:, j] - a.dynamic_mean[c]) / a.dynamic_std[c]
            delta[:, j] = np.log1p(delta[:, j]) / a.delta_scale

        stat = rec.static_vector()
        static_row = np.array(
            [
                (
                    (stat[c] if not np.isnan(stat[c]) else a.static_mean[c])
                    - a.static_mean[c]
                )
                / a.static_std[c]
                for c in STATIC_COLS
            ],
            dtype=np.float32,
        )
        static_block = np.tile(static_row, (T, 1))
        feats = np.concatenate([values, mask, delta, static_block], axis=1)
        return feats, rec.label[-T:].astype(np.int8), T

    def transform(self, records: list[PatientRecord]) -> TensorDataset:
        if self.artifacts is None:
            raise RuntimeError("call .fit() before .transform()")
        N = len(records)
        T = self.max_seq_len
        F = self.artifacts.n_features
        X = np.zeros((N, T, F), dtype=np.float32)
        y = np.zeros((N, T), dtype=np.int8)
        lengths = np.zeros(N, dtype=np.int32)
        pids: list[str] = []
        static_raw = np.zeros((N, len(STATIC_COLS)), dtype=np.float32)

        for i, rec in enumerate(records):
            feats, lab, n = self._encode_one(rec)
            X[i, :n] = feats
            y[i, :n] = lab
            lengths[i] = n
            pids.append(rec.pid)
            sv = rec.static_vector()
            static_raw[i] = [sv[c] for c in STATIC_COLS]

        return TensorDataset(
            X=X, y=y, lengths=lengths, pids=pids,
            static_raw=static_raw, feature_names=self.artifacts.feature_names,
        )

    def fit_transform(self, records: list[PatientRecord]) -> TensorDataset:
        return self.fit(records).transform(records)
