"""Sliding-window tabular features for the gradient-boosting baseline.

For each ICU hour ``t`` we summarise the preceding ``window`` hours of every
dynamic channel with statistics that a clinician would recognise -- current
value, recent mean / min / max / dispersion, linear trend, and how much of the
window was actually measured -- then append the standardised static covariates.
The row is labelled with ``y[t]`` and tagged with its stay index so grouped
cross-validation and the sequence-level utility score can be reconstructed.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from sepsis.constants import DYNAMIC_COLS, STATIC_COLS
from sepsis.data.preprocess import TensorDataset

_AGG = ("last", "mean", "min", "max", "std", "slope", "obs_frac", "delta")


@dataclasses.dataclass
class WindowedTable:
    X: np.ndarray            # (rows, n_features)
    y: np.ndarray            # (rows,)
    groups: np.ndarray       # (rows,) stay index into the source TensorDataset
    times: np.ndarray        # (rows,) hour index within the stay
    feature_names: list[str]

    def __len__(self) -> int:
        return self.X.shape[0]


class WindowFeatureExtractor:
    def __init__(self, window: int = 8):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = int(window)
        self._value_idx = list(range(len(DYNAMIC_COLS)))
        self._mask_idx = list(range(len(DYNAMIC_COLS), 2 * len(DYNAMIC_COLS)))
        self._delta_idx = list(range(2 * len(DYNAMIC_COLS), 3 * len(DYNAMIC_COLS)))
        self._static_idx = list(range(3 * len(DYNAMIC_COLS),
                                      3 * len(DYNAMIC_COLS) + len(STATIC_COLS)))
        self.feature_names = [
            f"{c}__{agg}" for c in DYNAMIC_COLS for agg in _AGG
        ] + [f"{c}__static" for c in STATIC_COLS]

    def transform(self, td: TensorDataset) -> WindowedTable:
        w = self.window
        rows_x: list[np.ndarray] = []
        rows_y: list[int] = []
        groups: list[int] = []
        times: list[int] = []
        n_dyn = len(DYNAMIC_COLS)
        ramp = np.arange(w, dtype=np.float64)
        ramp -= ramp.mean()
        ramp_ss = float((ramp * ramp).sum())

        for i in range(len(td)):
            n = int(td.lengths[i])
            seq = td.X[i]
            val = seq[:, self._value_idx]
            msk = seq[:, self._mask_idx]
            dlt = seq[:, self._delta_idx]
            static_row = seq[0, self._static_idx]
            for t in range(n):
                lo = max(0, t - w + 1)
                win = val[lo:t + 1]                       # (win_len, n_dyn)
                win_mask = msk[lo:t + 1]
                pad = w - win.shape[0]
                if pad:
                    win = np.vstack([np.repeat(win[:1], pad, axis=0), win])
                    win_mask = np.vstack(
                        [np.zeros((pad, n_dyn), dtype=win_mask.dtype), win_mask]
                    )
                mean = win.mean(axis=0)
                slope = ((ramp[:, None] * (win - mean)).sum(axis=0)) / ramp_ss
                feat = np.concatenate(
                    [
                        win[-1],                         # last
                        mean,                            # mean
                        win.min(axis=0),                 # min
                        win.max(axis=0),                 # max
                        win.std(axis=0),                 # std
                        slope,                           # linear trend
                        win_mask.mean(axis=0),           # observed fraction
                        dlt[t],                          # recency
                        static_row,
                    ]
                ).astype(np.float32)
                rows_x.append(feat)
                rows_y.append(int(td.y[i, t]))
                groups.append(i)
                times.append(t)

        return WindowedTable(
            X=np.vstack(rows_x),
            y=np.asarray(rows_y, dtype=np.int8),
            groups=np.asarray(groups, dtype=np.int32),
            times=np.asarray(times, dtype=np.int32),
            feature_names=self.feature_names,
        )
