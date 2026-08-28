"""Torch wrappers around :class:`TensorDataset` with padded-batch collation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from sepsis.data.preprocess import TensorDataset


class SequenceDataset(Dataset):
    """Yields ``(x[T,F], y[T], length)`` for one ICU stay."""

    def __init__(self, td: TensorDataset):
        self._x = td.X
        self._y = td.y
        self._len = td.lengths
        self.pids = td.pids
        self.feature_names = td.feature_names

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, i: int):
        n = int(self._len[i])
        return (
            torch.from_numpy(np.ascontiguousarray(self._x[i, :n])),
            torch.from_numpy(np.ascontiguousarray(self._y[i, :n])).float(),
            n,
        )


def collate_padded(batch):
    """Right-pad to the longest stay in the batch and build a boolean pad mask.

    ``pad_mask[i, t]`` is True where position ``t`` is real (not padding) -- the
    convention the models and the masked loss expect.
    """
    xs, ys, lens = zip(*batch)
    B = len(xs)
    T = max(lens)
    F = xs[0].shape[1]
    x = torch.zeros(B, T, F, dtype=torch.float32)
    y = torch.zeros(B, T, dtype=torch.float32)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    for i, (xi, yi, n) in enumerate(zip(xs, ys, lens)):
        x[i, :n] = xi
        y[i, :n] = yi
        pad_mask[i, :n] = True
    lengths = torch.tensor(lens, dtype=torch.long)
    return x, y, pad_mask, lengths
