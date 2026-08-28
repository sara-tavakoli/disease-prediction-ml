"""Leak-free, patient-level train/val/test splitting.

Two properties matter for a trustworthy sepsis benchmark:

1. **No patient crosses splits.** Windows from the same ICU stay are highly
   correlated; splitting at the window level inflates metrics.
2. **Stratify on the stay-level outcome** (ever-septic) so rare positives are
   represented in every split.

``group_by_hospital`` additionally supports the challenge's external-validation
protocol: train on set A, evaluate on the held-out set B (covariate shift across
institutions).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from sepsis.data.psv import PatientRecord


@dataclasses.dataclass(slots=True)
class GroupSplit:
    train: list[PatientRecord]
    val: list[PatientRecord]
    test: list[PatientRecord]

    def summary(self) -> dict[str, dict[str, float]]:
        def _stat(recs: list[PatientRecord]) -> dict[str, float]:
            n = len(recs)
            sep = sum(r.is_septic for r in recs)
            hrs = sum(r.n_hours for r in recs)
            pos_hrs = sum(int(r.label.sum()) for r in recs)
            return {
                "stays": n,
                "septic_stays": sep,
                "septic_rate": round(sep / max(n, 1), 4),
                "hours": hrs,
                "positive_hour_rate": round(pos_hrs / max(hrs, 1), 5),
            }

        return {k: _stat(v) for k, v in
                {"train": self.train, "val": self.val, "test": self.test}.items()}


def _stratified_indices(
    labels: np.ndarray, fracs: tuple[float, float, float], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx, val_idx, test_idx = [], [], []
    f_tr, f_va, _ = fracs
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(f_tr * n))
        n_va = int(round(f_va * n))
        train_idx.append(idx[:n_tr])
        val_idx.append(idx[n_tr:n_tr + n_va])
        test_idx.append(idx[n_tr + n_va:])
    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )


def make_splits(
    records: list[PatientRecord],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 20190804,
    group_by_hospital: bool = False,
    external_test_source: str = "setB",
) -> GroupSplit:
    rng = np.random.default_rng(seed)

    if group_by_hospital and len({r.source for r in records}) > 1:
        dev = [r for r in records if r.source != external_test_source]
        ext = [r for r in records if r.source == external_test_source]
        if dev and ext:
            labels = np.array([r.is_septic for r in dev], dtype=int)
            rel_val = val_fraction / (1.0 - test_fraction)
            tr, va, _ = _stratified_indices(
                labels, (1.0 - rel_val, rel_val, 0.0), rng
            )
            return GroupSplit(
                train=[dev[i] for i in tr],
                val=[dev[i] for i in va],
                test=ext,
            )

    labels = np.array([r.is_septic for r in records], dtype=int)
    f_tr = 1.0 - val_fraction - test_fraction
    tr, va, te = _stratified_indices(labels, (f_tr, val_fraction, test_fraction), rng)
    return GroupSplit(
        train=[records[i] for i in tr],
        val=[records[i] for i in va],
        test=[records[i] for i in te],
    )
