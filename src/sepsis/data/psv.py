"""Read PhysioNet/CinC 2019 pipe-separated patient files.

Each ``pXXXXXX.psv`` holds one ICU stay: rows are consecutive hours, columns are
the 34 physiological channels + 6 context columns + ``SepsisLabel``. Missing
measurements are ``NaN``. The label is already shifted +6h by the challenge
organisers for septic patients, so a row with ``SepsisLabel == 1`` means "sepsis
onset is <= 6 hours away or has occurred".
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis.constants import CHANNELS, LABEL_COL, STATIC_COLS
from sepsis.utils.logging import get_logger

log = get_logger("data.psv")


@dataclasses.dataclass(slots=True)
class PatientRecord:
    """One ICU stay in tidy form."""

    pid: str
    frame: pd.DataFrame  # index 0..T-1 (hours), columns = CHANNELS
    label: np.ndarray  # shape (T,), int8 in {0, 1}
    source: str = "unknown"  # e.g. "setA" / "setB" / "synthetic"

    @property
    def n_hours(self) -> int:
        return len(self.frame)

    @property
    def is_septic(self) -> bool:
        return bool(self.label.max()) if self.label.size else False

    @property
    def onset_hour(self) -> int | None:
        """First hour at which the (already +6h-shifted) label turns positive."""
        pos = np.flatnonzero(self.label == 1)
        return int(pos[0]) if pos.size else None

    def static_vector(self) -> dict[str, float]:
        row = self.frame.iloc[0]
        return {c: float(row[c]) for c in STATIC_COLS}


def load_psv(path: str | Path, source: str = "unknown") -> PatientRecord:
    path = Path(path)
    raw = pd.read_csv(path, sep="|")
    missing = [c for c in CHANNELS + [LABEL_COL] if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {missing}")
    label = raw[LABEL_COL].fillna(0).to_numpy(dtype=np.int8)
    frame = raw[CHANNELS].reset_index(drop=True).astype(np.float32)
    return PatientRecord(pid=path.stem, frame=frame, label=label, source=source)


def iter_patient_files(root: str | Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(psv_path, source_tag)`` for every patient under ``root``.

    Understands both the flat layout (``root/*.psv``) and the official nested
    layout (``root/training_setA/*.psv``, ``root/training_setB/*.psv``).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"data root does not exist: {root}")
    nested = sorted(root.glob("training_set*/"))
    if nested:
        for sub in nested:
            tag = "set" + sub.name.replace("training_set", "")
            for p in sorted(sub.glob("*.psv")):
                yield p, tag
    else:
        for p in sorted(root.glob("*.psv")):
            yield p, "flat"


def load_dataset(
    root: str | Path,
    limit: int | None = None,
    sources: set[str] | None = None,
) -> list[PatientRecord]:
    records: list[PatientRecord] = []
    for path, tag in iter_patient_files(root):
        if sources is not None and tag not in sources:
            continue
        records.append(load_psv(path, source=tag))
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise RuntimeError(f"no .psv files found under {root}")
    n_sep = sum(r.is_septic for r in records)
    log.info(
        "loaded %d stays (%.1f%% septic, %d hospital sources)",
        len(records),
        100.0 * n_sep / len(records),
        len({r.source for r in records}),
    )
    return records
