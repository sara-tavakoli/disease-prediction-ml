"""Subgroup performance and fairness gaps.

Sepsis models have documented disparities across sex and age. We stratify the
test hours by protected / clinically-relevant attributes and report, per group:
AUROC, AUPRC, calibration error, and the alarm rate + true-positive rate at the
**shared** operating threshold chosen on the full cohort. The headline numbers
are the max-min gaps:

* ``tpr_gap``           -- equal-opportunity violation (Hardt et al., 2016),
* ``alarm_rate_gap``    -- demographic-parity violation of the alert,
* ``calibration_gap``   -- worst-group minus best-group ECE.
"""

from __future__ import annotations

import numpy as np

from sepsis.constants import STATIC_COLS
from sepsis.evaluation.metrics import auprc, auroc
from sepsis.uncertainty.calibration import expected_calibration_error


def subgroup_assignments(static_raw: np.ndarray) -> dict[str, np.ndarray]:
    """Map each stay to a label for every subgroup axis.

    ``static_raw`` columns follow :data:`sepsis.constants.STATIC_COLS`
    = [Age, Gender, Unit1, Unit2, HospAdmTime].
    """
    col = {c: static_raw[:, i] for i, c in enumerate(STATIC_COLS)}
    age = col["Age"]
    age_band = np.select(
        [age < 40, age < 65, age < 80],
        ["<40", "40-64", "65-79"],
        default="80+",
    )
    sex = np.where(np.isnan(col["Gender"]), "unknown",
                   np.where(col["Gender"] >= 0.5, "male", "female"))
    unit = np.where(col["Unit1"] >= 0.5, "MICU",
                    np.where(col["Unit2"] >= 0.5, "SICU", "unknown"))
    return {"sex": sex.astype(object), "age_band": age_band.astype(object),
            "icu_unit": unit.astype(object)}


def _expand_to_hours(stay_labels: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.repeat(stay_labels[i], int(n)) for i, n in enumerate(lengths)]
    )


def subgroup_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    static_raw: np.ndarray,
    lengths: np.ndarray,
    threshold: float = 0.5,
    min_group_hours: int = 200,
) -> dict[str, dict]:
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    axes = subgroup_assignments(static_raw)
    report: dict[str, dict] = {}

    for axis, stay_group in axes.items():
        hour_group = _expand_to_hours(stay_group, lengths)
        groups: dict[str, dict[str, float]] = {}
        for g in sorted(set(hour_group.tolist())):
            m = hour_group == g
            if m.sum() < min_group_hours or y_true[m].max() == y_true[m].min():
                continue
            yt, yp = y_true[m], y_prob[m]
            pred = (yp >= threshold).astype(int)
            tp = int(np.sum(pred & (yt == 1)))
            fn = int(np.sum((pred == 0) & (yt == 1)))
            fp = int(np.sum(pred & (yt == 0)))
            tn = int(np.sum((pred == 0) & (yt == 0)))
            groups[g] = {
                "hours": int(m.sum()),
                "prevalence": float(yt.mean()),
                "auroc": auroc(yt, yp),
                "auprc": auprc(yt, yp),
                "tpr": tp / (tp + fn) if tp + fn else float("nan"),
                "fpr": fp / (fp + tn) if fp + tn else float("nan"),
                "alarm_rate": float(pred.mean()),
                "ece": expected_calibration_error(yt, yp)["ece"],
            }

        report[axis] = {
            "groups": groups,
            "auroc_gap": _max_min_gap(groups, "auroc"),
            "tpr_gap": _max_min_gap(groups, "tpr"),
            "alarm_rate_gap": _max_min_gap(groups, "alarm_rate"),
            "calibration_gap": _max_min_gap(groups, "ece"),
        }
    return report


def _max_min_gap(groups: dict[str, dict[str, float]], key: str) -> float:
    vals = [v[key] for v in groups.values() if np.isfinite(v[key])]
    return float(max(vals) - min(vals)) if len(vals) > 1 else float("nan")
