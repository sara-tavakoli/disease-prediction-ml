"""Discrimination and calibration metrics on flattened hour-level predictions.

Thin wrappers over scikit-learn plus two operating-point summaries that matter
clinically: sensitivity at a fixed specificity (how many onsets we catch if we
cap the false-alarm rate) and its dual.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def _clean(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    keep = (y_true >= 0) & np.isfinite(y_score)
    return y_true[keep].astype(int), y_score[keep]


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true, y_score = _clean(y_true, y_score)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true, y_score = _clean(y_true, y_score)
    if y_true.max() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true, y_prob = _clean(y_true, y_prob)
    return float(brier_score_loss(y_true, np.clip(y_prob, 0.0, 1.0)))


def sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, specificity: float = 0.85
) -> tuple[float, float]:
    """Return ``(sensitivity, threshold)`` at the requested specificity."""
    y_true, y_score = _clean(y_true, y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.flatnonzero((1.0 - fpr) >= specificity)
    if ok.size == 0:
        return 0.0, 1.0
    j = ok[np.argmax(tpr[ok])]
    return float(tpr[j]), float(thr[j])


def specificity_at_sensitivity(
    y_true: np.ndarray, y_score: np.ndarray, sensitivity: float = 0.85
) -> tuple[float, float]:
    y_true, y_score = _clean(y_true, y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = np.flatnonzero(tpr >= sensitivity)
    if ok.size == 0:
        return 0.0, 0.0
    j = ok[np.argmin(fpr[ok])]
    return float(1.0 - fpr[j]), float(thr[j])


def classification_summary(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_true, y_score = _clean(y_true, y_score)
    pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    sens, thr85 = sensitivity_at_specificity(y_true, y_score, 0.85)
    spec, _ = specificity_at_sensitivity(y_true, y_score, 0.85)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(y_true.size),
        "prevalence": float(y_true.mean()),
        "auroc": auroc(y_true, y_score),
        "auprc": auprc(y_true, y_score),
        "brier": brier(y_true, np.clip(y_score, 0, 1)),
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "sensitivity_at_spec85": float(sens),
        "specificity_at_sens85": float(spec),
        "threshold_at_spec85": float(thr85),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
