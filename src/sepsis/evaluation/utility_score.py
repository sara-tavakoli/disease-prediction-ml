"""The official PhysioNet/CinC 2019 time-dependent utility score.

Faithful re-implementation of ``evaluate_sepsis_score.py`` (Reyna et al., 2019;
https://github.com/physionetchallenges/evaluation-2019). A positive prediction
is rewarded on a ramp that peaks 6 hours before sepsis onset
(``dt_optimal = -6``), starts paying off at ``dt_early = -12`` and stops scoring
after ``dt_late = +3``; missed onsets are penalised, and false alarms on
non-septic stays cost ``u_fp = -0.05`` per hour. The cohort score is normalised
so that the always-negative "inaction" policy scores 0 and the oracle scores 1.

``prediction_utility`` is kept as a transparent per-timestep loop so it can be
unit-tested against the published worked examples; the cohort driver is
vectorised over patients.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from sepsis.constants import (
    UTILITY_DT_EARLY,
    UTILITY_DT_LATE,
    UTILITY_DT_OPTIMAL,
    UTILITY_MAX_TP,
    UTILITY_MIN_FN,
    UTILITY_U_FP,
    UTILITY_U_TN,
)


def prediction_utility(
    labels: np.ndarray,
    predictions: np.ndarray,
    dt_early: int = UTILITY_DT_EARLY,
    dt_optimal: int = UTILITY_DT_OPTIMAL,
    dt_late: int = UTILITY_DT_LATE,
    max_u_tp: float = UTILITY_MAX_TP,
    min_u_fn: float = UTILITY_MIN_FN,
    u_fp: float = UTILITY_U_FP,
    u_tn: float = UTILITY_U_TN,
) -> float:
    labels = np.asarray(labels).astype(int).ravel()
    predictions = np.asarray(predictions).astype(int).ravel()
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")

    if np.any(labels):
        is_septic = True
        t_sepsis = int(np.argmax(labels)) - dt_optimal
    else:
        is_septic = False
        t_sepsis = np.inf

    m_1 = float(max_u_tp) / float(dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = float(-max_u_tp) / float(dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = float(min_u_fn) / float(dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal

    n = len(labels)
    u = np.zeros(n)
    for t in range(n):
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u[t] = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                elif t <= t_sepsis + dt_late:
                    u[t] = m_2 * (t - t_sepsis) + b_2
            elif (not is_septic) and predictions[t]:
                u[t] = u_fp
            elif is_septic and (not predictions[t]):
                if t <= t_sepsis + dt_optimal:
                    u[t] = 0.0
                elif t <= t_sepsis + dt_late:
                    u[t] = m_3 * (t - t_sepsis) + b_3
            elif (not is_septic) and (not predictions[t]):
                u[t] = u_tn
    return float(np.sum(u))


def _best_and_inaction(labels: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels).astype(int).ravel()
    n = len(labels)
    best = np.zeros(n, dtype=int)
    if np.any(labels):
        t_sepsis = int(np.argmax(labels)) - UTILITY_DT_OPTIMAL
        lo = max(0, t_sepsis + UTILITY_DT_EARLY)
        hi = min(t_sepsis + UTILITY_DT_LATE + 1, n)
        best[lo:hi] = 1
    return (
        prediction_utility(labels, best),
        prediction_utility(labels, np.zeros(n, dtype=int)),
    )


@dataclasses.dataclass
class UtilityBreakdown:
    normalized: float
    observed: float
    best: float
    inaction: float
    threshold: float


def normalized_utility(
    seq_labels: list[np.ndarray],
    seq_binary_preds: list[np.ndarray],
    threshold: float = float("nan"),
) -> UtilityBreakdown:
    """Cohort-level normalised utility for a list of per-stay binary vectors."""
    obs = best = inact = 0.0
    for lab, pred in zip(seq_labels, seq_binary_preds):
        obs += prediction_utility(lab, pred)
        b, i = _best_and_inaction(lab)
        best += b
        inact += i
    denom = best - inact
    norm = (obs - inact) / denom if abs(denom) > 1e-12 else float("nan")
    return UtilityBreakdown(norm, obs, best, inact, threshold)


def best_threshold_by_utility(
    seq_labels: list[np.ndarray],
    seq_scores: list[np.ndarray],
    grid: np.ndarray | None = None,
) -> tuple[float, UtilityBreakdown]:
    """Sweep probability thresholds and return the one maximising cohort utility.

    Choosing the alarm threshold on the utility curve -- rather than on Youden's
    J or a fixed 0.5 -- is the operating-point selection the challenge rewards.
    """
    if grid is None:
        grid = np.round(np.linspace(0.02, 0.8, 40), 4)
    best_t, best_break = 0.5, None
    for thr in grid:
        preds = [(s >= thr).astype(int) for s in seq_scores]
        br = normalized_utility(seq_labels, preds, threshold=float(thr))
        if best_break is None or br.normalized > best_break.normalized:
            best_t, best_break = float(thr), br
    assert best_break is not None
    return best_t, best_break
