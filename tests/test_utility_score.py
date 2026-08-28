"""The utility score must match the published PhysioNet/CinC 2019 reference."""

from __future__ import annotations

import numpy as np
import pytest

from sepsis.evaluation.utility_score import (
    best_threshold_by_utility,
    normalized_utility,
    prediction_utility,
)


def test_reference_worked_example():
    # From evaluate_sepsis_score.py docstring.
    labels = [0, 0, 0, 0, 1, 1]
    predictions = [0, 0, 1, 1, 1, 1]
    assert prediction_utility(labels, predictions) == pytest.approx(3.388888888888889)


def test_inaction_on_nonseptic_scores_zero():
    labels = np.zeros(30, dtype=int)
    inaction = np.zeros(30, dtype=int)
    assert prediction_utility(labels, inaction) == pytest.approx(0.0)


def test_normalised_inaction_is_the_zero_reference_even_when_septic():
    # For a septic stay, never alarming is penalised (missed onset), but the
    # cohort normalisation maps the inaction policy to exactly 0 by construction.
    labels = np.zeros(30, dtype=int)
    labels[20:] = 1
    inaction = np.zeros(30, dtype=int)
    assert prediction_utility(labels, inaction) < 0.0
    br = normalized_utility([labels], [inaction])
    assert br.normalized == pytest.approx(0.0, abs=1e-9)


def test_oracle_prediction_scores_one():
    labels = np.zeros(40, dtype=int)
    labels[25:] = 1  # onset (shifted) at t=25
    # Oracle: fire over [t_sepsis-12, t_sepsis+3] where t_sepsis = 25 + 6 = 31
    oracle = np.zeros(40, dtype=int)
    oracle[19:35] = 1
    br = normalized_utility([labels], [oracle])
    assert br.normalized == pytest.approx(1.0, abs=1e-9)


def test_false_positive_penalty_on_nonseptic():
    labels = np.zeros(10, dtype=int)
    preds = np.zeros(10, dtype=int)
    preds[3] = 1
    assert prediction_utility(labels, preds) == pytest.approx(-0.05)


def test_best_threshold_selects_a_probability_in_grid():
    rng = np.random.default_rng(0)
    seq_labels, seq_scores = [], []
    for _ in range(40):
        n = rng.integers(20, 45)
        lab = np.zeros(n, dtype=int)
        if rng.random() < 0.3:
            onset = rng.integers(10, n)
            lab[onset:] = 1
        score = np.clip(0.1 + 0.7 * lab + rng.normal(0, 0.15, n), 0, 1)
        seq_labels.append(lab)
        seq_scores.append(score)
    thr, br = best_threshold_by_utility(seq_labels, seq_scores)
    assert 0.0 < thr < 1.0
    assert -1.0 <= br.normalized <= 1.0
