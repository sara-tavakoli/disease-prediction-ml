from __future__ import annotations

import numpy as np
import pytest

from sepsis.evaluation.bootstrap import bootstrap_ci
from sepsis.evaluation.decision_curve import decision_curve
from sepsis.evaluation.metrics import (
    auprc,
    auroc,
    classification_summary,
    sensitivity_at_specificity,
)


def test_auroc_auprc_perfect_and_random():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.3, 0.4, 0.6, 0.7, 0.8, 0.8])
    assert auroc(y, s) == pytest.approx(1.0)
    assert auprc(y, s) == pytest.approx(1.0)


def test_sensitivity_at_specificity_monotone():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.2).astype(int)
    s = np.clip(0.2 * y + rng.normal(0, 0.3, 2000), 0, 1)
    sens_hi, _ = sensitivity_at_specificity(y, s, 0.95)
    sens_lo, _ = sensitivity_at_specificity(y, s, 0.70)
    assert sens_lo >= sens_hi - 1e-9


def test_classification_summary_contains_confusion_counts():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    s = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.3, 0.6, 0.7])
    out = classification_summary(y, s, threshold=0.5)
    assert out["tp"] + out["fp"] + out["fn"] + out["tn"] == len(y)
    assert 0.0 <= out["auroc"] <= 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    n_groups = 60
    y, s, g = [], [], []
    for gi in range(n_groups):
        n = rng.integers(10, 30)
        yi = (rng.random(n) < 0.25).astype(int)
        si = np.clip(0.3 * yi + rng.normal(0, 0.3, n), 0, 1)
        y.append(yi)
        s.append(si)
        g.append(np.full(n, gi))
    y, s, g = np.concatenate(y), np.concatenate(s), np.concatenate(g)
    ci = bootstrap_ci(auroc, y, s, g, n_resamples=300, seed=0)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.hi - ci.lo > 0


def test_decision_curve_model_beats_treat_none_somewhere():
    rng = np.random.default_rng(2)
    y = (rng.random(3000) < 0.1).astype(int)
    p = np.clip(0.6 * y + rng.normal(0, 0.15, 3000), 0, 1)
    dc = decision_curve(y, p)
    assert np.any(dc.net_benefit_model > dc.net_benefit_none)
