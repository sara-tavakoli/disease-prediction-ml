"""Split-conformal risk sets must attain their nominal coverage on exchangeable
hold-out data, and never emit an empty set by default."""

from __future__ import annotations

import numpy as np

from sepsis.uncertainty.conformal import ConformalRiskClassifier


def _synthetic_scores(n, seed, signal=2.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.15).astype(int)
    logits = signal * (y - 0.5) + rng.normal(0, 1.0, n)
    p1 = 1.0 / (1.0 + np.exp(-logits))
    return y, p1


def test_marginal_coverage_within_tolerance():
    y_cal, p_cal = _synthetic_scores(6000, seed=1)
    y_te, p_te = _synthetic_scores(6000, seed=2)
    cp = ConformalRiskClassifier(alpha=0.1, mondrian=False).fit(y_cal, p_cal)
    out = cp.evaluate(y_te, p_te)
    # finite-sample guarantee is >= 1 - alpha; allow a little slack downward
    assert out["empirical_coverage"] >= 0.87
    assert out["frac_empty"] == 0.0
    assert 1.0 <= out["avg_set_size"] <= 2.0


def test_mondrian_improves_minority_class_coverage():
    y_cal, p_cal = _synthetic_scores(8000, seed=3)
    y_te, p_te = _synthetic_scores(8000, seed=4)
    marg = ConformalRiskClassifier(alpha=0.1, mondrian=False).fit(y_cal, p_cal)
    mond = ConformalRiskClassifier(alpha=0.1, mondrian=True).fit(y_cal, p_cal)
    m1 = marg.evaluate(y_te, p_te)["coverage_class1"]
    d1 = mond.evaluate(y_te, p_te)["coverage_class1"]
    assert d1 >= min(0.85, m1)


def test_higher_alpha_gives_smaller_sets():
    y_cal, p_cal = _synthetic_scores(5000, seed=5)
    y_te, p_te = _synthetic_scores(5000, seed=6)
    tight = ConformalRiskClassifier(alpha=0.3, mondrian=False).fit(y_cal, p_cal)
    loose = ConformalRiskClassifier(alpha=0.05, mondrian=False).fit(y_cal, p_cal)
    assert (
        tight.evaluate(y_te, p_te)["avg_set_size"]
        <= loose.evaluate(y_te, p_te)["avg_set_size"] + 1e-9
    )
