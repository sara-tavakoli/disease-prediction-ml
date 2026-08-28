from __future__ import annotations

import numpy as np

from sepsis.features.tabular import WindowFeatureExtractor
from sepsis.uncertainty.calibration import (
    expected_calibration_error,
    fit_calibrator,
    reliability_curve,
)


def test_window_features_row_count_matches_valid_hours(tensors):
    td = tensors["train"]
    tab = WindowFeatureExtractor(window=6).transform(td)
    assert len(tab) == int(td.lengths.sum())
    assert tab.X.shape[1] == len(tab.feature_names)
    assert set(np.unique(tab.groups)).issubset(set(range(len(td))))
    assert tab.times.max() < td.X.shape[1]


def test_window_features_are_causal(tensors):
    """Row at (stay i, time t) must be identical whether or not later hours
    exist in the sequence."""
    td = tensors["val"]
    fe = WindowFeatureExtractor(window=5)
    full = fe.transform(td)

    import dataclasses

    i = 3
    n = int(td.lengths[i])
    cut = max(6, n // 2)
    td2 = dataclasses.replace(
        td,
        X=td.X[i : i + 1].copy(),
        y=td.y[i : i + 1].copy(),
        lengths=np.array([cut], dtype=np.int32),
        pids=[td.pids[i]],
        static_raw=td.static_raw[i : i + 1],
    )
    part = fe.transform(td2)
    rows_full = full.X[(full.groups == i) & (full.times < cut)]
    np.testing.assert_allclose(rows_full, part.X, rtol=1e-5, atol=1e-5)


def test_isotonic_calibration_reduces_ece():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.2).astype(int)
    # badly miscalibrated: squash toward 0.5
    p = 0.5 + 0.15 * (2 * y - 1) + rng.normal(0, 0.05, 4000)
    p = np.clip(p, 0, 1)
    cal = fit_calibrator(y[:2000], p[:2000], "isotonic")
    ece_before = expected_calibration_error(y[2000:], p[2000:])["ece"]
    ece_after = expected_calibration_error(y[2000:], cal.transform(p[2000:]))["ece"]
    assert ece_after < ece_before


def test_temperature_scaling_preserves_ranking():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.3).astype(int)
    p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, 3000), 1e-3, 1 - 1e-3)
    cal = fit_calibrator(y, p, "temperature")
    q = cal.transform(p)
    assert np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(q)))[0, 1] > 0.999


def test_reliability_curve_bins_are_within_unit_interval():
    rng = np.random.default_rng(2)
    y = (rng.random(1000) < 0.4).astype(int)
    p = rng.random(1000)
    rc = reliability_curve(y, p, n_bins=10)
    assert np.all((rc["confidence"] >= 0) & (rc["confidence"] <= 1))
    assert np.isclose(rc["weight"].sum(), 1.0)
