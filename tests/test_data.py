from __future__ import annotations

import numpy as np

from sepsis.constants import CHANNELS, SEPSIS_LABEL_SHIFT_HOURS
from sepsis.data.psv import load_psv
from sepsis.data.splits import make_splits
from sepsis.data.synthetic import generate_cohort, generate_patient, write_cohort


def test_synthetic_is_deterministic():
    a = generate_cohort(50, 0.2, seed=123)
    b = generate_cohort(50, 0.2, seed=123)
    assert [r.pid for r in a] == [r.pid for r in b]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x.label, y.label)
        np.testing.assert_allclose(x.frame.to_numpy(), y.frame.to_numpy(), equal_nan=True)


def test_synthetic_prevalence_is_respected():
    recs = generate_cohort(500, 0.1, seed=1)
    rate = np.mean([r.is_septic for r in recs])
    assert abs(rate - 0.1) < 1e-9


def test_labels_have_the_six_hour_lead():
    rng = np.random.default_rng(0)
    rec = generate_patient("p0", rng, force_septic=True, prevalence=1.0)
    onset = rec.onset_hour
    assert onset is not None
    # every positive hour is >= onset; the label turned on at onset (=t_sepsis-6)
    assert rec.label[onset] == 1 and rec.label[:onset].sum() == 0
    assert SEPSIS_LABEL_SHIFT_HOURS == 6


def test_columns_and_missingness_shape():
    rng = np.random.default_rng(2)
    rec = generate_patient("p1", rng, force_septic=False)
    assert list(rec.frame.columns) == CHANNELS
    # labs are mostly missing, ICULOS never missing
    assert rec.frame["ICULOS"].notna().all()
    assert rec.frame["Fibrinogen"].isna().mean() > 0.5


def test_roundtrip_through_psv(tmp_path):
    recs = generate_cohort(6, 0.5, seed=9)
    write_cohort(recs, tmp_path)
    back = load_psv(tmp_path / "training_setSYN" / f"{recs[0].pid}.psv", source="setSYN")
    np.testing.assert_array_equal(back.label, recs[0].label)
    assert back.frame.shape == recs[0].frame.shape


def test_patient_level_split_has_no_shared_ids():
    recs = generate_cohort(120, 0.2, seed=3)
    sp = make_splits(recs, 0.2, 0.2, seed=3)
    ids = [{r.pid for r in part} for part in (sp.train, sp.val, sp.test)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert sum(len(s) for s in ids) == 120
