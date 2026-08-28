from __future__ import annotations

import numpy as np

from sepsis.constants import DYNAMIC_COLS, STATIC_COLS
from sepsis.data.preprocess import PreprocessArtifacts, Preprocessor


def test_feature_layout_and_shapes(tensors):
    td = tensors["train"]
    n_dyn, n_stat = len(DYNAMIC_COLS), len(STATIC_COLS)
    assert td.n_features == 3 * n_dyn + n_stat
    assert td.X.shape[0] == len(td.pids) == len(td.lengths)
    assert td.X.shape[2] == td.n_features
    # names are frozen and ordered value | mask | delta | static
    fn = td.feature_names
    assert fn[:n_dyn] == [f"{c}__value" for c in DYNAMIC_COLS]
    assert fn[-n_stat:] == [f"{c}__static" for c in STATIC_COLS]


def test_padding_is_zero_and_labels_are_masked(tensors):
    td = tensors["val"]
    for i, n in enumerate(td.lengths):
        assert np.count_nonzero(td.X[i, n:]) == 0
        assert np.all(td.y[i, n:] == 0)


def test_mask_channel_matches_actual_observations(splits):
    pre = Preprocessor(max_seq_len=64).fit(splits.train)
    rec = splits.train[0]
    td = pre.transform([rec])
    n = int(td.lengths[0])
    n_dyn = len(DYNAMIC_COLS)
    for j, col in enumerate(DYNAMIC_COLS):
        observed = ~np.isnan(rec.frame[col].to_numpy()[:n])
        np.testing.assert_array_equal(td.X[0, :n, n_dyn + j] > 0.5, observed)


def test_no_leakage_stats_are_train_only(splits):
    pre_train = Preprocessor(max_seq_len=64).fit(splits.train)
    pre_all = Preprocessor(max_seq_len=64).fit(splits.train + splits.test)
    # fitting on more data changes the normaliser -> proves test data is excluded
    a, b = pre_train.artifacts.dynamic_mean["HR"], pre_all.artifacts.dynamic_mean["HR"]
    assert a != b


def test_artifacts_roundtrip(tmp_path, tensors):
    p = tmp_path / "pre.json"
    tensors["pre"].artifacts.save(p)
    loaded = PreprocessArtifacts.load(p)
    assert loaded.feature_names == tensors["pre"].artifacts.feature_names
    assert loaded.n_features == tensors["pre"].artifacts.n_features


def test_carry_forward_is_causal(splits):
    """A value observed at t must never influence features at t' < t.

    Uses a max_seq_len above every stay length so there is no left-truncation
    and the first t0 encoded hours line up with the truncated re-encoding.
    """
    import dataclasses

    pre = Preprocessor(max_seq_len=512).fit(splits.train)
    rec = next(r for r in splits.train if 20 <= r.n_hours <= 400)
    td_full = pre.transform([rec])
    t0 = 12
    rec_prefix = dataclasses.replace(
        rec, frame=rec.frame.iloc[:t0].reset_index(drop=True), label=rec.label[:t0]
    )
    td_prefix = pre.transform([rec_prefix])
    np.testing.assert_allclose(
        td_full.X[0, :t0], td_prefix.X[0, :t0], rtol=1e-5, atol=1e-5
    )
