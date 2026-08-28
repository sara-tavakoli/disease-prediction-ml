from __future__ import annotations

import numpy as np
import pytest

from sepsis.config import ExperimentConfig
from sepsis.data.preprocess import Preprocessor
from sepsis.data.splits import make_splits
from sepsis.data.synthetic import generate_cohort


@pytest.fixture(scope="session")
def cohort():
    return generate_cohort(n_patients=120, prevalence=0.2, seed=7)


@pytest.fixture(scope="session")
def splits(cohort):
    return make_splits(cohort, val_fraction=0.2, test_fraction=0.2, seed=7, group_by_hospital=False)


@pytest.fixture(scope="session")
def tensors(splits):
    pre = Preprocessor(max_seq_len=64).fit(splits.train)
    return {
        "pre": pre,
        "train": pre.transform(splits.train),
        "val": pre.transform(splits.val),
        "test": pre.transform(splits.test),
    }


@pytest.fixture
def smoke_config(tmp_path):
    return ExperimentConfig.load(
        "configs/base.yaml",
        "configs/smoke.yaml",
        overrides=[
            f"output_dir={tmp_path / 'run'}",
            "data.synthetic_n_patients=120",
            "train.epochs=1",
            "model.name=gru",
        ],
    )


@pytest.fixture(autouse=True)
def _quiet_np():
    np.seterr(all="ignore")
    yield
