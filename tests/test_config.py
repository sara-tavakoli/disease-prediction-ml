from __future__ import annotations

import pytest

from sepsis.config import ExperimentConfig


def test_compose_yaml_files_left_to_right():
    cfg = ExperimentConfig.load("configs/base.yaml", "configs/model_lstm.yaml")
    assert cfg.model.name == "lstm"
    assert cfg.output_dir == "artifacts/lstm"
    assert cfg.data.source == "synthetic"  # inherited from base


def test_dotted_overrides_and_type_coercion():
    cfg = ExperimentConfig.load(
        "configs/base.yaml",
        overrides=["train.epochs=3", "model.dropout=0.4", "train.amp=true",
                   "train.pos_weight=none", "data.source=physionet"],
    )
    assert cfg.train.epochs == 3 and isinstance(cfg.train.epochs, int)
    assert cfg.model.dropout == 0.4
    assert cfg.train.amp is True
    assert cfg.train.pos_weight is None
    assert cfg.data.source == "physionet"


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.load("configs/base.yaml", overrides=["train.nonsense=1"])


def test_validation_catches_bad_values():
    with pytest.raises(ValueError):
        ExperimentConfig.load("configs/base.yaml", overrides=["model.name=svm"])
    with pytest.raises(ValueError):
        ExperimentConfig.load(
            "configs/base.yaml",
            overrides=["model.name=transformer", "model.hidden_size=100",
                       "model.num_heads=8"],
        )


def test_roundtrip_dict(tmp_path):
    cfg = ExperimentConfig.load("configs/base.yaml", "configs/model_tcn.yaml")
    p = tmp_path / "c.json"
    cfg.save(p)
    import json

    again = ExperimentConfig.from_dict(json.loads(p.read_text()))
    assert again.to_dict() == cfg.to_dict()
