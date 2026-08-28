"""Typed configuration objects with YAML loading and CLI-style overrides.

A deliberately small, dependency-free alternative to Hydra: dataclasses that
validate themselves, load from ``configs/*.yaml`` and accept ``a.b=c`` overrides
so every experiment is fully described by a single serialisable object.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class DataConfig:
    root: str = "data/raw"
    processed_dir: str = "data/processed"
    source: str = "physionet"  # "physionet" | "synthetic"
    synthetic_n_patients: int = 4000
    synthetic_prevalence: float = 0.08
    resample_hours: int = 1
    max_seq_len: int = 336  # 14 days; longer stays are left-truncated
    label_shift_hours: int = 6
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    group_by_hospital: bool = True  # split so set A / set B never mix
    seed: int = 20190804

    def __post_init__(self) -> None:
        if self.source not in {"physionet", "synthetic"}:
            raise ValueError(f"unknown data source: {self.source}")
        if not 0 < self.val_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError("val/test fractions must lie in (0, 1)")
        if self.val_fraction + self.test_fraction >= 0.9:
            raise ValueError("train split would be < 10%")


@dataclass
class ModelConfig:
    name: str = "transformer"  # transformer|lstm|gru|tcn|lightgbm
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 8  # transformer only
    dropout: float = 0.2
    tcn_kernel_size: int = 3  # tcn only
    bidirectional: bool = False  # rnn only; leaks future -> keep False
    # gradient-boosting baseline
    gbm_num_leaves: int = 64
    gbm_learning_rate: float = 0.03
    gbm_n_estimators: int = 600
    gbm_window: int = 8  # hours of history per tabular row

    def __post_init__(self) -> None:
        allowed = {"transformer", "lstm", "gru", "tcn", "lightgbm"}
        if self.name not in allowed:
            raise ValueError(f"model.name must be one of {sorted(allowed)}")
        if self.name == "transformer" and self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    loss: str = "focal"  # focal|weighted_bce|bce
    focal_gamma: float = 2.0
    pos_weight: float | None = None  # None -> derived from train prevalence
    early_stopping_patience: int = 7
    monitor: str = "val_auprc"  # metric maximised for checkpointing
    num_workers: int = 0
    device: str = "auto"  # auto|cpu|cuda|mps
    amp: bool = False
    seed: int = 20190804
    mlflow_experiment: str = "sepsis-early-warning"
    run_name: str | None = None


@dataclass
class UncertaintyConfig:
    calibration: str = "isotonic"  # isotonic|platt|temperature|none
    conformal_alpha: float = 0.1  # target miscoverage for risk sets
    conformal_mondrian: bool = True  # class-conditional (per-timestep label)
    mc_dropout_samples: int = 30


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    output_dir: str = "artifacts/run"
    tags: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ IO --
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        return _build(cls, raw)

    @classmethod
    def load(
        cls,
        *yaml_paths: str | Path,
        overrides: list[str] | None = None,
    ) -> ExperimentConfig:
        merged: dict[str, Any] = {}
        for p in yaml_paths:
            p = Path(p)
            if not p.is_absolute() and not p.exists():
                p = DEFAULT_CONFIG_DIR / p
            with open(p) as fh:
                _deep_update(merged, yaml.safe_load(fh) or {})
        for item in overrides or []:
            _apply_override(merged, item)
        return cls.from_dict(merged)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _coerce(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _apply_override(cfg: dict[str, Any], item: str) -> None:
    if "=" not in item:
        raise ValueError(f"override must look like 'a.b=c', got {item!r}")
    dotted, _, value = item.partition("=")
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = _coerce(value)


def _build(tp: type, raw: Any) -> Any:
    if not is_dataclass(tp):
        return raw
    kwargs: dict[str, Any] = {}
    valid = {f.name for f in fields(tp)}
    # ``from __future__ import annotations`` turns field types into strings, so
    # resolve them once to real classes for the nested-dataclass check.
    try:
        hints = typing.get_type_hints(tp)
    except Exception:  # pragma: no cover - defensive
        hints = {}
    for key, value in (raw or {}).items():
        if key not in valid:
            raise ValueError(f"{tp.__name__}: unknown config key {key!r}")
        ftype = hints.get(key)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        else:
            kwargs[key] = value
    return tp(**kwargs)
