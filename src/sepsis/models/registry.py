"""Build a sequence model from a :class:`ModelConfig`."""

from __future__ import annotations

from sepsis.config import ModelConfig
from sepsis.models.sequence import (
    RNNRiskModel,
    TCNRiskModel,
    TransformerRiskModel,
    _BaseSequenceModel,
)

SEQUENCE_MODELS = ("lstm", "gru", "tcn", "transformer")


def is_sequence_model(name: str) -> bool:
    return name.lower() in SEQUENCE_MODELS


def build_sequence_model(cfg: ModelConfig, input_size: int) -> _BaseSequenceModel:
    name = cfg.name.lower()
    if name in {"lstm", "gru"}:
        return RNNRiskModel(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            cell=name,
            bidirectional=cfg.bidirectional,
        )
    if name == "tcn":
        return TCNRiskModel(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=max(cfg.num_layers, 3),
            kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.dropout,
        )
    if name == "transformer":
        return TransformerRiskModel(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
    raise ValueError(f"{cfg.name!r} is not a sequence model ({SEQUENCE_MODELS})")
