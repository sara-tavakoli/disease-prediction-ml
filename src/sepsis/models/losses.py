"""Imbalance-aware, padding-safe losses for per-timestep binary labels.

Positive hours are ~2% of the corpus, so plain BCE collapses to the majority
class. Both losses below (a) ignore padded timesteps via the boolean pad mask
and (b) up-weight positives -- ``MaskedFocalLoss`` additionally down-weights
easy negatives with the Lin et al. (2017) focal term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    m = pad_mask.to(x.dtype)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


class MaskedBCE(nn.Module):
    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight)) if pos_weight else None,
            persistent=False,
        )

    def forward(self, logits, targets, pad_mask):
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        return _masked_mean(loss, pad_mask)


class MaskedFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight)) if pos_weight else None,
            persistent=False,
        )

    def forward(self, logits, targets, pad_mask):
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal = (1.0 - p_t).clamp_min(1e-6) ** self.gamma
        return _masked_mean(focal * ce, pad_mask)


def build_loss(name: str, focal_gamma: float = 2.0, pos_weight: float | None = None):
    name = name.lower()
    if name == "focal":
        return MaskedFocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    if name == "weighted_bce":
        return MaskedBCE(pos_weight=pos_weight if pos_weight else 10.0)
    if name == "bce":
        return MaskedBCE(pos_weight=None)
    raise ValueError(f"unknown loss {name!r}")
