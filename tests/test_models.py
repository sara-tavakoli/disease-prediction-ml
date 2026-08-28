from __future__ import annotations

import pytest
import torch

from sepsis.config import ModelConfig
from sepsis.models.losses import MaskedFocalLoss, build_loss
from sepsis.models.registry import build_sequence_model

MODELS = ["lstm", "gru", "tcn", "transformer"]


@pytest.mark.parametrize("name", MODELS)
def test_output_shape_and_padding_invariance(name):
    torch.manual_seed(0)
    m = build_sequence_model(ModelConfig(name=name, hidden_size=32, num_layers=2), input_size=16)
    m.eval()
    B, T, F = 4, 20, 16
    x = torch.randn(B, T, F)
    pad = torch.ones(B, T, dtype=torch.bool)
    pad[0, 12:] = False  # first sequence is length 12
    with torch.no_grad():
        out = m(x, pad)
    assert out.shape == (B, T)

    # padding the tail with garbage must not change the valid-region logits
    x2 = x.clone()
    x2[0, 12:] = 999.0
    with torch.no_grad():
        out2 = m(x2, pad)
    torch.testing.assert_close(out[0, :12], out2[0, :12], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", MODELS)
def test_strict_causality(name):
    """Perturbing input at time t must leave outputs at t' < t unchanged."""
    torch.manual_seed(0)
    m = build_sequence_model(ModelConfig(name=name, hidden_size=32, num_layers=2), input_size=8)
    m.eval()
    T = 16
    x = torch.randn(1, T, 8)
    pad = torch.ones(1, T, dtype=torch.bool)
    with torch.no_grad():
        base = m(x, pad)
    x_pert = x.clone()
    x_pert[0, 10:] += 5.0
    with torch.no_grad():
        pert = m(x_pert, pad)
    torch.testing.assert_close(base[0, :10], pert[0, :10], rtol=1e-4, atol=1e-4)


def test_focal_loss_ignores_padding():
    logits = torch.zeros(2, 5)
    targets = torch.zeros(2, 5)
    targets[0, 0] = 1.0
    full = torch.ones(2, 5, dtype=torch.bool)
    half = full.clone()
    half[:, 3:] = False
    loss_fn = MaskedFocalLoss(gamma=2.0)
    # extreme padded logits must not leak into the masked loss
    logits_pad = logits.clone()
    logits_pad[:, 3:] = 50.0
    assert torch.isclose(loss_fn(logits, targets, half), loss_fn(logits_pad, targets, half))


def test_build_loss_names():
    for n in ["focal", "weighted_bce", "bce"]:
        assert build_loss(n) is not None
    with pytest.raises(ValueError):
        build_loss("nope")


def test_mc_dropout_mode_activates_only_dropout():
    m = build_sequence_model(ModelConfig(name="gru", dropout=0.5), input_size=8)
    m.mc_dropout_mode()
    drops = [mod.training for mod in m.modules() if isinstance(mod, torch.nn.Dropout)]
    assert drops and all(drops)
