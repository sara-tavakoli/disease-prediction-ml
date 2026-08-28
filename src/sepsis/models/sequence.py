"""Causal sequence encoders that emit a sepsis-risk logit at every ICU hour.

Every model here is strictly **causal**: the prediction at hour ``t`` depends
only on hours ``<= t``. RNNs are unidirectional, the TCN uses left-only
("causal") padding, and the Transformer combines a subsequent-position mask with
the batch padding mask. This mirrors the online setting of the challenge, where
a score must be emitted each hour from the data available so far.

Shared interface
----------------
``forward(x, pad_mask) -> logits``  with ``x: (B, T, F)``, ``pad_mask: (B, T)``
(True = real timestep) and ``logits: (B, T)``.
``mc_dropout_mode()``  puts *only* dropout layers into training mode so
:mod:`sepsis.uncertainty.mc_dropout` can sample a predictive distribution.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _BaseSequenceModel(nn.Module):
    is_sequence = True

    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = int(input_size)

    def mc_dropout_mode(self) -> None:
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x, pad_mask))


# --------------------------------------------------------------------------- #
# Recurrent
# --------------------------------------------------------------------------- #
class RNNRiskModel(_BaseSequenceModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell: str = "lstm",
        bidirectional: bool = False,
    ):
        super().__init__(input_size)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell.lower()]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x, pad_mask=None):
        lengths = (
            pad_mask.sum(dim=1).clamp_min(1).cpu()
            if pad_mask is not None
            else torch.full((x.size(0),), x.size(1))
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return self.head(self.drop(out)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Temporal Convolutional Network (Bai et al., 2018)
# --------------------------------------------------------------------------- #
class _Chomp(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[..., : -self.size].contiguous() if self.size else x


class _TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
            ),
            _Chomp(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(c_out, c_out, kernel, padding=pad, dilation=dilation)
            ),
            _Chomp(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.act = nn.ReLU()

    def forward(self, x):
        res = x if self.down is None else self.down(x)
        return self.act(self.net(x) + res)


class TCNRiskModel(_BaseSequenceModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__(input_size)
        layers = []
        c_in = input_size
        for i in range(num_layers):
            layers.append(_TCNBlock(c_in, hidden_size, kernel_size, dilation=2**i, dropout=dropout))
            c_in = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, pad_mask=None):
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# Transformer encoder
# --------------------------------------------------------------------------- #
class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerRiskModel(_BaseSequenceModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
        max_len: int = 512,
    ):
        super().__init__(input_size)
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.pos = _PositionalEncoding(hidden_size, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # disable the nested-tensor fast path: it is incompatible with an
        # explicit float/bool attn mask and unimplemented on some backends (MPS)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self._last_attn: list[torch.Tensor] | None = None

    def forward(self, x, pad_mask=None):
        B, T, _ = x.shape
        h = self.pos(self.in_proj(x))
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        key_padding = (~pad_mask) if pad_mask is not None else None
        h = self.encoder(h, mask=causal, src_key_padding_mask=key_padding)
        return self.head(self.drop(h)).squeeze(-1)

    @torch.no_grad()
    def attention_rollout(self, x, pad_mask=None) -> torch.Tensor:
        """Attention-rollout (Abnar & Zuidema, 2020): product of per-layer,
        head-averaged attention with a residual term. Returns ``(B, T, T)``."""
        self.eval()
        B, T, _ = x.shape
        h = self.pos(self.in_proj(x))
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        key_padding = (~pad_mask) if pad_mask is not None else None
        rollout = torch.eye(T, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        for layer in self.encoder.layers:
            attn = layer.self_attn(
                h,
                h,
                h,
                attn_mask=causal,
                key_padding_mask=key_padding,
                need_weights=True,
                average_attn_weights=True,
            )[1]
            attn = attn + torch.eye(T, device=x.device).unsqueeze(0)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            rollout = attn @ rollout
            h = layer(h, src_mask=causal, src_key_padding_mask=key_padding)
        return rollout
