"""Turn Transformer attention-rollout into a 'which past hour mattered' profile.

For a stay, ``attention_rollout`` gives a ``(T, T)`` matrix; row ``t`` is how the
prediction at hour ``t`` distributes credit over hours ``<= t``. We summarise the
final real hour's row as a normalised look-back profile and also report the
mean look-back distance -- a small distance means the model is reacting to
*recent* deterioration, a large one means it leans on the admission context.
"""

from __future__ import annotations

import numpy as np
import torch


def temporal_attention_profile(model, x: np.ndarray, device: str = "cpu") -> dict:
    if not hasattr(model, "attention_rollout"):
        raise TypeError("model has no attention_rollout(); use a TransformerRiskModel")
    model.eval().to(device)
    T = x.shape[0]
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    pad = torch.ones(1, T, dtype=torch.bool, device=device)
    roll = model.attention_rollout(xt, pad)[0].cpu().numpy()  # (T, T)
    last_row = roll[T - 1]
    last_row = last_row / max(last_row.sum(), 1e-9)
    lookback = np.arange(T)[::-1]  # hours before the decision point
    return {
        "rollout": roll,
        "lookback_profile": last_row,
        "mean_lookback_hours": float(np.sum(last_row * lookback)),
        "top_hours": np.argsort(last_row)[::-1][:5].tolist(),
    }
