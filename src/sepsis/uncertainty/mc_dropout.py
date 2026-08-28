"""Monte-Carlo dropout for epistemic uncertainty (Gal & Ghahramani, 2016).

Keeping dropout active at inference and averaging ``T`` stochastic forward
passes approximates the predictive distribution of a Bayesian NN. We return the
mean risk plus the between-sample standard deviation, which behaves as a
model-uncertainty signal: it is largest where the training data were sparse
(e.g. rare lab patterns, very long stays).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def mc_dropout_predict(
    model,
    loader: DataLoader,
    n_samples: int = 30,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    model.to(device)
    if not hasattr(model, "mc_dropout_mode"):
        raise TypeError("model must expose mc_dropout_mode()")

    mean_list, var_list, y_list = [], [], []
    for x, y, pad_mask, _ in loader:
        x = x.to(device)
        pm = pad_mask.to(device)
        samples = torch.empty(n_samples, *y.shape, device=device)
        for s in range(n_samples):
            model.mc_dropout_mode()
            samples[s] = torch.sigmoid(model(x, pm))
        m = samples.mean(0)
        v = samples.var(0, unbiased=False)
        flat = pad_mask.bool()
        mean_list.append(m[flat].cpu().numpy())
        var_list.append(v[flat].cpu().numpy())
        y_list.append(y[flat].cpu().numpy())

    mean = np.concatenate(mean_list)
    var = np.concatenate(var_list)
    return {
        "mean": mean,
        "std": np.sqrt(var),
        "y_true": np.concatenate(y_list).astype(np.int8),
        # predictive entropy of the Bernoulli mean (total uncertainty)
        "entropy": -(
            mean * np.log(np.clip(mean, 1e-9, 1))
            + (1 - mean) * np.log(np.clip(1 - mean, 1e-9, 1))
        ),
    }
