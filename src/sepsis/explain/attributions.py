"""Feature attributions for both model families.

* ``tree_shap_summary`` -- exact TreeSHAP (Lundberg et al., 2020) for the
  LightGBM baseline; returns global mean-|SHAP| rankings and the raw matrix for
  beeswarm plots.
* ``integrated_gradients`` -- Sundararajan et al. (2017) path-integrated
  gradients for the causal sequence models, attributing each hour's risk to
  every (past timestep, feature) input. A zero vector is the natural baseline
  because inputs are z-scored. Includes a completeness check
  (sum of attributions ~= f(x) - f(baseline)).
* ``group_attributions`` -- fold the 107 engineered columns back onto the ~34
  physiological channels (value + mask + delta + static share a name) so the
  story is clinically legible.
"""

from __future__ import annotations

import numpy as np
import torch

from sepsis.constants import DYNAMIC_COLS, STATIC_COLS


# --------------------------------------------------------------------------- #
# TreeSHAP for the GBM baseline
# --------------------------------------------------------------------------- #
def tree_shap_summary(gbm_model, table, max_samples: int = 4000, seed: int = 0) -> dict:
    import shap

    X = table.X
    if len(X) > max_samples:
        idx = np.random.default_rng(seed).choice(len(X), max_samples, replace=False)
        X = X[idx]
    explainer = shap.TreeExplainer(gbm_model.booster)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):  # older shap returns [neg, pos]
        sv = sv[1]
    sv = np.asarray(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    names = table.feature_names
    return {
        "feature_names": names,
        "mean_abs_shap": mean_abs.tolist(),
        "ranking": [(names[i], float(mean_abs[i])) for i in order],
        "shap_values": sv,
        "X": X,
        "expected_value": float(np.ravel(explainer.expected_value)[-1]),
    }


# --------------------------------------------------------------------------- #
# Integrated Gradients for sequence models
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _forward_last(model, x, pad_mask):
    logits = model(x, pad_mask)
    return logits


def integrated_gradients(
    model,
    x: np.ndarray,
    target_t: int | None = None,
    steps: int = 64,
    baseline: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """Attribute the risk at ``target_t`` (default: last real hour) of a single
    stay ``x`` of shape ``(T, F)`` to every input entry."""
    model.eval().to(device)
    T, F = x.shape
    tgt = T - 1 if target_t is None else int(target_t)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    base = (
        torch.zeros_like(x_t)
        if baseline is None
        else torch.tensor(baseline, dtype=torch.float32, device=device)
    )
    pad_mask = torch.ones(1, T, dtype=torch.bool, device=device)

    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    grad_sum = torch.zeros_like(x_t)
    for a in alphas:
        xi = (base + a * (x_t - base)).unsqueeze(0).clone().requires_grad_(True)
        logit = model(xi, pad_mask)[0, tgt]
        (grad,) = torch.autograd.grad(logit, xi)
        grad_sum += grad[0]
    avg_grad = grad_sum / steps
    attributions = (x_t - base) * avg_grad

    with torch.no_grad():
        f_x = float(model(x_t.unsqueeze(0), pad_mask)[0, tgt])
        f_base = float(model(base.unsqueeze(0), pad_mask)[0, tgt])
    attr = attributions.cpu().numpy()
    return {
        "attributions": attr,  # (T, F)
        "target_t": tgt,
        "completeness_gap": float(f_x - f_base - attr.sum()),
        "f_x": f_x,
        "f_baseline": f_base,
        "per_feature": attr.sum(axis=0),  # (F,)
        "per_timestep": attr.sum(axis=1),  # (T,)
    }


def group_attributions(
    per_feature: np.ndarray, feature_names: list[str]
) -> list[tuple[str, float]]:
    """Sum signed attributions across the value/mask/delta/static columns that
    share a channel name, then rank by absolute total."""
    base_names = list(DYNAMIC_COLS) + list(STATIC_COLS)
    totals: dict[str, float] = dict.fromkeys(base_names, 0.0)
    for name, val in zip(feature_names, np.asarray(per_feature).ravel()):
        base = name.split("__")[0]
        totals.setdefault(base, 0.0)
        totals[base] += float(val)
    return sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)
