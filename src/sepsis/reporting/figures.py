"""Matplotlib figures for the results report. Every function writes a PNG and
returns its path; all are safe to call in a headless CI job."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def plot_reliability(curves: dict[str, dict], path: str | Path) -> str:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, rc in curves.items():
        ax.plot(rc["confidence"], rc["accuracy"], "o-", ms=4, label=name)
    ax.set_xlabel("mean predicted risk")
    ax.set_ylabel("observed sepsis rate")
    ax.set_title("Reliability diagram")
    ax.legend()
    return _save(fig, path)


def plot_decision_curve(dc, path: str | Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dc.thresholds, dc.net_benefit_model, label="model", lw=2)
    ax.plot(dc.thresholds, dc.net_benefit_all, label="treat all", ls="--")
    ax.plot(dc.thresholds, dc.net_benefit_none, label="treat none", color="grey")
    ax.set_ylim(min(-0.02, float(np.min(dc.net_benefit_model))), None)
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision-curve analysis")
    ax.legend()
    return _save(fig, path)


def plot_robustness(noise_rows, miss_rows, path: str | Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    s = [r["sigma"] for r in noise_rows]
    axes[0].plot(s, [r["auroc"] for r in noise_rows], "o-", label="AUROC")
    axes[0].plot(s, [r["auprc"] for r in noise_rows], "s-", label="AUPRC")
    axes[0].set_xlabel("Gaussian noise sigma (z-units)")
    axes[0].set_title("Sensor-noise robustness")
    axes[0].legend()
    f = [r["extra_missing_frac"] for r in miss_rows]
    axes[1].plot(f, [r["auroc"] for r in miss_rows], "o-", label="AUROC")
    axes[1].plot(f, [r["auprc"] for r in miss_rows], "s-", label="AUPRC")
    axes[1].set_xlabel("additional fraction of measurements dropped")
    axes[1].set_title("Missing-data robustness")
    axes[1].legend()
    return _save(fig, path)


def plot_utility_threshold(grid, utilities, chosen, path: str | Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, utilities, "-o", ms=3)
    ax.axvline(chosen, color="crimson", ls="--", label=f"chosen = {chosen:.3f}")
    ax.set_xlabel("alarm threshold")
    ax.set_ylabel("normalised utility")
    ax.set_title("PhysioNet utility vs. operating point")
    ax.legend()
    return _save(fig, path)


def plot_subgroup_gaps(report: dict, path: str | Path) -> str:
    axes_names = list(report.keys())
    fig, axs = plt.subplots(1, len(axes_names), figsize=(5 * len(axes_names), 4),
                            squeeze=False)
    for k, axis in enumerate(axes_names):
        groups = report[axis]["groups"]
        names = list(groups.keys())
        auroc = [groups[g]["auroc"] for g in names]
        axs[0][k].bar(names, auroc)
        axs[0][k].set_ylim(0.5, 1.0)
        axs[0][k].set_title(f"{axis}  (AUROC gap = {report[axis]['auroc_gap']:.3f})")
        axs[0][k].tick_params(axis="x", rotation=30)
    return _save(fig, path)


def plot_shap_bar(ranking: list[tuple[str, float]], path: str | Path, top: int = 20) -> str:
    items = ranking[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(items) + 1))
    ax.barh([n for n, _ in items], [v for _, v in items])
    ax.set_xlabel("mean |contribution|")
    ax.set_title("Global feature importance")
    return _save(fig, path)


def plot_risk_trajectory(hours, risk, onset_hour, alarm_threshold, path: str | Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(hours, risk, "-", lw=2)
    ax.axhline(alarm_threshold, color="crimson", ls="--", label="alarm threshold")
    if onset_hour is not None:
        ax.axvline(onset_hour, color="k", ls=":", label="label onset (t-6h)")
    ax.set_xlabel("ICU hour")
    ax.set_ylabel("calibrated sepsis risk")
    ax.set_ylim(0, 1)
    ax.set_title("Example risk trajectory")
    ax.legend()
    return _save(fig, path)


def plot_training_history(history: list[dict], path: str | Path) -> str:
    if not history:
        fig, ax = plt.subplots()
        return _save(fig, path)
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, [h["train_loss"] for h in history], "b-", label="train loss")
    ax1.plot(ep, [h["val_loss"] for h in history], "b--", label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="b")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_auprc"] for h in history], "g-", label="val AUPRC")
    ax2.plot(ep, [h["val_auroc"] for h in history], "g:", label="val AUROC")
    ax2.set_ylabel("val metric", color="g")
    ax1.set_title("Training history")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    return _save(fig, path)
