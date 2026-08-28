# ---
# jupyter:
#   jupytext:
#     text_representation: {extension: .py, format_name: percent}
# ---

# %% [markdown]
# # 01 · Exploratory data analysis
#
# Cohort composition, missingness structure, and how vitals/labs move in the
# hours around sepsis onset. Runs on whatever `data.source` resolves to — the
# committed `data/sample/` real stays by default, or the synthetic generator.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sepsis.constants import LABS, VITALS
from sepsis.data.psv import load_dataset
from sepsis.data.synthetic import generate_cohort

try:
    records = load_dataset("data/sample")
    print(f"loaded {len(records)} real PhysioNet sample stays")
except Exception:
    records = generate_cohort(2000, 0.08, seed=0)
    print(f"using {len(records)} synthetic stays")

# %% [markdown]
# ## Cohort-level summary

# %%
n = len(records)
sep = sum(r.is_septic for r in records)
hours = sum(r.n_hours for r in records)
pos_hours = sum(int(r.label.sum()) for r in records)
print(pd.Series({
    "stays": n,
    "septic stays": sep,
    "septic-stay rate": round(sep / n, 4),
    "total ICU hours": hours,
    "positive hours": pos_hours,
    "positive-hour rate": round(pos_hours / hours, 5),
    "median stay length (h)": int(np.median([r.n_hours for r in records])),
}))

# %% [markdown]
# ## Missingness by channel
# Labs are the classic >90%-missing problem; vitals are mostly present.

# %%
frames = pd.concat([r.frame for r in records], ignore_index=True)
miss = frames[VITALS + LABS].isna().mean().sort_values()
ax = miss.plot.barh(figsize=(7, 10))
ax.set_xlabel("fraction missing")
ax.set_title("Per-channel missing rate")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Onset-aligned trajectories
# Average each vital in a window around the (label) onset hour for septic stays,
# vs a matched random index for non-septic stays.

# %%
W = 12
def aligned(channel):
    sep_stack, ctl_stack = [], []
    for r in records:
        x = r.frame[channel].to_numpy(dtype=float)
        if r.is_septic:
            t0 = r.onset_hour
            lo, hi = t0 - W, t0 + W
            if lo >= 0 and hi < len(x):
                sep_stack.append(x[lo:hi])
        else:
            if len(x) > 2 * W + 1:
                c = np.random.randint(W, len(x) - W)
                ctl_stack.append(x[c - W:c + W])
    return np.nanmean(sep_stack, axis=0), np.nanmean(ctl_stack, axis=0)

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, ch in zip(axes.ravel(), ["HR", "Resp", "MAP", "Temp", "Lactate", "WBC"]):
    s, c = aligned(ch)
    t = np.arange(-W, W)
    ax.plot(t, s, label="septic (onset-aligned)")
    ax.plot(t, c, label="non-septic")
    ax.axvline(0, color="k", ls=":")
    ax.set_title(ch)
    ax.set_xlabel("hours from onset")
axes[0, 0].legend()
plt.tight_layout()
plt.show()
