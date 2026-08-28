# %% [markdown]
# # 02 · Model comparison
#
# Train each architecture on the synthetic cohort and line up the headline
# metrics. For real results, run `make train-all` after `sepsis download --full`
# and point `RUN_DIRS` at `artifacts/<model>/`.

# %%
import json
from pathlib import Path

import pandas as pd

from sepsis.config import ExperimentConfig
from sepsis.training.experiment import run_experiment

MODELS = ["lightgbm", "lstm", "gru", "tcn", "transformer"]
RUN_DIRS = {m: Path(f"artifacts/{m}") for m in MODELS}

# %%
for m in MODELS:
    if (RUN_DIRS[m] / "results.json").exists():
        print(f"{m}: cached")
        continue
    cfg = ExperimentConfig.load(
        "configs/base.yaml", f"configs/model_{m}.yaml",
        overrides=["data.source=synthetic", "data.synthetic_n_patients=3000",
                   "data.group_by_hospital=false", "train.epochs=12"],
    )
    run_experiment(cfg)

# %% [markdown]
# ## Leaderboard

# %%
rows = []
for m in MODELS:
    r = json.loads((RUN_DIRS[m] / "results.json").read_text())
    d = r["discrimination"]
    rows.append({
        "model": m,
        "AUROC": round(d["auroc"]["point"], 4),
        "AUROC 95% CI": f'[{d["auroc"]["lo"]:.3f}, {d["auroc"]["hi"]:.3f}]',
        "AUPRC": round(d["auprc"]["point"], 4),
        "utility": round(r["utility"]["test_utility"]["normalized"], 4),
        "ECE(cal)": round(r["calibration"]["ece_calibrated"]["ece"], 4),
        "conf. coverage": round(r["conformal"]["empirical_coverage"], 3),
        "sens@spec85": round(d["summary_at_operating_point"]["sensitivity_at_spec85"], 3),
    })
leaderboard = pd.DataFrame(rows).set_index("model").sort_values("AUPRC", ascending=False)
leaderboard

# %% [markdown]
# ## Reliability curves side by side

# %%
from IPython.display import Image, display  # noqa: E402

for m in MODELS:
    p = RUN_DIRS[m] / "figures" / "reliability.png"
    if p.exists():
        print(m)
        display(Image(str(p)))
