# %% [markdown]
# # 04 · Fairness & robustness
#
# Read a finished run's `results.json` and visualise the subgroup gaps, the
# noise/missingness degradation curves, calibration before/after, and the
# decision curve.

# %%
import json
from pathlib import Path

import pandas as pd

RUN = Path("artifacts/transformer")  # any completed run directory
res = json.loads((RUN / "results.json").read_text())

# %% [markdown]
# ## Subgroup performance

# %%
for axis, block in res["fairness"].items():
    print(f"\n=== {axis} ===  "
          f"AUROC gap={block['auroc_gap']:.3f}  TPR gap={block['tpr_gap']:.3f}  "
          f"alarm-rate gap={block['alarm_rate_gap']:.3f}  "
          f"calibration gap={block['calibration_gap']:.3f}")
    display(pd.DataFrame(block["groups"]).T.round(4))

# %% [markdown]
# ## Robustness curves

# %%
rob = res["robustness"]
noise = pd.DataFrame(rob["noise"]).set_index("sigma")
miss = pd.DataFrame(rob["missingness"]).set_index("extra_missing_frac")
display(noise)
display(miss)

ax = noise[["auroc", "auprc"]].plot(marker="o", title="Sensor-noise robustness")
ax.set_xlabel("Gaussian noise sigma (z-units)")
ax2 = miss[["auroc", "auprc"]].plot(marker="s", title="Missing-data robustness")
ax2.set_xlabel("extra fraction of measurements dropped")

# %% [markdown]
# ## Calibration & decision curve (pre-rendered figures)

# %%
from IPython.display import Image, display  # noqa: E402

for fig in ["reliability.png", "decision_curve.png", "utility_threshold.png",
            "subgroup_gaps.png", "robustness.png", "example_trajectory.png"]:
    p = RUN / "figures" / fig
    if p.exists():
        print(fig)
        display(Image(str(p)))

# %% [markdown]
# ## Calibration numbers

# %%
c = res["calibration"]
print("ECE  uncalibrated:", round(c["ece_uncalibrated"]["ece"], 4))
print("ECE  calibrated  :", round(c["ece_calibrated"]["ece"], 4), f"({c['method']})")
print("conformal:", json.dumps(res["conformal"], indent=2))
