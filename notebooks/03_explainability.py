# %% [markdown]
# # 03 · Explainability
#
# TreeSHAP for the gradient-boosting baseline, Integrated Gradients for a
# sequence model, a global surrogate tree for both, and Transformer attention
# rollout.

# %%
import json
from pathlib import Path

import numpy as np

from sepsis.config import ExperimentConfig
from sepsis.data.preprocess import Preprocessor
from sepsis.data.splits import make_splits
from sepsis.data.synthetic import generate_cohort
from sepsis.explain.attributions import group_attributions, integrated_gradients
from sepsis.explain.attention import temporal_attention_profile

# %% [markdown]
# ## Rebuild a small dataset and a trained Transformer

# %%
cohort = generate_cohort(1500, 0.1, seed=1)
sp = make_splits(cohort, 0.15, 0.15, seed=1, group_by_hospital=False)
pre = Preprocessor(max_seq_len=200).fit(sp.train)
train_td, test_td = pre.transform(sp.train), pre.transform(sp.test)

from sepsis.config import ExperimentConfig  # noqa: E402
from sepsis.training.trainer import SequenceTrainer  # noqa: E402

cfg = ExperimentConfig.load("configs/base.yaml", "configs/model_transformer.yaml",
                            overrides=["train.epochs=8"])
fit = SequenceTrainer(cfg).fit(train_td, pre.transform(sp.val))
model = fit.model

# %% [markdown]
# ## Integrated Gradients on a septic stay
# Attribution of the final-hour risk to every (past hour, feature); folded onto
# physiological channels. `completeness_gap` should be near zero.

# %%
i = next(k for k in range(len(test_td)) if test_td.y[k].max() > 0)
n = int(test_td.lengths[i])
ig = integrated_gradients(model, test_td.X[i, :n], target_t=n - 1, steps=64)
print("completeness gap:", round(ig["completeness_gap"], 4))
grouped = group_attributions(ig["per_feature"], test_td.feature_names)
for name, val in grouped[:12]:
    print(f"  {name:16s} {val:+.4f}")

# %%
import matplotlib.pyplot as plt  # noqa: E402

attr = ig["attributions"]
plt.figure(figsize=(10, 4))
plt.plot(ig["per_timestep"])
plt.axvline(test_td.y[i, :n].argmax(), color="k", ls=":", label="label onset")
plt.title("Integrated-Gradients attribution by hour (final-hour risk)")
plt.xlabel("ICU hour")
plt.legend()
plt.show()

# %% [markdown]
# ## Attention rollout — how far back does the model look?

# %%
prof = temporal_attention_profile(model, test_td.X[i, :n])
print("mean look-back:", round(prof["mean_lookback_hours"], 1), "hours")
plt.figure(figsize=(9, 3))
plt.bar(np.arange(n)[::-1], prof["lookback_profile"])
plt.xlabel("hours before the decision point")
plt.ylabel("attention weight")
plt.title("Look-back profile at the final hour")
plt.show()
