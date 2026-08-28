# Methodology

This document specifies the problem, the data pipeline, the models, and every
evaluation and auditing procedure the codebase implements. It is written so that
the results are reproducible from the config alone.

---

## 1. Problem definition

**Task.** At every ICU hour `t`, emit a probability that a patient will meet the
Sepsis-3 criteria within the next 6 hours. This is the online early-warning
formulation of the *PhysioNet/Computing in Cardiology Challenge 2019*
(Reyna et al., 2020).

**Label.** The challenge provides `SepsisLabel` already shifted **+6 h**: for a
septic patient with clinical onset `t_sepsis`, `SepsisLabel = 1` for every hour
`t ≥ t_sepsis − 6`. We use the label as given; `data.label_shift_hours: 6` is
recorded in the config for provenance and is asserted in the tests.

**Prediction is strictly causal.** Every model produces the score for hour `t`
from inputs `≤ t` only:

* RNNs are unidirectional (`bidirectional=false`, validated in
  `tests/test_models.py::test_strict_causality`);
* the TCN uses left-only ("causal") convolution padding with a `Chomp` crop;
* the Transformer applies a subsequent-position mask **and** the batch padding
  mask;
* sliding-window tabular features look backwards only.

---

## 2. Data

### 2.1 Sources

| source | description | how to get it |
| --- | --- | --- |
| `physionet` | 40 336 ICU stays, `training_setA` (Beth Israel, ~20 k) + `training_setB` (Emory, ~20 k), 40 channels/hour | `sepsis download --full` |
| `physionet` (sample) | 240 stays committed under `data/sample/` | shipped in the repo; used for smoke tests |
| `synthetic` | physiologically-motivated generator, configurable size/prevalence | `sepsis synth` or `data.source=synthetic` |

The **synthetic** generator (`sepsis.data.synthetic`) draws static covariates
from ICU marginals, simulates each vital as a mean-reverting
Ornstein–Uhlenbeck process around an age-dependent baseline, samples labs
sparsely (per-channel hourly observation probability matching the real
~90 %+ missing rate), and, for septic patients, ramps a deterioration signal
(tachycardia, tachypnoea, rising lactate/WBC/creatinine/temperature, falling
MAP/platelets) into the hours before a sampled onset, then applies the same
+6 h label shift. It exists so the whole pipeline — including SHAP, conformal
calibration and the subgroup fairness audit — runs offline and in CI.

### 2.2 Splitting (`sepsis.data.splits`)

* **Patient-level.** No ICU stay contributes hours to more than one split
  (`tests/test_data.py::test_patient_level_split_has_no_shared_ids`). Splitting
  at the window level would leak highly autocorrelated hours and inflate every
  metric.
* **Stratified** on the stay-level outcome (ever-septic) so the ~8 % positive
  stays appear in train/val/test in proportion.
* **External validation.** With `group_by_hospital: true` and both hospital
  sources present, training uses `training_setA`, validation is a stratified
  slice of `setA`, and **test is the entire `training_setB`** — an
  institution-shift evaluation.

### 2.3 Preprocessing (`sepsis.data.preprocess`)

Hourly grid (already native). For each **dynamic** channel (8 vitals + 26 labs +
`ICULOS`) and hour we emit three numbers, following the GRU-D missing-data
representation (Che et al., 2018):

| feature | definition |
| --- | --- |
| `<chan>__value` | last observed measurement carried forward; back-filled with the **train** mean before the first observation; z-scored with **train-only** mean/SD; clipped to physiologic ranges first |
| `<chan>__mask` | 1 iff the channel was actually measured this hour |
| `<chan>__delta` | hours since last measurement, `log1p`-scaled by `log1p(max_seq_len)` |

**Static** covariates `[Age, Gender, Unit1, Unit2, HospAdmTime]` are
median-imputed, z-scored and broadcast over time. Final width
`3·34 + 5 = 107`; the column order is frozen in `PreprocessArtifacts` and reused
by training, evaluation **and** the serving API. Stays longer than
`max_seq_len` (default 336 h) are left-truncated to the most recent window.
`tests/test_preprocess.py` asserts the mask matches real observations, that
normaliser statistics change when test data is added (no leakage), and that the
first `t` encoded hours are identical whether or not later hours exist (causal
carry-forward).

---

## 3. Models

| name | family | key points |
| --- | --- | --- |
| `lightgbm` | gradient-boosted trees | sliding-window tabular features (`last/mean/min/max/std/slope/obs_frac/delta` per channel + static); `scale_pos_weight` from train imbalance; early stopping on val AUPRC |
| `lstm`, `gru` | unidirectional RNN | packed padded sequences; per-hour linear head |
| `tcn` | temporal CNN | exponentially dilated causal blocks with weight norm + residual (Bai et al., 2018) |
| `transformer` | encoder | learned input projection + sinusoidal positions, causal + padding masks, GELU MLP; exposes attention-rollout (Abnar & Zuidema, 2020) |

**Loss.** Positive hours are ~2 % of the corpus. Default is a **masked focal
loss** (Lin et al., 2017, `γ=2`) with `pos_weight` set to `#neg/#pos` on the
training split; `weighted_bce` and `bce` are alternatives. Padded timesteps are
excluded from the loss (`tests/test_models.py::test_focal_loss_ignores_padding`).

**Optimisation.** AdamW, cosine LR decay to zero, gradient-norm clipping,
early stopping on **validation AUPRC** with best-weight restore. Every run is
seeded (`sepsis.utils.seeding.seed_everything`, deterministic torch) and logged
to MLflow (falls back to a local `./mlruns` store).

---

## 4. Evaluation

All metrics are computed on flattened, non-padded test hours unless noted.

### 4.1 Discrimination
AUROC, AUPRC, Brier; sensitivity at 85 % specificity and specificity at 85 %
sensitivity; full confusion counts at the chosen operating threshold.

### 4.2 PhysioNet utility score (`sepsis.evaluation.utility_score`)
A faithful re-implementation of the official `evaluate_sepsis_score.py`. A
positive prediction is rewarded on a piecewise-linear ramp that starts at
`t_sepsis − 12`, peaks (`+1`) at `t_sepsis − 6`, and decays to 0 by
`t_sepsis + 3`; missed septic hours are penalised down to `−2`; false alarms on
non-septic stays cost `−0.05/h`. The cohort score is normalised so the
always-negative *inaction* policy scores 0 and the oracle scores 1. Verified
against the published worked example (`3.3888…`) in
`tests/test_utility_score.py`. The alarm threshold is selected by **sweeping the
utility curve on the validation set**, not Youden's J or 0.5.

### 4.3 Uncertainty
* **Confidence intervals** — patient-clustered bootstrap (resample *stays*, not
  hours) for AUROC/AUPRC.
* **Calibration** — isotonic / Platt / temperature fitted on validation;
  reliability diagrams and ECE/MCE before vs after; temperature scaling is
  verified rank-preserving.
* **Conformal risk sets** (`sepsis.uncertainty.conformal`) — split conformal
  with nonconformity `1 − p_true`; threshold at the
  `⌈(n+1)(1−α)⌉/n` calibration quantile; Mondrian (class-conditional) variant
  for coverage *within* the rare positive class. Test-time sets are `{0}`,
  `{1}` or the uninformative `{0,1}`; the fraction of `{0,1}` sets is a direct
  uncertainty readout. Coverage is checked against `1 − α` in the tests.
* **MC-dropout** (`sepsis.uncertainty.mc_dropout`) — `T` stochastic passes with
  dropout active give the predictive mean, epistemic SD and entropy.

### 4.4 Clinical value
Decision-curve analysis (Vickers & Elkin, 2006): net benefit of the model vs
"treat all" / "treat none" across threshold probabilities.

---

## 5. Fairness audit (`sepsis.audit.fairness`)

Test hours are stratified by **sex**, **age band** (`<40 / 40–64 / 65–79 / 80+`)
and **ICU unit** (MICU/SICU/unknown). Per group we report AUROC, AUPRC, TPR,
FPR, alarm rate and ECE **at the shared cohort threshold**, and the headline
max–min gaps: `tpr_gap` (equal-opportunity violation, Hardt et al., 2016),
`alarm_rate_gap` (demographic parity of the alert) and `calibration_gap`.

---

## 6. Robustness audit (`sepsis.audit.robustness`)

* **Sensor noise** — Gaussian perturbation of the standardised value channels at
  `σ ∈ {0, 0.1, 0.25, 0.5, 1.0, 1.5}` (mask/delta/static untouched); report the
  AUROC/AUPRC degradation curve.
* **Extra missingness** — randomly blank an additional `{0, 10, 25, 50}%` of the
  observed measurements, then re-derive carry-forward values and recency, and
  re-score.

A trustworthy model degrades gracefully and monotonically.

---

## 7. Explainability (`sepsis.explain`)

| method | applies to | output |
| --- | --- | --- |
| **TreeSHAP** (Lundberg et al., 2020) | LightGBM | exact global mean-\|SHAP\| ranking + beeswarm matrix |
| **Integrated Gradients** (Sundararajan et al., 2017) | sequence models | per-(hour, feature) attribution of the final-hour risk, with a completeness check; folded onto physiological channels |
| **PDP + ALE** (Friedman 2001; Apley & Zhu 2020) | LightGBM | marginal effect curves for top features (ALE is correlation-robust) |
| **Global surrogate tree** | any | depth-4 tree fitted to model logits; reports fidelity `R²` and alarm-agreement |
| **Attention rollout** (Abnar & Zuidema, 2020) | Transformer | "which past hour mattered" profile + mean look-back distance |

---

## 8. Serving (`sepsis.serve`)

`run_experiment` writes a self-contained bundle: `config.json`,
`preprocess.json`, `model.pt`/`model.txt`, `calibrator.joblib`,
`conformal.joblib`, `operating_point.json`. `SepsisPredictor` replays the exact
preprocessing and returns, per hour, the calibrated risk, the conformal set, and
(for sequence models) the grouped Integrated-Gradients top drivers of the most
recent hour. The FastAPI app exposes `/health`, `/model`, `/predict`.

---

## 9. Reproducibility checklist

* One `ExperimentConfig` (YAML + `a.b=c` overrides) fully determines a run and is
  saved next to its outputs.
* Global seeding + deterministic torch.
* Train-only preprocessing statistics; patient-level, stratified splits;
  optional cross-institution test.
* Official utility score with a regression test against the published number.
* Patient-clustered CIs.
* `pytest -q` (48 tests) + `ruff` run in CI on Python 3.10–3.12, plus a
  2-epoch train-and-serve smoke job.

## References
See `docs/references.bib`.
