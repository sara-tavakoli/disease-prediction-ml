# Early Sepsis Prediction from Clinical Time Series: A Reproducible Pipeline with Calibration, Conformal Uncertainty, Fairness and Robustness Auditing

*Technical report — companion to the `disease-prediction-ml` codebase.*

## Abstract

We present an end-to-end, reproducible pipeline for hour-by-hour sepsis
early-warning on the PhysioNet/Computing in Cardiology Challenge 2019 data. The
pipeline pairs five model families — a gradient-boosting baseline and four
strictly causal deep sequence encoders (LSTM, GRU, temporal convolutional
network, Transformer) — with (i) a leak-free, patient-level data protocol and a
GRU-D-style missing-data representation, (ii) the official time-dependent
utility score with utility-curve threshold selection, (iii) post-hoc calibration
*and* split/Mondrian conformal prediction sets with finite-sample coverage,
(iv) MC-dropout epistemic uncertainty, (v) a subgroup fairness audit and a
perturbation/shift robustness audit, and (vi) model-agnostic and
model-specific explainability. All estimates carry patient-clustered bootstrap
confidence intervals. A serialised run bundle drives a FastAPI service that
replays the exact preprocessing. The utility-score implementation is
unit-tested against the published reference value.

## 1. Introduction

Sepsis is a leading cause of in-hospital mortality, and each hour of delayed
treatment measurably increases risk. The 2019 Challenge framed the problem as an
*online* one: at every ICU hour, emit a probability that the patient will meet
Sepsis-3 criteria within six hours, scored by a bespoke utility function that
rewards early-but-not-too-early alarms and penalises both misses and false
alarms.

Most published entries optimise the utility score directly and stop there. A
model that is going to be trusted at the bedside needs more: probabilities that
mean what they say, an explicit notion of "insufficient evidence", evidence that
performance does not collapse for a subgroup or under sensor noise, and
explanations a clinician can interrogate. This report describes a pipeline that
treats those requirements as first-class.

## 2. Data and protocol

**Data.** PhysioNet/CinC 2019 `training_setA` (Beth Israel Deaconess, ~20 000
stays) and `training_setB` (Emory, ~20 000). Each stay is an hourly table of 8
vitals, 26 labs and 6 context columns; labs are >90 % missing. The label is
pre-shifted +6 h for septic patients.

**Splitting.** Patient-level and stratified on the stay-level outcome. With
`group_by_hospital`, development uses set A and the **entire set B is the test
set**, an institution-shift evaluation.

**Representation.** Per dynamic channel and hour: carried-forward z-scored value
(train-only statistics; train-mean back-fill), a binary *measured-now* mask, and
`log1p` time-since-last-measurement (Che et al., 2018). Static covariates are
median-imputed, z-scored, broadcast. Width 107; column order frozen and reused
by training, evaluation and serving.

## 3. Models

| family | summary |
| --- | --- |
| LightGBM | sliding-window features (`last/mean/min/max/std/slope/obs_frac/delta` + static); `scale_pos_weight`; early stopping on val AUPRC |
| LSTM / GRU | unidirectional, packed sequences, per-hour head |
| TCN | dilated causal convolutions, weight norm, residual (Bai et al., 2018) |
| Transformer | causal + padding masks, sinusoidal positions, GELU MLP; attention rollout |

Training: masked focal loss (`γ=2`) with `pos_weight = #neg/#pos`; AdamW; cosine
decay; gradient clipping; early stopping on validation AUPRC with weight
restore; deterministic seeding; MLflow logging.

## 4. Evaluation

* **Discrimination:** AUROC, AUPRC, Brier, sensitivity@85 %-specificity.
* **Utility:** faithful re-implementation of `evaluate_sepsis_score.py`,
  normalised so inaction = 0 and oracle = 1; **threshold selected on the
  validation utility curve**.
* **Calibration:** isotonic/Platt/temperature on validation; ECE/MCE and
  reliability diagrams before vs after.
* **Conformal:** split conformal, nonconformity `1 − p_true`, threshold at the
  `⌈(n+1)(1−α)⌉/n` quantile; Mondrian variant for class-conditional coverage.
  Report empirical coverage, average set size, and the fraction of
  uninformative `{0,1}` sets.
* **Epistemic uncertainty:** MC-dropout predictive mean/SD/entropy.
* **Clinical value:** decision-curve analysis vs treat-all / treat-none.
* **Confidence intervals:** patient-clustered bootstrap (resample stays).

## 5. Fairness audit

Test hours stratified by sex, age band and ICU unit. Per group: AUROC, AUPRC,
TPR, FPR, alarm rate, ECE at the shared threshold. Headline: `tpr_gap`
(equal-opportunity), `alarm_rate_gap` (demographic parity of the alert),
`calibration_gap`.

## 6. Robustness audit

Gaussian sensor noise on the standardised value channels
(`σ ∈ {0…1.5}`) and additional random missingness (`{0…50}%`) with recomputed
carry-forward/recency; report AUROC/AUPRC degradation curves. The
cross-institution test (set A → set B) is the distribution-shift probe.

## 7. Explainability

TreeSHAP (LightGBM), Integrated Gradients with a completeness check (sequence
models, folded onto physiological channels), PDP + ALE, a depth-4 global
surrogate tree with fidelity `R²` and alarm-agreement, and Transformer attention
rollout summarised as a look-back profile.

## 8. Results

Synthetic sanity benchmark (5 000 stays, 8 % prevalence, 25 epochs,
seed 20190804), produced by `scripts/reproduce_all.sh synthetic`:

<!-- RESULTS:START -->
| model | AUROC (95% CI) | AUPRC (95% CI) | utility | sens@spec85 | ECE (raw → cal) | conf. cov. (α=0.1) | params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 0.998 [0.996, 0.999] | 0.989 [0.984, 0.994] | 0.873 | 0.993 | 0.001 → 0.001 | 0.997 | — |
| lstm | 0.998 [0.997, 0.999] | 0.989 [0.983, 0.993] | 0.877 | 0.995 | 0.127 → 0.001 | 0.997 | 255105 |
| gru | 0.998 [0.997, 0.999] | 0.992 [0.986, 0.995] | 0.886 | 0.997 | 0.111 → 0.001 | 0.997 | 191361 |
| tcn | 0.998 [0.996, 0.999] | 0.992 [0.987, 0.995] | 0.889 | 0.994 | 0.086 → 0.001 | 0.998 | 501505 |
| transformer | 0.998 [0.996, 0.999] | 0.991 [0.986, 0.995] | 0.894 | 0.997 | 0.081 → 0.001 | 0.997 | 609153 |
<!-- RESULTS:END -->

Observations to expect and report on the real corpus:

* AUPRC ≫ prevalence for all models; Transformer/TCN competitive with or ahead
  of the RNNs; LightGBM a strong, fast baseline.
* Isotonic calibration reduces ECE by a large factor; temperature scaling
  leaves AUROC unchanged (rank-preserving).
* Conformal empirical coverage ≈ `1 − α`; the `{0,1}` fraction concentrates near
  onset and in short stays.
* Fairness gaps are small but non-zero — inspect any `tpr_gap` or
  `calibration_gap` > 0.1.
* Graceful, monotone degradation under noise and missingness; a measurable but
  bounded drop from set A → set B.

## 9. Limitations

Two US academic hospitals; charted-data label timing noise; operating point
tuned to the challenge utility rather than a site false-alarm budget; the
synthetic source is a methodological stand-in only. The pipeline is for research
and education — it is not a medical device and has no prospective validation.

## References

`docs/references.bib`.
