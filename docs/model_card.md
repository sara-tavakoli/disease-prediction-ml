# Model Card — Sepsis Early-Warning (Transformer)

Following Mitchell et al. (2019). Fill the bracketed numbers from
`artifacts/<run>/results.json` after training on the full PhysioNet corpus.

## Model details
* **Developed by:** sara-tavakoli (open-source project).
* **Architecture:** 3-layer causal Transformer encoder, `d_model=128`, 8 heads,
  GELU MLP, sinusoidal positions, subsequent-position + padding masks. ~[N]
  parameters.
* **Input:** up to 336 hours × 107 engineered features (value / observed-mask /
  time-since per physiological channel + standardised static covariates).
* **Output:** per-hour probability of sepsis onset within 6 h, post-hoc
  calibrated (isotonic), plus a split-conformal prediction set.
* **Version / license:** 0.1.0 / MIT. Code:
  <https://github.com/sara-tavakoli/disease-prediction-ml>.

## Intended use
* **Intended:** research and education on ICU early-warning modelling,
  methodology benchmarking (calibration, conformal, fairness, robustness).
* **Out of scope:** clinical decision-making. Not a medical device, not
  validated prospectively, no regulatory clearance.

## Training data
PhysioNet/CinC 2019 `training_setA` (Beth Israel Deaconess). Validation is a
stratified slice of set A. See `docs/methodology.md §2`.

## Evaluation data
`training_setB` (Emory) — a **different institution**, i.e. every reported number
is an external-validation estimate. Metrics on flattened non-padded hours;
95 % CIs from a patient-clustered bootstrap.

## Quantitative analysis (fill in)
| metric | value (95 % CI) |
| --- | --- |
| AUROC | [ ] |
| AUPRC | [ ] |
| Normalised utility score | [ ] |
| Sensitivity @ 85 % specificity | [ ] |
| ECE (isotonic) | [ ] |
| Conformal coverage (target 0.90) | [ ] |

### Subgroup performance
Report AUROC / TPR / alarm-rate / ECE and the max–min gaps for **sex**,
**age band**, **ICU unit** from `results.json["fairness"]`. Investigate any
`tpr_gap` or `calibration_gap` > 0.1.

### Robustness
Report the AUROC/AUPRC curves under Gaussian sensor noise and extra missingness
from `results.json["robustness"]`.

## Ethical considerations & limitations
* Two US academic hospitals — geography, care patterns and coding are not
  representative; do not assume transfer.
* Labels derive from Sepsis-3 operationalised on charted data; label timing
  noise is inherited.
* Alarm fatigue: the operating point is tuned for the challenge utility, not a
  site-specific false-alarm budget. Re-tune before any deployment study.
* The `synthetic` source is a methodological stand-in and must never be used for
  clinical claims.

## Caveats and recommendations
Recalibrate and re-select the threshold on local data; monitor subgroup
calibration over time; keep a human in the loop; treat the conformal `{0,1}`
set as "insufficient evidence".
