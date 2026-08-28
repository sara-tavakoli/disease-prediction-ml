# Early Sepsis Prediction from Clinical Time Series

[![ci](https://github.com/sara-tavakoli/disease-prediction-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/sara-tavakoli/disease-prediction-ml/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![code style: ruff](https://img.shields.io/badge/style-ruff-261230)](https://docs.astral.sh/ruff/)

> **Disease-prediction case study** on the hardest tabular-clinical problem in the
> open literature: predicting sepsis **6 hours before onset**, one score per ICU
> hour, from irregular and mostly-missing vital signs and labs
> ([PhysioNet/CinC Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)).

This is not a single model — it is a **research-grade pipeline**:

* four **causal** deep sequence encoders (LSTM · GRU · Temporal CNN · Transformer)
  plus a gradient-boosting baseline, all trained with an imbalance-aware focal
  loss on a GRU-D-style missing-data representation;
* the **official time-dependent utility score** (re-implemented and unit-tested
  against the published number) with the alarm threshold chosen on its curve;
* **calibrated probabilities** (isotonic / Platt / temperature) *and*
  **split / Mondrian conformal prediction sets** with finite-sample coverage;
* **MC-dropout** epistemic uncertainty;
* a **fairness audit** (subgroup AUROC / TPR / alarm-rate / calibration gaps for
  sex, age band, ICU unit) and a **robustness audit** (Gaussian sensor noise and
  extra-missingness stress curves; cross-institution external test);
* **explainability** — TreeSHAP, Integrated Gradients (with completeness check),
  PDP/ALE, a global surrogate tree, and Transformer attention rollout;
* patient-**clustered bootstrap** confidence intervals and **decision-curve
  analysis**;
* a **FastAPI** service that replays the exact preprocessing and returns a
  calibrated hourly risk trajectory with conformal sets and top risk drivers;
* MLflow tracking, typed config, Docker, pre-commit, and **48 tests** in CI on
  Python 3.10–3.12.

Full method spec: [`docs/methodology.md`](docs/methodology.md) ·
architecture + data flow: [`docs/architecture.md`](docs/architecture.md) ·
[`docs/model_card.md`](docs/model_card.md) ·
[`docs/paper.md`](docs/paper.md).

---

## Quickstart

```bash
make setup                      # venv + editable install + pre-commit
source .venv/bin/activate

# 1) smoke run on the built-in synthetic cohort (no download), ~30 s
sepsis train --config configs/base.yaml configs/smoke.yaml configs/model_transformer.yaml

# 2) real data — a 240-stay sample ships in the repo; get the full ~40k corpus with:
sepsis download --full          # ~2.6 GB into data/raw/
sepsis train --config configs/base.yaml configs/model_transformer.yaml \
             configs/data_physionet.yaml

# 3) serve the trained bundle
sepsis serve --run-dir artifacts/transformer --port 8000
curl localhost:8000/health
```

Every run writes a self-contained bundle to `artifacts/<name>/`:
`results.json` (every metric, CI, subgroup table, robustness curve, ranking),
`figures/*.png`, and the serialised model + calibrator + conformal + operating
point. See [`docs/architecture.md`](docs/architecture.md#serialised-bundle).

---

## Results

Synthetic cohort (5 000 stays, 8 % prevalence, 25 epochs, seed 20190804) — a
reproducible sanity benchmark produced by `scripts/reproduce_all.sh`. Numbers on
the **real** PhysioNet corpus require `sepsis download --full`; fill the model
card from `artifacts/<run>/results.json`.

<!-- RESULTS:START -->
_Run `bash scripts/reproduce_all.sh` to populate this table._
<!-- RESULTS:END -->

Representative figures (`artifacts/<run>/figures/`):

| reliability | decision curve | robustness | utility vs threshold |
| --- | --- | --- | --- |
| calibration before/after | net benefit vs treat-all/none | AUROC under noise & missingness | operating-point selection |

---

## The problem in one picture

```
ICU hour:      1   2   3   4  ...  t
vitals/labs:  [irregular, ~90% missing]
                        │
                 causal encoder  ──►  P(sepsis within 6h | data ≤ t)
                        │
        calibration ► conformal set ► utility-tuned alarm ► top drivers
```

The label is the challenge's **+6 h-shifted** `SepsisLabel`: a correct alarm
fires up to 12 h early, is worth most at −6 h, and stops scoring after +3 h
(`sepsis.evaluation.utility_score`, verified against the reference
`3.3888…` example).

---

## Repository layout

```
src/sepsis/
  data/         PSV IO · synthetic generator · leak-free splits · GRU-D preprocessing
  features/     sliding-window tabular features (GBM)
  models/       masked focal/BCE losses · causal LSTM/GRU/TCN/Transformer · LightGBM
  training/     Trainer (early stopping, MLflow) · run_experiment · `sepsis` CLI
  evaluation/   metrics · official utility score · clustered bootstrap · decision curves
  uncertainty/  calibration + ECE/reliability · split & Mondrian conformal · MC-dropout
  explain/      TreeSHAP · Integrated Gradients · PDP/ALE · surrogate tree · attention
  audit/        subgroup fairness gaps · noise & missingness robustness
  reporting/    headless matplotlib figures
  serve/        Pydantic schemas · SepsisPredictor · FastAPI app
configs/        base.yaml + one file per model + data/smoke variants
notebooks/      01_eda · 02_modeling · 03_explainability · 04_fairness_robustness  (jupytext .py)
tests/          48 tests: utility-score parity, conformal coverage, causality, no-leakage, API
```

---

## Design decisions that matter

| decision | why |
| --- | --- |
| **Patient-level, stratified splits**; optional **train-A / test-B** | window-level splitting leaks autocorrelated hours; cross-hospital test measures real generalisation |
| **Train-only** normaliser + imputation priors | prevents target leakage; asserted in tests |
| **Strictly causal** models (unidirectional RNN, chomped TCN, masked Transformer) | matches the online setting; perturbation tests enforce it |
| **Focal loss + `pos_weight`** from train prevalence | positive hours are ~2–5 % |
| Threshold from the **utility curve**, not 0.5 / Youden | that is what the challenge rewards |
| **Conformal** sets on top of calibration | calibration fixes average probabilities; conformal gives a per-prediction coverage guarantee and an explicit "don't know" |
| **Clustered** bootstrap CIs | hours within a stay are not independent |

---

## Reproducing / extending

```bash
make train-all          # every model on the configured source
make test               # 48 tests
make cov                # coverage
bash scripts/reproduce_all.sh physionet    # after `sepsis download --full`
docker compose up        # API on :8000  + MLflow UI on :5000
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for ideas (learned imputation, adaptive
conformal, a MIMIC-IV loader, group-DRO).

---

## Data use & citation

PhysioNet/CinC 2019 data are released under the PhysioNet Credentialed Health
Data License. If you use this repository, cite the challenge
(Reyna et al., *Crit Care Med* 2020) and this project
([`CITATION.cff`](CITATION.cff)). **Not a medical device. Research and education
only.**
