# Changelog

All notable changes to this project are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Initial sepsis early-warning pipeline: causal sequence models (LSTM / GRU /
  dilated TCN / masked Transformer) and a LightGBM baseline over sliding-window
  features, on the PhysioNet/CinC 2019 task.
- Leak-free patient-level stratified splits with optional cross-hospital
  external test.
- GRU-D-style value / mask / delta preprocessing with train-only statistics.
- Faithful PhysioNet utility score (unit-tested against the published value),
  utility-curve threshold selection, patient-clustered bootstrap CIs, and
  decision-curve analysis.
- Uncertainty: isotonic / Platt / temperature calibration with ECE and
  reliability diagrams; split and Mondrian conformal risk sets with coverage
  checks; MC-dropout.
- Fairness / robustness audit: subgroup gaps (sex / age / unit),
  Gaussian-noise and extra-missingness robustness curves.
- Explainability: TreeSHAP, Integrated Gradients (with completeness), PDP / ALE,
  a global surrogate tree, and Transformer attention rollout.
- FastAPI service replaying the exact preprocessing and returning a calibrated
  hourly risk trajectory, conformal sets, and top drivers.
- Infra: pytest suite, ruff, pre-commit, GitHub Actions (Python 3.10–3.12 plus
  train/serve smoke), Dockerfile, docker-compose (API + MLflow), Makefile.
- Docs: methodology, architecture diagram, model card, technical report,
  bibliography; jupytext notebooks for EDA / modelling / explainability /
  fairness. Sample of 240 real PhysioNet stays for the real-data code path.

### Changed
- MLflow logging self-disables (single warning, then no-op) when the tracking
  store is read-only or unavailable, instead of raising once per epoch.
