"""Early sepsis prediction from clinical time series.

Sub-packages
------------
data          Ingestion (PhysioNet PSV + a physiologically-motivated synthetic
              generator), leak-free patient-level splitting, and hourly
              preprocessing with explicit missingness handling.
features      Sliding-window tabular feature extraction for gradient-boosting
              baselines.
models        Gradient-boosting baseline plus masked LSTM / GRU / TCN /
              Transformer sequence encoders and imbalance-aware losses.
training      Trainer, early stopping, MLflow logging and the ``sepsis`` CLI.
evaluation    Discrimination metrics, the official PhysioNet utility score,
              bootstrap confidence intervals and decision-curve analysis.
uncertainty   Post-hoc calibration, split/Mondrian conformal risk sets and
              MC-dropout epistemic uncertainty.
explain       SHAP, attention roll-out, PDP/ALE and a global surrogate tree.
audit         Subgroup fairness metrics and perturbation / shift robustness.
serve         FastAPI inference service returning a calibrated risk trajectory.
"""

from __future__ import annotations

__version__ = "0.1.0"
