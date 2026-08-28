# Architecture

## Pipeline

```mermaid
flowchart TD
    A[PhysioNet PSV<br/>or synthetic generator] --> B[load_dataset<br/>PatientRecord list]
    B --> C{make_splits<br/>patient-level, stratified}
    C -->|train| D[Preprocessor.fit<br/>train-only stats]
    C -->|val/test| E[Preprocessor.transform]
    D --> E
    E --> F[TensorDataset<br/>N x T x 107]

    F --> G1[Sequence branch<br/>LSTM / GRU / TCN / Transformer]
    F --> G2[GBM branch<br/>WindowFeatureExtractor -> LightGBM]

    G1 --> H[per-stay hourly risk scores]
    G2 --> H

    H --> I[Calibrator.fit on val<br/>isotonic / Platt / temperature]
    I --> J[ConformalRiskClassifier.fit on val]
    I --> K[best_threshold_by_utility on val]

    J --> L[Evaluation on test]
    K --> L
    L --> M1[discrimination + clustered bootstrap CIs]
    L --> M2[PhysioNet utility score]
    L --> M3[decision-curve analysis]
    L --> M4[fairness: subgroup gaps]
    L --> M5[robustness: noise + missingness]
    L --> M6[explainability: SHAP / IG / PDP-ALE / surrogate / attention]

    M1 & M2 & M3 & M4 & M5 & M6 --> N[results.json + figures/]
    I & J & K & G1 & G2 --> O[serialised bundle]
    O --> P[SepsisPredictor -> FastAPI /predict]
```

## Package layout

| package | responsibility |
| --- | --- |
| `sepsis.data` | PSV IO, synthetic cohort, leak-free splitting, GRU-D-style preprocessing, torch datasets |
| `sepsis.features` | sliding-window tabular features for the GBM |
| `sepsis.models` | masked losses, causal LSTM/GRU/TCN/Transformer, LightGBM wrapper, registry |
| `sepsis.training` | trainer (early stopping, MLflow), `run_experiment` orchestrator, `sepsis` CLI |
| `sepsis.evaluation` | metrics, official utility score, clustered bootstrap, decision curves |
| `sepsis.uncertainty` | calibration + diagnostics, split/Mondrian conformal, MC-dropout |
| `sepsis.explain` | TreeSHAP, Integrated Gradients, PDP/ALE, global surrogate tree, attention rollout |
| `sepsis.audit` | subgroup fairness, perturbation/shift robustness |
| `sepsis.reporting` | headless matplotlib figures |
| `sepsis.serve` | Pydantic schemas, `SepsisPredictor`, FastAPI app |

## Serialised bundle (`artifacts/<run>/`)

```
config.json           full ExperimentConfig (round-trips)
preprocess.json       PreprocessArtifacts: frozen feature order + train stats
model.pt | model.txt  torch checkpoint (best val AUPRC) or LightGBM text model
calibrator.joblib     fitted post-hoc calibrator
conformal.joblib      fitted ConformalRiskClassifier
operating_point.json  {alarm_threshold, conformal_alpha}
results.json          every metric, CI, subgroup table, robustness curve, ranking
figures/*.png         reliability, decision curve, robustness, utility-threshold,
                      subgroup gaps, global attributions, example trajectory,
                      training history
surrogate_rules.txt   human-readable depth-4 surrogate tree
```
