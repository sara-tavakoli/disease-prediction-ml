"""FastAPI inference service.

    uvicorn sepsis.serve.api:app --port 8000
    SEPSIS_RUN_DIR=artifacts/transformer uvicorn sepsis.serve.api:app

Endpoints
---------
GET  /health   liveness + whether a model is loaded
GET  /model    the served run's config summary and operating point
POST /predict  score an ICU stay; returns the hourly calibrated risk
               trajectory, conformal prediction sets and (optionally) the
               top risk drivers for the most recent hour
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from sepsis.serve.inference import SepsisPredictor
from sepsis.serve.schemas import PredictRequest, PredictResponse
from sepsis.utils.logging import get_logger

log = get_logger("serve.api")
app = FastAPI(
    title="Sepsis Early-Warning API",
    version="0.1.0",
    description="Hourly sepsis-risk trajectory with calibrated probabilities "
    "and conformal prediction sets.",
)


@lru_cache(maxsize=1)
def _predictor() -> SepsisPredictor:
    run_dir = os.environ.get("SEPSIS_RUN_DIR", "artifacts/run")
    log.info("loading model bundle from %s", run_dir)
    return SepsisPredictor(run_dir)


@app.get("/health")
def health() -> dict:
    try:
        p = _predictor()
        loaded = True
        model_name = p.bundle.config.model.name
    except Exception as exc:  # pragma: no cover
        loaded, model_name = False, None
        log.warning("model not loaded: %s", exc)
    return {"status": "ok", "model_loaded": loaded, "model_name": model_name}


@app.get("/model")
def model_info() -> dict:
    p = _predictor()
    c = p.bundle.config
    return {
        "model": c.model.name,
        "hidden_size": c.model.hidden_size,
        "num_layers": c.model.num_layers,
        "n_features": p.bundle.artifacts.n_features,
        "max_seq_len": p.bundle.artifacts.max_seq_len,
        "alarm_threshold": p.bundle.alarm_threshold,
        "conformal_alpha": p.bundle.conformal_alpha,
        "data_source": c.data.source,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        p = _predictor()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"model unavailable: {exc}") from exc
    rows = [h.as_row() for h in req.hours]
    try:
        result = p.predict(req.patient_id, rows, explain=req.explain)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(**result)
