"""Slow end-to-end guard: the full pipeline runs, writes a loadable bundle, and
the API scores a stay from it."""

from __future__ import annotations

import pytest

from sepsis.training.experiment import run_experiment

pytestmark = pytest.mark.slow


def test_pipeline_end_to_end_and_serving(smoke_config, tmp_path):
    results = run_experiment(smoke_config)

    for key in (
        "discrimination",
        "calibration",
        "conformal",
        "utility",
        "fairness",
        "robustness",
        "explainability",
    ):
        assert key in results

    run_dir = smoke_config.output_dir
    for f in (
        "config.json",
        "preprocess.json",
        "model.pt",
        "calibrator.joblib",
        "conformal.joblib",
        "operating_point.json",
        "results.json",
    ):
        assert (
            run_dir / f
            if hasattr(run_dir, "__truediv__")
            else __import__("pathlib").Path(run_dir) / f
        ).exists()

    # calibration should not make ECE dramatically worse
    c = results["calibration"]
    assert c["ece_calibrated"]["ece"] <= c["ece_uncalibrated"]["ece"] + 0.05

    # conformal coverage is in a sane band
    assert 0.4 <= results["conformal"]["empirical_coverage"] <= 1.0

    # --- serving ---
    from fastapi.testclient import TestClient

    from sepsis.serve.api import _predictor, app

    _predictor.cache_clear()
    import os

    os.environ["SEPSIS_RUN_DIR"] = str(smoke_config.output_dir)
    client = TestClient(app)
    assert client.get("/health").json()["model_loaded"] is True

    rows = [
        {
            "HR": 90 + i,
            "O2Sat": 97,
            "SBP": 120 - i,
            "MAP": 78 - i,
            "Resp": 18,
            "Temp": 37.0,
            "Age": 70,
            "Gender": 1,
            "Unit1": 1,
            "Unit2": 0,
            "HospAdmTime": -12,
            "ICULOS": i + 1,
        }
        for i in range(12)
    ]
    resp = client.post("/predict", json={"patient_id": "t1", "hours": rows})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_hours"] == 12
    assert len(body["trajectory"]) == 12
    assert all(0.0 <= h["risk"] <= 1.0 for h in body["trajectory"])
