"""Load a trained run and score a fresh ICU stay, hour by hour.

A :class:`PredictionBundle` is the on-disk contract written by
``sepsis.training.experiment.run_experiment``:

    run_dir/
      config.json            the full ExperimentConfig
      preprocess.json        PreprocessArtifacts (frozen feature order + stats)
      model.pt | model.txt   sequence checkpoint or LightGBM text model
      calibrator.joblib      fitted post-hoc calibrator
      conformal.joblib       fitted ConformalRiskClassifier
      operating_point.json   {alarm_threshold, conformal_alpha}

Scoring is **causal**: hour ``t`` only ever sees hours ``<= t``, so the returned
trajectory is exactly what an online monitor would have emitted in real time.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sepsis.config import ExperimentConfig
from sepsis.constants import CHANNELS
from sepsis.data.preprocess import PreprocessArtifacts, Preprocessor
from sepsis.data.psv import PatientRecord
from sepsis.utils.logging import get_logger

log = get_logger("serve.inference")


@dataclasses.dataclass
class PredictionBundle:
    config: ExperimentConfig
    artifacts: PreprocessArtifacts
    alarm_threshold: float
    conformal_alpha: float
    run_dir: Path

    @classmethod
    def load(cls, run_dir: str | Path) -> PredictionBundle:
        run_dir = Path(run_dir)
        cfg = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
        art = PreprocessArtifacts.load(run_dir / "preprocess.json")
        op = json.loads((run_dir / "operating_point.json").read_text())
        return cls(
            cfg, art, float(op["alarm_threshold"]), float(op.get("conformal_alpha", 0.1)), run_dir
        )


class SepsisPredictor:
    def __init__(self, run_dir: str | Path):
        self.bundle = PredictionBundle.load(run_dir)
        self.pre = Preprocessor.from_artifacts(self.bundle.artifacts)
        self._model = None
        self._is_sequence = self.bundle.config.model.name != "lightgbm"
        self._calibrator = self._maybe(joblib.load, "calibrator.joblib")
        self._conformal = self._maybe(joblib.load, "conformal.joblib")

    # ------------------------------------------------------------------ load --
    def _maybe(self, fn, name):
        p = self.bundle.run_dir / name
        return fn(p) if p.exists() else None

    @property
    def model(self):
        if self._model is not None:
            return self._model
        cfg = self.bundle.config
        if self._is_sequence:
            import torch

            from sepsis.models.registry import build_sequence_model

            net = build_sequence_model(cfg.model, self.bundle.artifacts.n_features)
            ckpt = torch.load(self.bundle.run_dir / "model.pt", map_location="cpu")
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
            self._model = net
        else:
            from sepsis.models.gbm import GBMRiskModel

            self._model = GBMRiskModel(cfg.model).load(str(self.bundle.run_dir / "model.txt"))
        return self._model

    # --------------------------------------------------------------- predict --
    def _record_from_rows(self, patient_id: str, rows: list[dict]) -> PatientRecord:
        frame = pd.DataFrame([{c: r.get(c) for c in CHANNELS} for r in rows])
        frame = frame.reindex(columns=CHANNELS).astype("float32")
        if frame["ICULOS"].isna().all():
            frame["ICULOS"] = np.arange(1, len(frame) + 1, dtype="float32")
        return PatientRecord(
            pid=patient_id,
            frame=frame,
            label=np.zeros(len(frame), dtype=np.int8),
            source="live",
        )

    def predict(self, patient_id: str, rows: list[dict], explain: bool = True) -> dict:
        rec = self._record_from_rows(patient_id, rows)
        td = self.pre.transform([rec])
        n = int(td.lengths[0])

        if self._is_sequence:
            import torch

            x = torch.tensor(td.X[:1, :n], dtype=torch.float32)
            pad = torch.ones(1, n, dtype=torch.bool)
            with torch.no_grad():
                raw = torch.sigmoid(self.model(x, pad))[0].numpy().astype(float)
        else:
            from sepsis.features.tabular import WindowFeatureExtractor

            tab = WindowFeatureExtractor(self.bundle.config.model.gbm_window).transform(td)
            raw = np.zeros(n)
            p = self.model.predict(tab)
            for row in range(len(tab)):
                if int(tab.times[row]) < n:
                    raw[int(tab.times[row])] = p[row]

        cal = self._calibrator.transform(raw) if self._calibrator else raw
        thr = self.bundle.alarm_threshold
        traj = []
        first_alarm = None
        for t in range(n):
            cset = [0, 1]
            uncertain = True
            if self._conformal is not None:
                s = self._conformal.predict_set(np.array([cal[t]]))[0]
                cset = [c for c in (0, 1) if s[c]]
                uncertain = len(cset) == 2
            alarm = bool(cal[t] >= thr)
            if alarm and first_alarm is None:
                first_alarm = t
            traj.append(
                {
                    "hour": t,
                    "risk": float(cal[t]),
                    "risk_uncalibrated": float(raw[t]),
                    "mc_dropout_std": None,
                    "conformal_set": cset,
                    "conformal_uncertain": uncertain,
                    "alarm": alarm,
                }
            )

        drivers = []
        if explain and self._is_sequence and n > 0:
            drivers = self._drivers(td.X[0], n)

        return {
            "patient_id": patient_id,
            "n_hours": n,
            "model_name": self.bundle.config.model.name,
            "alarm_threshold": thr,
            "conformal_alpha": self.bundle.conformal_alpha,
            "first_alarm_hour": first_alarm,
            "max_risk": float(np.max(cal)) if n else 0.0,
            "trajectory": traj,
            "top_drivers": drivers,
        }

    def _drivers(self, x_full: np.ndarray, n: int, k: int = 8) -> list[dict]:
        from sepsis.explain.attributions import group_attributions, integrated_gradients

        ig = integrated_gradients(self.model, x_full[:n], target_t=n - 1, steps=32)
        grouped = group_attributions(ig["per_feature"], self.bundle.artifacts.feature_names)
        return [{"feature": f, "contribution": float(v)} for f, v in grouped[:k]]
