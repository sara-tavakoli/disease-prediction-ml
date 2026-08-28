"""A physiologically-motivated synthetic sepsis cohort.

The generator is *not* a claim about real epidemiology. Its job is to produce
data with the same shape, missingness structure, class imbalance and
sepsis-like temporal deterioration as PhysioNet/CinC 2019 so the entire
pipeline -- preprocessing, sequence models, conformal calibration, SHAP,
subgroup fairness -- is runnable and testable offline. Real experiments should
use :func:`sepsis.data.download.ensure_physionet`.

Design
------
* Static covariates (age, sex, ICU unit, admission offset, stay length) are
  drawn from plausible ICU marginals.
* Each vital sign follows a mean-reverting Ornstein-Uhlenbeck process around a
  patient-specific baseline; baselines depend weakly on age so subgroup effects
  exist for the fairness audit to find.
* Labs are sampled sparsely (per-channel hourly observation probability) and
  otherwise left missing, matching the ~90%+ missing-rate of real labs.
* Septic patients receive a deterioration signal ramped in over the hours
  before a sampled onset time: tachycardia, tachypnoea, rising lactate / WBC /
  creatinine / temperature, falling MAP and platelets. The binary label is then
  shifted +6h exactly as the challenge does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sepsis.constants import (
    CHANNELS,
    LABS,
    PHYSIOLOGIC_RANGES,
    POPULATION_MEDIANS,
    SEPSIS_LABEL_SHIFT_HOURS,
    VITALS,
)
from sepsis.data.psv import PatientRecord

# per-hour probability that a lab is drawn (roughly matches real sparsity)
_LAB_OBS_PROB: dict[str, float] = {
    "Glucose": 0.11,
    "Potassium": 0.09,
    "Creatinine": 0.08,
    "BUN": 0.08,
    "Hct": 0.08,
    "Hgb": 0.08,
    "WBC": 0.07,
    "Platelets": 0.07,
    "Calcium": 0.06,
    "Chloride": 0.06,
    "HCO3": 0.06,
    "Magnesium": 0.05,
    "Lactate": 0.05,
    "pH": 0.045,
    "PaCO2": 0.04,
    "BaseExcess": 0.04,
    "FiO2": 0.05,
    "SaO2": 0.04,
    "Phosphate": 0.035,
    "PTT": 0.03,
    "AST": 0.02,
    "Alkalinephos": 0.02,
    "Bilirubin_total": 0.02,
    "Fibrinogen": 0.012,
    "TroponinI": 0.012,
    "Bilirubin_direct": 0.006,
}
_VITAL_MISS_PROB = 0.06  # vitals present most hours, with short gaps

# OU parameters per vital: (reversion theta, process sigma)
_OU: dict[str, tuple[float, float]] = {
    "HR": (0.25, 3.0),
    "O2Sat": (0.4, 0.8),
    "Temp": (0.15, 0.15),
    "SBP": (0.3, 4.0),
    "MAP": (0.3, 3.0),
    "DBP": (0.3, 3.0),
    "Resp": (0.3, 1.2),
    "EtCO2": (0.3, 2.0),
}


def _clip(name: str, x: np.ndarray) -> np.ndarray:
    lo, hi = PHYSIOLOGIC_RANGES.get(name, (-np.inf, np.inf))
    return np.clip(x, lo, hi)


def _patient_static(rng: np.random.Generator) -> dict[str, float]:
    age = float(np.clip(rng.normal(64, 17), 16, 100))
    gender = float(rng.integers(0, 2))
    unit = rng.choice([1.0, 2.0, np.nan], p=[0.45, 0.45, 0.10])
    unit1 = 1.0 if unit == 1.0 else (0.0 if unit == 2.0 else np.nan)
    unit2 = 1.0 if unit == 2.0 else (0.0 if unit == 1.0 else np.nan)
    hosp_adm = float(-np.abs(rng.exponential(60.0)))
    return {"Age": age, "Gender": gender, "Unit1": unit1, "Unit2": unit2, "HospAdmTime": hosp_adm}


def _baselines(age: float, rng: np.random.Generator) -> dict[str, float]:
    age_z = (age - 64.0) / 17.0
    return {
        "HR": 82 + 4 * rng.standard_normal() - 1.5 * age_z,
        "O2Sat": 97.5 + 0.7 * rng.standard_normal() - 0.3 * max(age_z, 0),
        "Temp": 36.9 + 0.15 * rng.standard_normal(),
        "SBP": 122 + 8 * rng.standard_normal() + 3.0 * age_z,
        "MAP": 80 + 5 * rng.standard_normal() + 1.5 * age_z,
        "DBP": 60 + 5 * rng.standard_normal(),
        "Resp": 17.5 + 1.5 * rng.standard_normal() + 0.4 * age_z,
        "EtCO2": 34 + 3 * rng.standard_normal(),
    }


def _ou_series(
    baseline: float, theta: float, sigma: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    x = np.empty(n, dtype=np.float64)
    x[0] = baseline + sigma * rng.standard_normal()
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (baseline - x[t - 1]) + sigma * rng.standard_normal()
    return x


def _deterioration_ramp(n: int, onset: int, lead: int) -> np.ndarray:
    """Return a 0->1 ramp that starts ``lead`` hours before ``onset``."""
    ramp = np.zeros(n)
    start = max(0, onset - lead)
    if onset > start:
        ramp[start:onset] = np.linspace(0.0, 1.0, onset - start)
    ramp[onset:] = 1.0
    return ramp


def generate_patient(
    pid: str,
    rng: np.random.Generator,
    force_septic: bool | None = None,
    prevalence: float = 0.08,
) -> PatientRecord:
    n = int(np.clip(rng.gamma(shape=2.0, scale=22.0) + 8, 8, 336))
    static = _patient_static(rng)
    base = _baselines(static["Age"], rng)

    septic = rng.random() < prevalence if force_septic is None else force_septic
    onset = int(rng.integers(6, n)) if (septic and n > 7) else -1
    if onset < 0:
        septic = False
    sev = 0.6 + 0.8 * rng.random()  # per-patient severity multiplier
    lead = int(rng.integers(6, 16))
    ramp = _deterioration_ramp(n, onset, lead) if septic else np.zeros(n)

    cols: dict[str, np.ndarray] = {}

    # --- vitals: OU process + sepsis deterioration + missingness --------------
    for v in VITALS:
        theta, sigma = _OU[v]
        series = _ou_series(base[v], theta, sigma, n, rng)
        if septic:
            if v == "HR":
                series += sev * 28 * ramp
            elif v == "Resp":
                series += sev * 10 * ramp
            elif v == "Temp":
                series += sev * 1.4 * ramp * (1.0 - 0.5 * (ramp > 0.85))
            elif v in {"SBP", "MAP", "DBP"}:
                series -= sev * (22 if v == "SBP" else 15) * ramp
            elif v == "O2Sat":
                series -= sev * 4 * ramp
            elif v == "EtCO2":
                series -= sev * 5 * ramp
        series = _clip(v, series)
        miss = rng.random(n) < _VITAL_MISS_PROB
        series[miss] = np.nan
        cols[v] = series.astype(np.float32)

    # --- labs: sparse draws around population medians ------------------------
    for lab in LABS:
        med = POPULATION_MEDIANS.get(lab, 0.0)
        spread = max(abs(med) * 0.18, 0.05)
        obs_p = _LAB_OBS_PROB.get(lab, 0.02)
        drawn = rng.random(n) < obs_p
        values = np.full(n, np.nan, dtype=np.float32)
        base_val = med + spread * rng.standard_normal()
        for t in np.flatnonzero(drawn):
            val = base_val + spread * 0.5 * rng.standard_normal()
            if septic:
                if lab == "Lactate":
                    val += sev * 4.5 * ramp[t]
                elif lab == "WBC":
                    val += sev * 9.0 * ramp[t] * rng.choice([1.0, -0.6])
                elif lab == "Creatinine":
                    val += sev * 1.1 * ramp[t]
                elif lab == "Platelets":
                    val -= sev * 90.0 * ramp[t]
                elif lab == "pH":
                    val -= sev * 0.12 * ramp[t]
                elif lab == "HCO3":
                    val -= sev * 6.0 * ramp[t]
            values[t] = _clip(lab, np.array([val]))[0]
        cols[lab] = values

    # --- context columns ---------------------------------------------------
    cols["Age"] = np.full(n, static["Age"], dtype=np.float32)
    cols["Gender"] = np.full(n, static["Gender"], dtype=np.float32)
    cols["Unit1"] = np.full(n, static["Unit1"], dtype=np.float32)
    cols["Unit2"] = np.full(n, static["Unit2"], dtype=np.float32)
    cols["HospAdmTime"] = np.full(n, static["HospAdmTime"], dtype=np.float32)
    cols["ICULOS"] = np.arange(1, n + 1, dtype=np.float32)

    frame = pd.DataFrame({c: cols[c] for c in CHANNELS})

    # --- label with the challenge's +6h shift -----------------------------
    label = np.zeros(n, dtype=np.int8)
    if septic:
        t_alarm = max(0, onset - SEPSIS_LABEL_SHIFT_HOURS)
        label[t_alarm:] = 1

    return PatientRecord(pid=pid, frame=frame, label=label, source="synthetic")


def generate_cohort(
    n_patients: int = 4000,
    prevalence: float = 0.08,
    seed: int = 20190804,
) -> list[PatientRecord]:
    """Deterministic cohort with an exact-ish septic count."""
    rng = np.random.default_rng(seed)
    n_sep = int(round(n_patients * prevalence))
    flags = np.array([True] * n_sep + [False] * (n_patients - n_sep))
    rng.shuffle(flags)
    out: list[PatientRecord] = []
    for i, is_sep in enumerate(flags):
        child = np.random.default_rng(rng.integers(0, 2**63 - 1))
        out.append(
            generate_patient(f"s{i:06d}", child, force_septic=bool(is_sep), prevalence=prevalence)
        )
    return out


def write_cohort(records: list[PatientRecord], out_dir: str) -> None:
    """Serialise a cohort to ``training_setSYN/*.psv`` for the standard loader."""
    from pathlib import Path

    from sepsis.constants import LABEL_COL

    sub = Path(out_dir) / "training_setSYN"
    sub.mkdir(parents=True, exist_ok=True)
    for rec in records:
        out = rec.frame.copy()
        out[LABEL_COL] = rec.label
        out.to_csv(sub / f"{rec.pid}.psv", sep="|", index=False, na_rep="NaN", float_format="%.4g")
