"""Column names, feature groups and challenge constants for PhysioNet/CinC 2019.

Reference: Reyna et al., "Early Prediction of Sepsis from Clinical Data: The
PhysioNet/Computing in Cardiology Challenge 2019", Critical Care Medicine, 2020.
"""

from __future__ import annotations

# --- Vital signs (8), sampled ~hourly ---------------------------------------
VITALS: list[str] = [
    "HR",  # heart rate (bpm)
    "O2Sat",  # pulse oximetry (%)
    "Temp",  # temperature (deg C)
    "SBP",  # systolic BP (mmHg)
    "MAP",  # mean arterial pressure (mmHg)
    "DBP",  # diastolic BP (mmHg)
    "Resp",  # respiration rate (breaths/min)
    "EtCO2",  # end-tidal CO2 (mmHg)
]

# --- Laboratory values (26), sparsely sampled -----------------------------
LABS: list[str] = [
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
]

# --- Demographics / context (6) -----------------------------------------
DEMOGRAPHICS: list[str] = [
    "Age",  # years (capped at 100 in the source)
    "Gender",  # 1 = male, 0 = female
    "Unit1",  # administrative ICU identifier (MICU)
    "Unit2",  # administrative ICU identifier (SICU)
    "HospAdmTime",  # hours between hospital and ICU admission (<= 0)
    "ICULOS",  # ICU length-of-stay counter (hours since ICU admit)
]

CHANNELS: list[str] = VITALS + LABS + DEMOGRAPHICS
LABEL_COL: str = "SepsisLabel"

# Static columns are carried forward unchanged and never differenced.
STATIC_COLS: list[str] = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
# Time-varying channels that get delta / rolling / time-since features.
DYNAMIC_COLS: list[str] = VITALS + LABS + ["ICULOS"]

# --- Challenge scoring constants (utility function) --------------------------
# The label is shifted +6h for sepsis patients, so a "just in time" alarm fires
# 6h before the clinical onset of sepsis (t_sepsis - 6).
SEPSIS_LABEL_SHIFT_HOURS: int = 6
UTILITY_DT_EARLY: int = -12  # earliest hour (rel. to t_sepsis) rewarded
UTILITY_DT_OPTIMAL: int = -6  # hour of maximum reward
UTILITY_DT_LATE: int = 3  # last hour a positive prediction still scores
UTILITY_MAX_TP: float = 1.0
UTILITY_MIN_FN: float = -2.0
UTILITY_U_FP: float = -0.05
UTILITY_U_TN: float = 0.0

# Physiologically plausible clipping ranges used for outlier handling and by the
# synthetic generator. (low, high)
PHYSIOLOGIC_RANGES: dict[str, tuple[float, float]] = {
    "HR": (20.0, 300.0),
    "O2Sat": (50.0, 100.0),
    "Temp": (30.0, 44.0),
    "SBP": (40.0, 300.0),
    "MAP": (20.0, 250.0),
    "DBP": (10.0, 200.0),
    "Resp": (2.0, 80.0),
    "EtCO2": (5.0, 80.0),
    "pH": (6.6, 7.8),
    "Lactate": (0.1, 30.0),
    "WBC": (0.0, 100.0),
    "Creatinine": (0.1, 20.0),
    "Platelets": (5.0, 1200.0),
    "Glucose": (10.0, 1000.0),
    "BUN": (1.0, 200.0),
    "Age": (14.0, 100.0),
}

# Approximate population medians (PhysioNet training set A/B) used as a static
# imputation prior when a channel is never observed for a patient.
POPULATION_MEDIANS: dict[str, float] = {
    "HR": 84.0,
    "O2Sat": 98.0,
    "Temp": 36.9,
    "SBP": 120.0,
    "MAP": 79.0,
    "DBP": 59.0,
    "Resp": 18.0,
    "EtCO2": 33.0,
    "BaseExcess": 0.0,
    "HCO3": 24.0,
    "FiO2": 0.5,
    "pH": 7.38,
    "PaCO2": 41.0,
    "SaO2": 97.0,
    "AST": 30.0,
    "BUN": 17.0,
    "Alkalinephos": 82.0,
    "Calcium": 8.4,
    "Chloride": 106.0,
    "Creatinine": 1.0,
    "Bilirubin_direct": 0.3,
    "Glucose": 128.0,
    "Lactate": 1.7,
    "Magnesium": 2.0,
    "Phosphate": 3.5,
    "Potassium": 4.1,
    "Bilirubin_total": 0.7,
    "TroponinI": 0.1,
    "Hct": 31.0,
    "Hgb": 10.4,
    "PTT": 34.0,
    "WBC": 10.6,
    "Fibrinogen": 280.0,
    "Platelets": 200.0,
    "Age": 62.0,
    "Gender": 1.0,
    "Unit1": 0.0,
    "Unit2": 0.0,
    "HospAdmTime": -50.0,
    "ICULOS": 1.0,
}

SUBGROUP_DEFS: dict[str, str] = {
    "sex": "Gender (1=male, 0=female)",
    "age_band": "Age discretised into <40 / 40-64 / 65-79 / 80+",
    "icu_unit": "Unit1 vs Unit2 vs unknown",
}
