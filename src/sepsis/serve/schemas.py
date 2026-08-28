"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from sepsis.constants import CHANNELS

_OPTIONAL_FLOAT = (float, type(None))


class HourObservation(BaseModel):
    """One ICU hour. Every physiological channel is optional; omitted channels
    are treated as *not measured* this hour (the model handles missingness
    natively). ``ICULOS`` defaults to the row position if not supplied."""

    model_config = {"extra": "forbid"}

    HR: float | None = None
    O2Sat: float | None = None
    Temp: float | None = None
    SBP: float | None = None
    MAP: float | None = None
    DBP: float | None = None
    Resp: float | None = None
    EtCO2: float | None = None
    BaseExcess: float | None = None
    HCO3: float | None = None
    FiO2: float | None = None
    pH: float | None = None
    PaCO2: float | None = None
    SaO2: float | None = None
    AST: float | None = None
    BUN: float | None = None
    Alkalinephos: float | None = None
    Calcium: float | None = None
    Chloride: float | None = None
    Creatinine: float | None = None
    Bilirubin_direct: float | None = None
    Glucose: float | None = None
    Lactate: float | None = None
    Magnesium: float | None = None
    Phosphate: float | None = None
    Potassium: float | None = None
    Bilirubin_total: float | None = None
    TroponinI: float | None = None
    Hct: float | None = None
    Hgb: float | None = None
    PTT: float | None = None
    WBC: float | None = None
    Fibrinogen: float | None = None
    Platelets: float | None = None
    Age: float | None = None
    Gender: float | None = None
    Unit1: float | None = None
    Unit2: float | None = None
    HospAdmTime: float | None = None
    ICULOS: float | None = None

    def as_row(self) -> dict[str, float | None]:
        d = self.model_dump()
        return {c: d.get(c) for c in CHANNELS}


class PredictRequest(BaseModel):
    model_config = {"extra": "forbid"}

    patient_id: str = Field("anonymous", max_length=64)
    hours: list[HourObservation] = Field(..., min_length=1, max_length=1000)
    explain: bool = Field(True, description="attach top risk drivers for the last hour")

    @field_validator("hours")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("at least one hour of data is required")
        return v


class Driver(BaseModel):
    feature: str
    contribution: float


class HourPrediction(BaseModel):
    hour: int
    risk: float
    risk_uncalibrated: float
    mc_dropout_std: float | None = None
    conformal_set: list[int]
    conformal_uncertain: bool
    alarm: bool


class PredictResponse(BaseModel):
    patient_id: str
    n_hours: int
    model_name: str
    alarm_threshold: float
    conformal_alpha: float
    first_alarm_hour: int | None
    max_risk: float
    trajectory: list[HourPrediction]
    top_drivers: list[Driver] = []
