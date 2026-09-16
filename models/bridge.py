"""
Turns the payload the Go backend sends (see handlerSendReadingsToModel)
into the shapes the existing model code (patient_model.py, calibration.py,
prediction.py, api.py) expects.
"""
from datetime import datetime
import numpy as np


def _parse_timestamp(ts: str) -> datetime:
    # Go encodes timestamps with time.RFC3339, e.g. "2026-07-24T10:00:00Z"
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build_model_inputs(readings: list[dict]):
    """
    readings: list of dicts shaped like the Go `ModelReading` struct:
        {"timestamp": str, "glucose": float, "insulin": float,
         "meal": float, "exercise_duration": float, "exercise_intensity": float}

    Returns (cgm, meal_schedule, insulin_schedule) in the format
    simulate_patient / calibrate_patient / predict_glucose already expect:
      - cgm: np.ndarray of glucose readings
      - meal_schedule / insulin_schedule: {minute_offset: amount}
        where minute_offset is minutes since the first reading.

    NOTE: patient_model.py's ODE integrator advances one simulated
    "minute" per readings interval. If your device readings aren't
    actually spaced ~1 minute apart, the minute offsets below will
    under/over-compress real elapsed time relative to what the ODE
    model assumes. Flag this with whoever owns patient_model.py --
    it's a modeling assumption, not something this layer can silently fix.
    """
    if not readings:
        raise ValueError("no readings provided")

    timestamps = [_parse_timestamp(r["timestamp"]) for r in readings]
    t0 = timestamps[0]

    cgm = np.array([float(r["glucose"]) for r in readings], dtype=float)

    meal_schedule = {}
    insulin_schedule = {}

    for reading, ts in zip(readings, timestamps):
        minute = int((ts - t0).total_seconds() // 60)
        meal = float(reading.get("meal") or 0)
        insulin = float(reading.get("insulin") or 0)
        if meal > 0:
            meal_schedule[minute] = meal
        if insulin > 0:
            insulin_schedule[minute] = insulin

    return cgm, meal_schedule, insulin_schedule