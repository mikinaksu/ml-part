"""
Maps a real patient_id (from the Go backend / Postgres DB) to the
physiological parameter row the ODE model (patient_model.py) needs
to run calibration and simulation.

TODO (important): vpatient_params.csv contains synthetic/simulated
patients from whatever simulator your colleague used for training --
it has no relationship to your real signed-up users. Right now this
just deterministically picks a row so the pipeline runs end-to-end,
but before this goes anywhere near real predictions you need a real
per-patient parameter store (e.g. a `patient_model_params` table,
fitted/updated per user over time) instead of borrowing a synthetic row.
"""
import pandas as pd

_PATIENTS_DF = pd.read_csv("vpatient_params.csv")


def get_patient_params(patient_id: int):
    row_index = patient_id % len(_PATIENTS_DF)
    return _PATIENTS_DF.iloc[row_index]