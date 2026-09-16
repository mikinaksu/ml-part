import numpy as np

from patient_model import simulate_patient
from config import MEAL_SCHEDULE
from config import INSULIN_SCHEDULE
from config import SIMULATION_TIME

def predict_glucose(
    params,
    meal_schedule,
    insulin_schedule
):

    _, _, cgm = simulate_patient(
        params,
        MEAL_SCHEDULE,
        INSULIN_SCHEDULE,
        SIMULATION_TIME
    )

    # glucose30 = np.clip(glucose30, 40, 400)
    # glucose60 = np.clip(glucose60, 40, 400)
    # glucose120 = np.clip(glucose120, 40, 400)

    return {
        "glucose_30": float(cgm[29]),
        "glucose_60": float(cgm[59]),
        "glucose_120": float(cgm[119])
    }