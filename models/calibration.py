import numpy as np
from scipy.optimize import minimize
from config import MEAL_SCHEDULE
from config import INSULIN_SCHEDULE
from patient_model import simulate_patient


def calibrate_patient(
    cgm,
    meal_schedule,
    insulin_schedule,
    params_init
):

    theta0 = [
        params_init["kabs"] * 0.6,
        params_init["kp1"] * 1.3,
        params_init["Vm0"] * 0.8
    ]

    def loss(theta):

        p = params_init.copy()

        p["kabs"] = theta[0]
        p["kp1"] = theta[1]
        p["Vm0"] = theta[2]

        _, _, cgm_model = simulate_patient(
            p,
            MEAL_SCHEDULE,
            INSULIN_SCHEDULE,
            len(cgm)
        )

        return np.mean((cgm - cgm_model) ** 2)

    result = minimize(
        loss,
        theta0,
        method="Nelder-Mead",
        options={
            "maxiter":40,
            "xatol":1e-3,
            "fatol":1e-3
        }
    )

    fitted = params_init.copy()

    fitted["kabs"] = result.x[0]
    fitted["kp1"] = result.x[1]
    fitted["Vm0"] = result.x[2]

    return fitted