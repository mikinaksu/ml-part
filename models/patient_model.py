import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
def model(t, x, p, meal, insulin):
    dxdt = np.zeros(13)

    d = meal * 1000
    insulin = insulin * 6000 / p["BW"]

    qsto = x[0] + x[1]
    kgut = p["kmax"]

    dxdt[0] = -p["kmax"] * x[0] + d
    dxdt[1] = p["kmax"] * x[0] - kgut * x[1]
    dxdt[2] = kgut * x[1] - p["kabs"] * x[2]

    # =====================
    # Поступление глюкозы
    # =====================
    Rat = p["f"] * p["kabs"] * x[2] / p["BW"]

    # =====================
    # Продукция глюкозы
    # =====================
    EGP = p["kp1"] - p["kp2"] * x[3] - p["kp3"] * x[8]

    # =====================
    # Независимое потребление
    # =====================
    Uii = p["Fsnc"]

    # =====================
    # Почечное выведение
    # =====================
    if x[3] > p["ke2"]:
        Et = p["ke1"] * (x[3] - p["ke2"])
    else:
        Et = 0

    # =====================
    # Глюкоза плазмы
    # =====================
    dxdt[3] = (
        max(EGP, 0)
        + Rat
        - Uii
        - Et
        - p["k1"] * x[3]
        + p["k2"] * x[4]
    )

    # =====================
    # Глюкоза тканей
    # =====================
    Vmt = p["Vm0"] + p["Vmx"] * x[6]
    Kmt = p["Km0"]
    Uid = Vmt * x[4] / (Kmt + x[4])

    dxdt[4] = (
        -Uid
        + p["k1"] * x[3]
        - p["k2"] * x[4]
    )

    # =====================
    # Инсулин в плазме
    # =====================
    dxdt[5] = (
        -(p["m2"] + p["m4"]) * x[5]
        + p["m1"] * x[9]
        + p["ka1"] * x[10]
        + p["ka2"] * x[11]
    )
    It = x[5] / p["Vi"]

    # =====================
    # Действие инсулина
    # =====================
    dxdt[6] = -p["p2u"] * x[6] + p["p2u"] * (It - p["Ib"])
    dxdt[7] = -p["ki"] * (x[7] - It)
    dxdt[8] = -p["ki"] * (x[8] - x[7])

    # =====================
    # Инсулин печени
    # =====================
    dxdt[9] = (
        -(p["m1"] + p["m30"]) * x[9]
        + p["m2"] * x[5]
    )

    # =====================
    # Подкожный инсулин
    # =====================
    dxdt[10] = insulin - (p["ka1"] + p["kd"]) * x[10]
    dxdt[11] = (
        p["kd"] * x[10]
        - p["ka2"] * x[11]
    )

    # =====================
    # Подкожная глюкоза (CGM)
    # =====================
    dxdt[12] = (
        -p["ksc"] * x[12]
        + p["ksc"] * x[3]
    )

    return dxdt

def simulate_patient(params, meal_schedule, insulin_schedule, minutes=300):

    x = np.array(params.iloc[2:15], dtype=float)

    glucose = []
    cgm = []
    time = []

    for t in range(minutes):

        meal = meal_schedule.get(t, 0)
        insulin = insulin_schedule.get(t, 0)

        sol = solve_ivp(
            model,
            [t, t + 1],
            x,
            args=(params, meal, insulin),
            t_eval=[t + 1]
        )

        x = sol.y[:, -1]

        glucose.append(x[3])

        cgm.append(
            x[12] + np.random.normal(0, 5)
        )

        time.append(t + 1)

    return (
        np.array(time),
        np.array(glucose),
        np.array(cgm)
    )