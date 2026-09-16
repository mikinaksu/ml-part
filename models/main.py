import pandas as pd
import numpy as np
from patient_model import simulate_patient
from api import digital_twin
from config import MEAL_SCHEDULE
from config import INSULIN_SCHEDULE
from config import SIMULATION_TIME

# =====================
# Загрузка пациентов
# =====================

patients = pd.read_csv("vpatient_params.csv")

patient_number = int(input("Введите номер пациента: "))

if patient_number < 0 or patient_number >= len(patients):
    print("Пациент не найден.")
    exit()

params = patients.iloc[patient_number]
print()

print("Выбранный пациент:")
print(params["Name"])

print()

print("Параметры пациента:")
print(params)

print()

print("kabs =", params["kabs"])
print("kp1 =", params["kp1"])
print("Vm0 =", params["Vm0"])

time, glucose, cgm = simulate_patient(
    params,
    MEAL_SCHEDULE,
    INSULIN_SCHEDULE,
    SIMULATION_TIME
)
# =====================
# Формирование последовательности для LSTM
# =====================

sequence = []

for i in range(12):

    sequence.append([

        cgm[i],

        INSULIN_SCHEDULE.get(i, 0),

        MEAL_SCHEDULE.get(i, 0)

    ])

sequence = np.array(sequence)

print()

print("CGM первые значения:")

print(cgm[:10])

# =====================
# Запрос к цифровому двойнику
# =====================


request = {

    "cgm": cgm,
    "meal_schedule": MEAL_SCHEDULE,
    "insulin_schedule": INSULIN_SCHEDULE,
    "params": params,
    "sequence": sequence
}

result = digital_twin(request)

if result is None or result["prediction"] is None:
    print()
    print("Недостаточно данных для персонального обучения модели.")
    print("Необходимо накопить около 500 измерений.")
    exit()

# =====================
# Вывод результата
# =====================

print()

print("  ")
print("ЦИФРОВОЙ ДВОЙНИК")
print("  ")


print()

print("Параметры после калибровки:")

print("kabs =", round(result["parameters"]["kabs"],5))
print("kp1  =", round(result["parameters"]["kp1"],5))
print("Vm0  =", round(result["parameters"]["Vm0"],5))

print()

print("  Прогноз:")

print("Математическая модель:")

print("30 минут :", round(result["prediction"]["glucose_30"], 2))
print("60 минут :", round(result["prediction"]["glucose_60"], 2))
print("120 минут:", round(result["prediction"]["glucose_120"], 2))

print()

if result["prediction_lstm"] is None:

    print("LSTM:")
    print("Недостаточно данных.")
    print("Требуется не менее 500 измерений.")

else:

    print("LSTM:")

    print("30 минут :", round(result["prediction_lstm"]["glucose_30"],2))
    print("60 минут :", round(result["prediction_lstm"]["glucose_60"],2))
    print("120 минут:", round(result["prediction_lstm"]["glucose_120"],2))

# =====================
# Сохранение JSON
# =====================

import json

print()

print(json.dumps(result, indent=4))
