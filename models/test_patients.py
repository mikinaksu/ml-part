import pandas as pd

from patient_model import simulate_patient
from api import digital_twin
from config import MEAL_SCHEDULE
from config import INSULIN_SCHEDULE
from config import SIMULATION_TIME
patients = pd.read_csv("vpatient_params.csv")

print()
print("    ТЕСТИРОВАНИЕ ЦИФРОВОГО ДВОЙНИКА ")
print()

for patient_number in range(10, 15):

    try:

        params = patients.iloc[patient_number]

        time, glucose, cgm = simulate_patient(
            params,
            MEAL_SCHEDULE,
            INSULIN_SCHEDULE,
            SIMULATION_TIME
        )
        request = {

            "type": "existing",

            "patient_id": patient_number,

            "cgm": cgm,

            "meal_schedule": MEAL_SCHEDULE,

            "insulin_schedule": INSULIN_SCHEDULE,

            "params": params

        }

        result = digital_twin(request)

        print("--------------------------------------")
        print("Пациент:", params["Name"])
        print()

        print("Калибровка: OK")
        print("kabs =", round(result["parameters"]["kabs"], 5))
        print("kp1  =", round(result["parameters"]["kp1"], 5))
        print("Vm0  =", round(result["parameters"]["Vm0"], 5))

        print()

        print("Прогноз: OK")
        print("30 мин :", round(result["prediction"]["glucose_30"], 2))
        print("60 мин :", round(result["prediction"]["glucose_60"], 2))
        print("120 мин:", round(result["prediction"]["glucose_120"], 2))

    except Exception as e:

        print("--------------------------------------")
        print("Пациент:", params["Name"])
        print("Ошибка:")
        print(e)

print()
print("   ТЕСТ ЗАВЕРШЕН")