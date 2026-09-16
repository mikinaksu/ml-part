from calibration import calibrate_patient
from prediction import predict_glucose
from lstm_predict import predict_glucose_lstm
CHECK_MIN_READINGS = False

def lstm_api(sequence):

    return predict_glucose_lstm(
        sequence
    )
def calibration_api(request):

    fitted = calibrate_patient(
        request["cgm"],
        request["meal_schedule"],
        request["insulin_schedule"],
        request["params"]
    )

    return {
        "kabs": float(fitted["kabs"]),
        "kp1": float(fitted["kp1"]),
        "Vm0": float(fitted["Vm0"])
    }


def prediction_api(request):

    return predict_glucose(
        request["params"],
        request["meal_schedule"],
        request["insulin_schedule"]
    )

def digital_twin(request):

    if request.get("type") == "new":
        return {
            "prediction": None,
            "parameters": None
        }
    if CHECK_MIN_READINGS:

        if len(request["cgm"]) < 500:
            return {
                "prediction": None,
                "parameters": None
            }

    fitted = calibration_api(request)

    new_params = request["params"].copy()

    new_params["kabs"] = fitted["kabs"]
    new_params["kp1"] = fitted["kp1"]
    new_params["Vm0"] = fitted["Vm0"]

    prediction = prediction_api(
        {
            "params": new_params,
            "meal_schedule": request["meal_schedule"],
            "insulin_schedule": request["insulin_schedule"]
        }
    )
    prediction_lstm = None

    if len(request["cgm"]) >= 500:

        sequence = request["sequence"]

        prediction_lstm = predict_glucose_lstm(sequence)
        return {
        "parameters": fitted,
        "prediction": prediction,
        "prediction_lstm": prediction_lstm
    }
    return {
    "parameters": fitted,
    "prediction": prediction,
    "prediction_lstm": prediction_lstm
}