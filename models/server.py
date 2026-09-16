"""
FastAPI wrapper around the existing model code (api.py / digital_twin).

Run with:
    uvicorn server:app --host 0.0.0.0 --port 9000

Env vars required:
    BACKEND_PREDICTIONS_URL   Full URL of the Go handlerAddPredictions route
                              (e.g. http://localhost:8080/predictions)
    AI_MODEL_ID               Same UUID as the Go backend's AI_MODEL_ID
    ODU_MODEL_ID              Same UUID as the Go backend's ODU_MODEL_ID
"""
import os
import logging

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks

from bridge import build_model_inputs
from patients import get_patient_params
from callback import send_predictions_to_backend

from api import digital_twin  # colleague's existing module

from callback import send_predictions_to_backend, BackendRejected

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_service")

app = FastAPI()

BACKEND_PREDICTIONS_URL = os.environ["BACKEND_PREDICTIONS_URL"]
AI_MODEL_ID = os.environ.get("AI_MODEL_ID")
ODU_MODEL_ID = os.environ.get("ODU_MODEL_ID")

MIN_READINGS = 12  # matches Go's `readings_amount` constant


def _run_and_callback(patient_id, model_id, last_reading_time, cgm, meal_schedule, insulin_schedule, params):
    try:
        request = {
            "type": "existing",
            "patient_id": patient_id,
            "cgm": cgm,
            "meal_schedule": meal_schedule,
            "insulin_schedule": insulin_schedule,
            "params": params,
        }

        result = digital_twin(request)

        # AI_MODEL_ID and ODU_MODEL_ID currently point at the same UUID in
        # your .env, so this branch is a no-op for now -- but once the two
        # models actually diverge (e.g. ODU_MODEL_ID -> LSTM), give it a
        # distinct ID and this routes automatically.
        if model_id == ODU_MODEL_ID and result.get("prediction_lstm"):
            prediction = result["prediction_lstm"]
        else:
            prediction = result["prediction"]

        if prediction is None:
            logger.info("No prediction produced for patient %s (insufficient/uncalibratable data)", patient_id)
            return

        send_predictions_to_backend(
            backend_url=BACKEND_PREDICTIONS_URL,
            model_id=model_id,
            patient_id=patient_id,
            last_reading_time=last_reading_time,
            prediction=prediction,
        )
    except BackendRejected:
        # Expected race between predictions and readings -- already logged
        # concisely inside send_predictions_to_backend, nothing more to add.
        pass
    except Exception:
        logger.exception("Prediction pipeline failed for patient %s", patient_id)

@app.post("/predict")
async def predict(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()
    readings = body.get("readings", [])

    patient_id_header = request.headers.get("Patient-ID")
    model_id = request.headers.get("Model-ID")
    last_reading_time = request.headers.get("Last-Reading-Timestamp")

    if not patient_id_header or not model_id or not last_reading_time:
        raise HTTPException(status_code=400, detail="missing Patient-ID / Model-ID / Last-Reading-Timestamp header")

    try:
        patient_id = int(patient_id_header)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid Patient-ID header")

    if len(readings) < MIN_READINGS:
        # Go already gates on this before calling us, but guard here too.
        return {"status": "skipped", "reason": "not enough readings"}

    cgm, meal_schedule, insulin_schedule = build_model_inputs(readings)
    params = get_patient_params(patient_id)

    # Return 2xx to Go immediately -- handlerSendReadingsToModel only checks
    # the status code, so we run the actual model + callback in the background.
    background_tasks.add_task(
        _run_and_callback,
        patient_id, model_id, last_reading_time,
        cgm, meal_schedule, insulin_schedule, params,
    )

    return {"status": "accepted"}