"""
Sends prediction results back to the Go backend's
handlerAddPredictions endpoint.
"""
from datetime import datetime, timezone
import logging
import requests

logger = logging.getLogger("model_service")


class BackendRejected(Exception):
    """Backend responded with a non-2xx status (e.g. recs service not ready yet)."""
    pass


def send_predictions_to_backend(
    backend_url: str,
    model_id: str,
    patient_id: int,
    last_reading_time: str,
    prediction: dict,
):
    """
    prediction: dict with keys glucose_30, glucose_60, glucose_120
    (this is exactly what prediction.py / lstm_predict.py already return).

    """
    payload = [{
        "glucose30": prediction["glucose_30"],
        "glucose60": prediction["glucose_60"],
        "glucose120": prediction["glucose_120"],
    }]

    headers = {
        "Content-Type": "application/json",
        "Model-ID": model_id,
        "Patient-ID": str(patient_id),
        "Last-Reading-Time": last_reading_time,
        "Created-At": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    resp = requests.post(backend_url, json=payload, headers=headers, timeout=10)

    if not resp.ok:
        logger.warning(
            "Backend rejected prediction for patient %s (status %s): %s",
            patient_id, resp.status_code, resp.text[:200],
        )
        raise BackendRejected(f"{resp.status_code}: {resp.text[:200]}")

    return resp