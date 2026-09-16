import joblib
import numpy as np

from tensorflow.keras.models import load_model


model = load_model(
    "glucose_lstm_model.keras"
)

scaler = joblib.load(
    "scaler.pkl"
)


def predict_glucose_lstm(sequence):

    sequence = np.asarray(sequence)
    print(sequence.shape)

    sequence = sequence.reshape(1,12,3)

    prediction = model.predict(
        sequence,
        verbose=0
    )

    glucose_min = scaler.data_min_[0]
    glucose_max = scaler.data_max_[0]

    prediction = (
        prediction *
        (glucose_max-glucose_min)
        + glucose_min
    )

    prediction = np.clip(
    prediction,
    40,
    400
)

    prediction[0][1] = (
        prediction[0][0] +
        prediction[0][1]
    ) / 2

    prediction[0][2] = (
        prediction[0][1] +
        prediction[0][2]
    ) / 2

    return {

        "glucose_30":
        float(prediction[0][0]),

        "glucose_60":
        float(prediction[0][1]),

        "glucose_120":
        float(prediction[0][2])

    } 

