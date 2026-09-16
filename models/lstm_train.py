import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from patient_model import simulate_patient
from config import (
    MEAL_SCHEDULE,
    INSULIN_SCHEDULE,
    SIMULATION_TIME
)

# ====================================
# Генерация датасета
# ====================================

patients = pd.read_csv("vpatient_params.csv")

dataset = []

for patient_id in range(len(patients)):

    params = patients.iloc[patient_id]

    time, glucose, cgm = simulate_patient(

        params,

        MEAL_SCHEDULE,

        INSULIN_SCHEDULE,

        SIMULATION_TIME

    )

    features = []

    for t in range(len(cgm)):

        features.append([

            cgm[t],

            INSULIN_SCHEDULE.get(t, 0),

            MEAL_SCHEDULE.get(t, 0)

        ])

    dataset.extend(features)

dataset = np.array(dataset)

print("Размер датасета:", dataset.shape)

# ====================================
# Нормализация
# ====================================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(dataset)

# ====================================
# Последовательности
# ====================================

def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(

        len(data) - sequence_length - 24

    ):

        X.append(

            data[i:i+sequence_length]

        )

        y.append([

            data[i+sequence_length+6,0],

            data[i+sequence_length+12,0],

            data[i+sequence_length+24,0]

        ])

    return np.array(X), np.array(y)

sequence_length = 12

X, y = create_sequences(

    scaled_data,

    sequence_length

)

print("X:", X.shape)
print("y:", y.shape)

# ====================================
# Train / Test
# ====================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# ====================================
# LSTM
# ====================================

model = Sequential()

model.add(

    Input(shape=(12,3))

)

model.add(

    LSTM(

        64,

        return_sequences=True

    )

)

model.add(

    Dropout(0.2)

)

model.add(

    LSTM(32)

)

model.add(

    Dense(

        16,

        activation="relu"

    )

)

model.add(

    Dense(3)

)

model.compile(

    optimizer="adam",

    loss="mse",

    metrics=["mae"]

)

# ====================================
# Обучение
# ====================================

history = model.fit(

    X_train,

    y_train,

    epochs=20,

    batch_size=32,

    validation_data=(

        X_test,

        y_test

    )

)

# ====================================
# Проверка
# ====================================

predictions = model.predict(X_test)

glucose_min = scaler.data_min_[0]
glucose_max = scaler.data_max_[0]

predictions_real = (

    predictions *

    (glucose_max-glucose_min)

    + glucose_min

)

y_test_real = (

    y_test *

    (glucose_max-glucose_min)

    + glucose_min

)

labels = [30,60,120]

for horizon, minutes in enumerate(labels):

    rmse = np.sqrt(

        mean_squared_error(

            y_test_real[:,horizon],

            predictions_real[:,horizon]

        )

    )

    mae = mean_absolute_error(

        y_test_real[:,horizon],

        predictions_real[:,horizon]

    )

    r2 = r2_score(

        y_test_real[:,horizon],

        predictions_real[:,horizon]

    )

    print()

    print(f"{minutes} минут")

    print("RMSE =", rmse)

    print("MAE =", mae)

    print("R² =", r2)

# ====================================
# Сохранение
# ====================================

model.save(

    "glucose_lstm_model.keras"

)

joblib.dump(

    scaler,

    "scaler.pkl"

)

print()

print("Модель сохранена.")