import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model

from pipeline import (
    MODEL_PATH,
    inverse_target,
    load_data,
    load_scaler,
    prepare_training_data,
)


def evaluate() -> dict[str, float]:
    data = load_data()
    _, X_test, _, y_test, fitted_scaler = prepare_training_data(data)
    scaler = load_scaler()
    model = load_model(MODEL_PATH)
    predictions = model.predict(X_test, verbose=0).ravel()
    actual = inverse_target(scaler, y_test)
    predicted = inverse_target(scaler, predictions)
    non_zero = actual != 0
    metrics = {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": float(mean_squared_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "MAPE": float(np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100),
    }
    if not np.allclose(fitted_scaler.data_min_, scaler.data_min_):
        raise ValueError("Saved scaler does not match the current training data.")
    return metrics


if __name__ == "__main__":
    print("\n----- LSTM MODEL EVALUATION -----")
    for name, value in evaluate().items():
        print(f"{name}: {value:.2f}")
