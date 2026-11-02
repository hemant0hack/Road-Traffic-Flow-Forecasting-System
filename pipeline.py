from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "traffic.csv"
MODEL_PATH = PROJECT_DIR / "traffic_lstm.keras"
SCALER_PATH = PROJECT_DIR / "traffic_scaler.save"
TIME_STEPS = 24
TEST_RATIO = 0.2

FEATURES = [
    "traffic_volume",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"date_time", "traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data["date_time"] = pd.to_datetime(data["date_time"], errors="raise")
    data = data.sort_values("date_time").reset_index(drop=True)
    data["hour"] = data["date_time"].dt.hour
    data["day_of_week"] = data["date_time"].dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_rush_hour"] = data["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data[FEATURES] = data[FEATURES].interpolate(limit_direction="both")
    if data[FEATURES].isnull().any().any():
        raise ValueError("Dataset contains missing numeric values after preprocessing.")
    return data


def create_sequences(
    scaled_values: np.ndarray,
    time_steps: int = TIME_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    if len(scaled_values) <= time_steps:
        raise ValueError("Dataset must contain more rows than the time-step window.")
    X = np.array([scaled_values[i - time_steps:i] for i in range(time_steps, len(scaled_values))])
    y = scaled_values[time_steps:, 0]
    return X, y


def prepare_training_data(
    data: pd.DataFrame,
    time_steps: int = TIME_STEPS,
    test_ratio: float = TEST_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    values = data[FEATURES].to_numpy(dtype=np.float32)
    split = int((len(values) - time_steps) * (1 - test_ratio))
    scaler = MinMaxScaler()
    scaler.fit(values[: split + time_steps])
    scaled_values = scaler.transform(values)
    X, y = create_sequences(scaled_values, time_steps)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test, scaler


def inverse_target(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    return np.asarray(values) * scaler.data_range_[0] + scaler.data_min_[0]


def save_scaler(scaler: MinMaxScaler, path: Path = SCALER_PATH) -> None:
    joblib.dump(scaler, path)


def load_scaler(path: Path = SCALER_PATH) -> MinMaxScaler:
    return joblib.load(path)
