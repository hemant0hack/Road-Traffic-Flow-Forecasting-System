import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

from evaluate_model import evaluate
from pipeline import (
    FEATURES,
    MODEL_PATH,
    SCALER_PATH,
    TIME_STEPS,
    create_sequences,
    inverse_target,
    load_data,
    load_scaler,
)


st.set_page_config(
    page_title="Traffic Flow Forecasting",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def get_scaler():
    return load_scaler(SCALER_PATH)


@st.cache_data
def get_predictions() -> tuple[pd.DataFrame, dict[str, float]]:
    data = get_data()
    scaler = get_scaler()
    model = get_model()
    scaled = scaler.transform(data[FEATURES].to_numpy(dtype=np.float32))
    X, y = create_sequences(scaled)
    split = int(len(X) * 0.8)
    predictions = model.predict(X[split:], verbose=0).ravel()
    actual = inverse_target(scaler, y[split:])
    predicted = inverse_target(scaler, predictions)
    results = pd.DataFrame(
        {
            "date_time": data["date_time"].iloc[TIME_STEPS + split:].to_numpy(),
            "actual": actual,
            "predicted": predicted,
        }
    )
    return results, evaluate()


def metric_status(value: float, mae: float) -> str:
    if value < mae:
        return "Low"
    if value < mae * 2:
        return "Moderate"
    return "High"


data = get_data()
predictions, metrics = get_predictions()
latest = float(data["traffic_volume"].iloc[-1])
latest_prediction = float(predictions["predicted"].iloc[-1])

st.title("🚦 Road Traffic Flow Forecasting")
st.caption("LSTM-based traffic forecasting using historical traffic and weather patterns")

with st.sidebar:
    st.header("Dashboard Controls")
    chart_points = st.slider("Prediction chart points", 50, 500, 200, 50)
    st.info(
        f"Model uses the previous {TIME_STEPS} records and {len(FEATURES)} input features."
    )

overview, analysis, prediction_tab, model_tab = st.tabs(
    ["Overview", "Traffic Analysis", "Predict Traffic", "Model Performance"]
)

with overview:
    st.subheader("Current status")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total records", f"{len(data):,}")
    col2.metric("Latest traffic", f"{latest:,.0f}")
    col3.metric("Next prediction", f"{latest_prediction:,.0f}")
    col4.metric("Status", metric_status(latest_prediction, metrics["MAE"]))

    st.subheader("Recent traffic")
    recent = data[["date_time", "traffic_volume"]].tail(200).set_index("date_time")
    st.line_chart(recent)
    st.dataframe(data.tail(10), use_container_width=True, hide_index=True)

with analysis:
    st.subheader("Traffic patterns")
    chart_col, day_col = st.columns(2)
    with chart_col:
        hourly = data.groupby("hour", as_index=True)["traffic_volume"].mean()
        st.bar_chart(hourly)
    with day_col:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily = data.assign(day_name=data["date_time"].dt.day_name()).groupby("day_name")[
            "traffic_volume"
        ].mean().reindex(days)
        st.bar_chart(daily)

    st.subheader("Weather and traffic")
    weather_feature = st.selectbox("Select weather feature", ["temp", "rain_1h", "snow_1h", "clouds_all"])
    weather_data = data[["date_time", weather_feature, "traffic_volume"]].tail(500).set_index("date_time")
    st.line_chart(weather_data)

with prediction_tab:
    st.subheader("Predict the next traffic value")
    st.write("Update the latest conditions. The model uses the previous 24 records as context.")
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        traffic_input = st.number_input("Current traffic volume", 0.0, 10000.0, latest, step=50.0)
        temperature = st.number_input("Temperature (Kelvin)", 150.0, 350.0, float(data["temp"].iloc[-1]))
        rain = st.number_input("Rain (mm)", 0.0, 500.0, float(data["rain_1h"].iloc[-1]))
        snow = st.number_input("Snow (mm)", 0.0, 500.0, float(data["snow_1h"].iloc[-1]))
    with form_col2:
        clouds = st.slider("Cloud coverage", 0, 100, int(data["clouds_all"].iloc[-1]))
        hour = st.slider("Hour of day", 0, 23, int(data["hour"].iloc[-1]))
        day = st.selectbox(
            "Day of week",
            list(range(7)),
            index=int(data["day_of_week"].iloc[-1]),
            format_func=lambda value: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][value],
        )

    if st.button("Predict traffic", type="primary"):
        recent = data[FEATURES].tail(TIME_STEPS).to_numpy(dtype=np.float32).copy()
        recent[-1] = [
            traffic_input,
            temperature,
            rain,
            snow,
            clouds,
            hour,
            day,
            int(day >= 5),
            int(hour in [7, 8, 9, 16, 17, 18]),
        ]
        scaled_input = get_scaler().transform(recent)[np.newaxis, ...]
        prediction = get_model().predict(scaled_input, verbose=0).ravel()[0]
        predicted_traffic = float(inverse_target(get_scaler(), prediction))
        st.success(f"Predicted traffic volume: {predicted_traffic:,.0f} vehicles")
        st.metric("Expected traffic level", metric_status(predicted_traffic, metrics["MAE"]))

with model_tab:
    st.subheader("LSTM evaluation on the chronological test set")
    metric_cols = st.columns(5)
    for column, (name, value) in zip(metric_cols, metrics.items()):
        suffix = "%" if name == "MAPE" else ""
        column.metric(name, f"{value:,.2f}{suffix}")

    st.subheader("Actual vs predicted")
    chart_data = predictions.tail(chart_points).set_index("date_time")[["actual", "predicted"]]
    st.line_chart(chart_data)

    st.subheader("Prediction errors")
    errors = predictions.assign(error=lambda frame: frame["actual"] - frame["predicted"])
    st.line_chart(errors.tail(chart_points).set_index("date_time")["error"])
    st.caption("Lower MAE, RMSE and MAPE are better. Higher R² is better.")

