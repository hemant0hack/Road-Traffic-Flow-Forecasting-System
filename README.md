# Road Traffic Flow Forecasting System

An LSTM-based deep learning project that forecasts traffic volume from
historical traffic and weather data. The project includes a Streamlit
dashboard for analysis, model evaluation, and interactive prediction.

## What this project does

1. Loads and validates `traffic.csv`.
2. Extracts time features from `date_time`.
3. Adds weekend and rush-hour indicators.
4. Scales the data without using future test information.
5. Creates sequences of the previous 24 records.
6. Trains an LSTM model to predict the next traffic-volume value.
7. Evaluates the model using MAE, MSE, RMSE, R2, and MAPE.
8. Displays charts and predictions in the Streamlit dashboard.

`TIME_STEPS = 24` means the model uses the previous 24 records as context.
For hourly data, this represents the previous 24 hours.

## Project structure

```text
Road-Traffic-Flow-Forecasting-System/
├── main.py                 # Streamlit dashboard
├── pipeline.py             # Shared data preparation and sequence logic
├── lstm_model.py           # LSTM architecture and training
├── evaluate_model.py       # Test-set metrics
├── traffic.csv             # Historical dataset
├── traffic_lstm.keras      # Trained LSTM model
├── traffic_scaler.save     # Fitted feature scaler
├── requirements.txt        # Python dependencies
└── README.md               # This guide
```

The old duplicate scripts were removed. Data preparation now lives in
`pipeline.py`, so training, evaluation, and the dashboard use the same logic.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Internet access for the first dependency installation
- At least 4 GB RAM recommended for TensorFlow

## Windows setup

Open PowerShell in the repository root and run:

```powershell
cd "C:\Users\Helium\OneDrive\Desktop\GITHUB\Road-Traffic-Flow-Forecasting-System"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The activation step is optional. Commands can also be run with
`.venv\Scripts\python.exe` directly.

## Start the dashboard

From the project directory, with the virtual environment active:

```powershell
python -m streamlit run main.py
```

Open `http://localhost:8501` in a browser. Stop the dashboard with `Ctrl+C`.

## Retrain the LSTM

Retraining is only needed after changing the dataset or model code:

```powershell
python lstm_model.py
python evaluate_model.py
```

Training overwrites `traffic_lstm.keras` and `traffic_scaler.save`. The
dashboard automatically uses these saved artifacts.

## Evaluate the saved model

```powershell
python evaluate_model.py
```

The evaluation uses the final chronological 20% of the sequence data. For
regression, lower MAE, RMSE, and MAPE are better; higher R2 is better.

## Dashboard sections

- **Overview:** latest traffic, predicted traffic, status, recent trend, and data preview.
- **Traffic Analysis:** hourly, daily, and weather-related traffic patterns.
- **Predict Traffic:** enter current conditions and get an LSTM prediction.
- **Model Performance:** dynamic metrics, actual-vs-predicted values, and error chart.

## Input features

The LSTM uses:

- `traffic_volume`
- `temp`
- `rain_1h`
- `snow_1h`
- `clouds_all`
- `hour`
- `day_of_week`
- `is_weekend`
- `is_rush_hour`

## Troubleshooting

### `ModuleNotFoundError`

Activate `.venv` and reinstall:

```powershell
python -m pip install -r requirements.txt
```

### Model or scaler file not found

Run training from the project directory:

```powershell
python lstm_model.py
```

### Port already in use

Run Streamlit on another port:

```powershell
python -m streamlit run main.py --server.port 8502
```
