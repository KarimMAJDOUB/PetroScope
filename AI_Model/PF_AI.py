import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from SQL_connect_data import Call_data_sql
from LSTM_AI import LSTM_model

def PF_LSTM(n_days):

    df_best, df_model, best_estimator, scaler = LSTM_model()

    model = best_estimator.model_

    look_back = int(df_best["look_back"].iloc[0])
    n_features = 3

    df=df_model

    values = df[["OIL_VOL", "GAS_VOL", "WAT_VOL"]].values.astype("float32")

    last_values = values[-look_back:]
    last_values_scaled = scaler.transform(last_values)

    current_seq = last_values_scaled.reshape(1, look_back, n_features)

    forecast_scaled = []

    for _ in range(n_days):
        next_step = model.predict(current_seq, verbose=0)
        forecast_scaled.append(next_step[0])

    current_seq = np.concatenate(
        (current_seq[:, 1:, :],
            next_step.reshape(1, 1, n_features)),
        axis=1
    )

    forecast_scaled = np.array(forecast_scaled)
    forecast = scaler.inverse_transform(forecast_scaled)

    last_date = df["DAYTIME"].iloc[-1]

    future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=n_days,
    freq="D"
    )

    df_forecast = pd.DataFrame(
    forecast,
    columns=["OIL_VOL", "GAS_VOL", "WAT_VOL"]
    )

    df_forecast["DAYTIME"] = future_dates
    df_forecast["id_model"] = df_best["id_model"].iloc[0]
    df_forecast["model_type"] = "LSTM_FORECAST"

    return df_forecast

