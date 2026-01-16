import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import uuid
import io
import torch
import joblib
import json
import logging
import pymysql
import sys
import os

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from skorch import NeuralNetRegressor


from SQL_connect_data import Call_data_sql
from LSTM_AI import LSTM_model
from TRANSFORMER_AI import transformer_model
from Model_Load import PF_Load

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

def predict_future_from_saved_model(id_model, n_days):
    """
    Charge un modèle sauvegardé et prédit les n prochains jours
    """
    # Récupérer les paramètres et le modèle depuis la DB
    df_params = Call_data_sql(f"""
        SELECT * FROM TRANSFORMER_PARAM 
        WHERE id_model = '{id_model}'
    """)
    
    if df_params.empty:
        raise ValueError(f"Modèle {id_model} non trouvé")
    
    row = df_params.iloc[0]
    
    # Charger le scaler
    scaler = joblib.load(io.BytesIO(row['scaler']))
    
    # Recréer le modèle avec les mêmes paramètres
    net = NeuralNetRegressor(
        module=transformer_model,
        module__input_dim=int(row['input_dim']),
        module__d_model=int(row['module__d_model']),
        module__nhead=int(row['module__nhead']),
        module__num_layers=int(row['module__num_layers']),
        module__dropout=float(row['module__dropout']),
        max_epochs=10,
        lr=float(row['lr']),
        batch_size=int(row['batch_size']),
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        device="cpu"
    )
    
    # Charger les poids
    net.initialize()
    net.module_.load_state_dict(torch.load(io.BytesIO(row['model'])))
    
    # Récupérer les dernières données
    look_back = int(row['look_back'])
    df = Call_data_sql(f"""
        SELECT DAYTIME,
            SUM(BORE_OIL_VOL) AS OIL_VOL,
            SUM(BORE_GAS_VOL) AS GAS_VOL,
            SUM(BORE_WAT_VOL) AS WAT_VOL
        FROM PWELL_DATA
        GROUP BY DAYTIME
        ORDER BY DAYTIME ASC
    """)
    
    # Prendre seulement les derniers look_back jours
    df = df.tail(look_back).reset_index(drop=True)
    df["DAYTIME"] = pd.to_datetime(df["DAYTIME"])
    
    values = df[["OIL_VOL", "GAS_VOL", "WAT_VOL"]].values.astype("float32")
    
    # Normaliser
    last_sequence = scaler.transform(values)
    
    # Prédire le futur
    predictions = []
    current_sequence = last_sequence.copy()
    input_dim = int(row['input_dim'])
    
    for _ in range(n_days):
        pred = net.predict(current_sequence.reshape(1, look_back, input_dim))
        predictions.append(pred[0])
        current_sequence = np.vstack([current_sequence[1:], pred[0]])
    
    predictions = np.array(predictions)
    predictions_inv = scaler.inverse_transform(predictions)
    
    # Créer les dates futures
    last_date = df["DAYTIME"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    
    # DataFrame résultat
    df_future = pd.DataFrame({
        "DAYTIME": future_dates,
        "OIL_VOL_pred": predictions_inv[:, 0],
        "GAS_VOL_pred": predictions_inv[:, 1],
        "WAT_VOL_pred": predictions_inv[:, 2],
        "id_model": id_model,
        "model_type": row['model_type']
    })
    
    return df_future

PF_Load(predict_future_from_saved_model("5dca9ac6-19f4-4f8f-bdac-181d455f5555", n_days=90))

