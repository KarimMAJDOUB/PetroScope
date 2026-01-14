import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pymysql
import logging
import sys
import os
import uuid


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.sql_config import sql_settings

from datetime import datetime
from SQL_connect_data import Call_data_sql

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

from scikeras.wrappers import KerasRegressor

logger = logging.getLogger(__name__)


def create_dataset(dataset, look_back):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, :])
        y.append(dataset[i+look_back, :])
    return np.array(X), np.array(y)

def build_model(units=64, learning_rate=0.001, look_back=20):
    model = Sequential()
    model.add(LSTM(units, input_shape=(look_back, 3)))
    model.add(Dense(3))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model

def LSTM_model():

    df = Call_data_sql("""
        SELECT DAYTIME,
            SUM(BORE_OIL_VOL) AS OIL_VOL,
            SUM(BORE_GAS_VOL) AS GAS_VOL,
            SUM(BORE_WAT_VOL) AS WAT_VOL
        FROM PWELL_DATA
        GROUP BY DAYTIME
    """)

    df["DAYTIME"] = pd.to_datetime(df["DAYTIME"])
    df = df.sort_values("DAYTIME").reset_index(drop=True)

    values = df[["OIL_VOL", "GAS_VOL", "WAT_VOL"]].values.astype("float32")

    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    look_back = 20

    X, y = create_dataset(values_scaled, look_back)

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    regressor = KerasRegressor(
        model=build_model,
        verbose=0
    )

    param_grid = {
        "model__units": [32, 64],
        "model__learning_rate": [0.001, 0.005],
        "batch_size": [16,32],
        "epochs": [30,50]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    grid = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=param_grid,
        n_iter=5,
        cv=tscv,
        scoring="neg_mean_squared_error",
        verbose=2,
        random_state=42
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    time_test = df["DAYTIME"].iloc[look_back + split : look_back + split + len(y_pred)]
    best_params = grid.best_params_
    best_score = (-grid.best_score_) 

    id_model=str(uuid.uuid4())

    df_best = pd.DataFrame([best_params])
    df_best["mse"] = best_score
    df_best["look_back"] = look_back  
    df_best["timestamp"] = pd.Timestamp.now()
    df_best["id_model"]=id_model  

    df_model = pd.DataFrame(
    data=y_pred_inv,
    columns=["OIL_VOL", "GAS_VOL", "WAT_VOL"]
    )

    df_model["DAYTIME"] = pd.to_datetime(time_test.values)
    df_model["id_model"] = id_model
    df_model["model_type"] = "LSTM"
    
    return df_best,df_model

def LSTM_param_load(df) -> None:
    """
    Loads data into a MySQL database using pymysql.
    """
    try:
        connection = pymysql.connect(
            user=sql_settings.user,
            password=sql_settings.password,
            database=sql_settings.database,
            cursorclass=sql_settings.cursorclass
        )
        logger.info(f"Connection established!")
    except pymysql.err.OperationalError as e:
        logger.error(f"Connection failed: {e}")
    

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS LSTM_PARAM (
                    id_model VARCHAR(36) PRIMARY KEY ,
                    timestamp DATETIME,
                    model__units INT,
                    model__learning_rate DECIMAL(10,10),
                    epochs INT,
                    batch_size INT,
                    mse DECIMAL(10,10),
                    look_back INT
                ); 
            """)

            insert_objects = """
                INSERT INTO LSTM_PARAM
                (id_model,
                    timestamp,
                    model__units  ,
                    model__learning_rate  ,
                    epochs  ,
                    batch_size  ,
                    mse  ,
                    look_back)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    timestamp = VALUES(timestamp),
                    model__units = VALUES(model__units),
                    model__learning_rate = VALUES(model__learning_rate),
                    epochs = VALUES(epochs),
                    batch_size = VALUES(batch_size),
                    mse = VALUES(mse),
                    look_back  = VALUES(look_back);
            """

            for _, row in df.iterrows():
                cursor.execute(
                    insert_objects,
                    (
                        row['id_model'],
                        row['timestamp'],
                        row['model__units'],
                        row['model__learning_rate'],
                        row['epochs'],
                        row['batch_size'],
                        row['mse'],
                        row['look_back'],
                    )
                )

        connection.commit()
    finally:
        logger.info("Connection closed")
        connection.close()




