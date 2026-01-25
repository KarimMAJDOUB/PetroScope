import logging
import uuid
import io
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import pymysql
<<<<<<< HEAD:src/ML/transformer.py
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
=======
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.sql_config import sql_settings

from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.model_selection import train_test_split
>>>>>>> 8591ab6be55cb0696defaf4fb4031201ac06fd61:AI_Model/TRANSFORMER_AI.py
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor

# Imports internes
from src.config.sql_config import sql_settings
from src.data_access.sql_reader import call_data_sql
from src.ml.model_load import model_load
from src.ml.lstm import lstm_model

logger = logging.getLogger(__name__)



def create_dataset(dataset, look_back):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i+look_back, :])
        y.append(dataset[i+look_back, :])
    return np.array(X), np.array(y)

def transformer_model(input_dim=3, d_model=64, nhead=4, num_layers=2, dropout=0.1):

    if d_model % nhead != 0:
        raise ValueError("d_model doit être divisible par nhead")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.input_proj = nn.Linear(input_dim, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                batch_first=True
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )

            self.output_layer = nn.Linear(d_model, input_dim)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.transformer(x)
            return self.output_layer(x[:, -1])

    return Model()

<<<<<<< HEAD:src/ML/transformer.py
def transformer():
    net=NeuralNetRegressor(
        module=transformer_model, 
        module__input_dim=3, 
        max_epochs=10, lr=1e-4, 
        batch_size=32, 
        optimizer=torch.optim.Adam, 
        criterion=nn.MSELoss, 
        device ="cpu"
    )

    # param_distributions = {
    #     "module__d_model": [32, 64],
    #     "module__nhead": [2, 4],
    #     "module__num_layers": [1, 2],
    #     "module__dropout": [0.0, 0.1, 0.2],
    #     "lr": [1e-4, 3e-4, 1e-3],
    #     "batch_size": [16, 32, 64]
    # }
=======
def transformer_AI():
    net = NeuralNetRegressor(
        module=transformer_model, 
        module__input_dim=3, 
        max_epochs=10, 
        lr=1e-4, 
        batch_size=32, 
        optimizer=torch.optim.Adam, 
        criterion=nn.MSELoss, 
        device="cpu"
    )
>>>>>>> 8591ab6be55cb0696defaf4fb4031201ac06fd61:AI_Model/TRANSFORMER_AI.py

    param_distributions = {
        "module__d_model": [32, 64],
        "module__nhead": [2, 4],
        "module__num_layers": [1, 2],
        "module__dropout": [0.0, 0.1, 0.2],
        "lr": [1e-4, 3e-4, 1e-3],
        "batch_size": [16, 32, 64]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    search = RandomizedSearchCV(
        estimator=net,
        param_distributions=param_distributions,
        n_iter=15,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=2,
        refit=True
    )

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

    scaler = StandardScaler()
    values = scaler.fit_transform(values)

    window_size = 30

    X, y = create_dataset(values, window_size)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    dates = df["DAYTIME"].iloc[window_size:].reset_index(drop=True)
    dates_train, dates_test = dates[:split], dates[split:]

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)

    # Générer l'ID unique du modèle
    id_model = str(uuid.uuid4())

    # Créer le dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)

    # Sauvegarder le modèle et le scaler dans des fichiers
    model_path = f"models/transformer_{id_model}.pth"
    scaler_path = f"models/scaler_{id_model}.pkl"

    torch.save(best_model.module_.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    df_model = pd.DataFrame({
        "OIL_VOL_test": y_test_inv[:, 0],
        "GAS_VOL_test": y_test_inv[:, 1],
        "WAT_VOL_test": y_test_inv[:, 2],
        "OIL_VOL_pred": y_pred_inv[:, 0],
        "GAS_VOL_pred": y_pred_inv[:, 1],
        "WAT_VOL_pred": y_pred_inv[:, 2]
    })

    df_model["DAYTIME"] = dates_test.values
    df_model["id_model"] = id_model
    df_model["model_type"] = "TRANSFORMER"

    best_params = search.best_params_
    best_score = -search.best_score_  # Convertir en valeur positive

    # Préparer df_model_store avec les chemins de fichiers
    df_model_store = pd.DataFrame([{
        "id_model": id_model,
        "created_at": pd.Timestamp.now(),
        "model_type": "TRANSFORMER",
        "mse": best_score,
        "look_back": window_size,
        "input_dim": 3,
<<<<<<< HEAD:src/ML/transformer.py
        "model_type": "TRANSFORMER"
    }

    metadata_json = json.dumps(metadata)

    df_model_store = pd.DataFrame([{
        "id_model": id_model,
        "model_blob": model_bytes,
        "scaler_blob": scaler_bytes,
        "best_params": params_json,
        "metadata": metadata_json,
        "created_at": pd.Timestamp.now()
    }])

    df_model_store = pd.DataFrame([{
        "id_model": id_model,
        "created_at": pd.Timestamp.now(),
        "model_type": "TRANSFORMER",
        "mse": best_score,
        "look_back": window_size,
        "input_dim": 3,
        **best_params,
        "model": model_bytes,
        "scaler": scaler_bytes
    }])

    print(df_model_store.columns)

    return df_model_store, df_model

=======
        "module__num_layers": best_params.get("module__num_layers"),
        "module__nhead": best_params.get("module__nhead"),
        "module__dropout": best_params.get("module__dropout"),
        "module__d_model": best_params.get("module__d_model"),
        "lr": best_params.get("lr"),
        "batch_size": best_params.get("batch_size"),
        "model_path": model_path,
        "scaler_path": scaler_path
    }])

    TRANSFORMER_param_load(df_model_store)
>>>>>>> 8591ab6be55cb0696defaf4fb4031201ac06fd61:AI_Model/TRANSFORMER_AI.py

    return df_model


def transformer_param_load(df) -> None:
    """
    Loads Transformer parameters into a MySQL database using pymysql.
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
        raise

    try:
        with connection.cursor() as cursor:
            # Créer la table avec model_path et scaler_path au lieu de BLOB
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS TRANSFORMER_PARAM (
                    id_model VARCHAR(36) PRIMARY KEY,
                    created_at DATETIME,
                    model_type VARCHAR(50),
                    mse FLOAT,
                    look_back INT,
                    input_dim INT,
                    module__num_layers INT,
                    module__nhead INT,
                    module__dropout FLOAT,
                    module__d_model INT,
                    lr FLOAT,
                    batch_size INT,
                    model_path VARCHAR(255),
                    scaler_path VARCHAR(255)
                );
            """)
            
            insert_objects = """
                INSERT INTO TRANSFORMER_PARAM
                (id_model,
                created_at,
                model_type,
                mse,
                look_back,
                input_dim,
                module__num_layers,
                module__nhead,
                module__dropout,
                module__d_model,
                lr,
                batch_size,
                model_path,
                scaler_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                created_at = VALUES(created_at),
                model_type = VALUES(model_type),
                mse = VALUES(mse),
                look_back = VALUES(look_back),
                input_dim = VALUES(input_dim),
                module__num_layers = VALUES(module__num_layers),
                module__nhead = VALUES(module__nhead),
                module__dropout = VALUES(module__dropout),
                module__d_model = VALUES(module__d_model),
                lr = VALUES(lr),
                batch_size = VALUES(batch_size),
                model_path = VALUES(model_path),
                scaler_path = VALUES(scaler_path);
            """

            for _, row in df.iterrows():
                cursor.execute(
                    insert_objects,
                    (
                        row['id_model'],
                        row['created_at'],
                        row['model_type'],
                        row['mse'],
                        row['look_back'],
                        row['input_dim'],
                        row['module__num_layers'],
                        row['module__nhead'],
                        row['module__dropout'],
                        row['module__d_model'],
                        row['lr'],
                        row['batch_size'],
                        row['model_path'],
                        row['scaler_path']
                    )
                )

        connection.commit()
        logger.info(f"Model parameters saved successfully with id: {df['id_model'].values[0]}")
    except Exception as e:
        logger.error(f"Error saving parameters: {e}")
        connection.rollback()
        raise
    finally:
        logger.info("Connection closed")
        connection.close()
<<<<<<< HEAD:src/ML/transformer.py



df_model_store, df_model = transformer()

transformer_param_load(df_model_store)
model_load(df_model)


=======
>>>>>>> 8591ab6be55cb0696defaf4fb4031201ac06fd61:AI_Model/TRANSFORMER_AI.py
