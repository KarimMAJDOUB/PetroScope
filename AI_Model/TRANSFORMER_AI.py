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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.sql_config import sql_settings
from Model_Load import model_load

from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_access.sql_reader import call_data_sql

logger = logging.getLogger(__name__)


def create_dataset(dataset, look_back):
    try:
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:i+look_back, :])
            y.append(dataset[i+look_back, :])
        return np.array(X), np.array(y)
    except Exception as e:
        logger.exception(f"[ERREUR_CREATION_DATASET] Échec de la création des séquences: {e}")
        raise


def transformer_model(input_dim=3, d_model=64, nhead=4, num_layers=2, dropout=0.1):
    try:
        if d_model % nhead != 0:
            raise ValueError("[ERREUR_PARAM_MODEL] d_model doit être divisible par nhead")

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
                try:
                    x = self.input_proj(x)
                    x = self.transformer(x)
                    return self.output_layer(x[:, -1])
                except Exception as e:
                    logger.exception(f"[ERREUR_FORWARD_MODEL] Échec du forward pass: {e}")
                    raise

        return Model()
    except Exception as e:
        logger.exception(f"[ERREUR_CREATION_MODEL] Impossible de créer le modèle Transformer: {e}")
        raise


def transomer_AI():
    try:
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

        param_distributions = {
            "module__d_model": [64],
            "module__nhead": [4],
            "module__num_layers": [2],
            "module__dropout": [0.1],
            "lr": [1e-4],
            "batch_size": [16]
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

        try:
            df = call_data_sql("""
                SELECT DAYTIME,
                    SUM(BORE_OIL_VOL) AS OIL_VOL,
                    SUM(BORE_GAS_VOL) AS GAS_VOL,
                    SUM(BORE_WAT_VOL) AS WAT_VOL
                FROM PWELL_DATA
                GROUP BY DAYTIME
            """)
        except Exception as e:
            logger.exception(f"[ERREUR_SQL] Échec de la récupération des données: {e}")
            raise

        df["DAYTIME"] = pd.to_datetime(df["DAYTIME"])
        df = df.sort_values("DAYTIME").reset_index(drop=True)

        values = df[["OIL_VOL", "GAS_VOL", "WAT_VOL"]].values.astype("float32")

        try:
            scaler = StandardScaler()
            values = scaler.fit_transform(values)
        except Exception as e:
            logger.exception(f"[ERREUR_SCALER] Échec de la normalisation des données: {e}")
            raise

        window_size = 30
        try:
            X, y = create_dataset(values, window_size)
        except Exception as e:
            logger.exception(f"[ERREUR_CREATION_SEQUENCE] Échec de la création des séquences X/y: {e}")
            raise

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        dates = df["DAYTIME"].iloc[window_size:].reset_index(drop=True)
        dates_train, dates_test = dates[:split], dates[split:]

        try:
            search.fit(X_train, y_train)
        except Exception as e:
            logger.exception(f"[ERREUR_TRAINING] Échec du training du modèle: {e}")
            raise

        best_model = search.best_estimator_
        try:
            y_pred = best_model.predict(X_test)
        except Exception as e:
            logger.exception(f"[ERREUR_PREDICTION] Échec de la prédiction sur X_test: {e}")
            raise

        try:
            y_pred_inv = scaler.inverse_transform(y_pred)
            y_test_inv = scaler.inverse_transform(y_test)
        except Exception as e:
            logger.exception(f"[ERREUR_INV_SCALER] Échec de la dénormalisation des prédictions: {e}")
            raise

        id_model = str(uuid.uuid4())

        try:
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
        except Exception as e:
            logger.exception(f"[ERREUR_CREATION_DF_MODEL] Impossible de créer le DataFrame df_model: {e}")
            raise

        try:
            buffer = io.BytesIO()
            torch.save(best_model.module_.state_dict(), buffer)
            model_bytes = buffer.getvalue()

            buffer = io.BytesIO()
            joblib.dump(scaler, buffer)
            scaler_bytes = buffer.getvalue()
        except Exception as e:
            logger.exception(f"[ERREUR_SERIALIZATION] Échec de la sauvegarde du modèle ou scaler: {e}")
            raise

        best_params = search.best_params_
        best_score = search.best_score_
        params_json = json.dumps(best_params)
        metadata = {"mse": best_score, "look_back": window_size, "input_dim": 3, "model_type": "TRANSFORMER"}
        metadata_json = json.dumps(metadata)

        try:
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
        except Exception as e:
            logger.exception(f"[ERREUR_CREATION_DF_STORE] Impossible de créer df_model_store: {e}")
            raise

        return df_model_store, df_model

    except Exception as e:
        logger.exception(f"[ERREUR_TRANSFORMER_AI] Échec complet de la fonction transomer_AI: {e}")
        raise


def TRANSFORMER_param_load(df) -> None:
    try:
        connection = pymysql.connect(
            user=sql_settings.user,
            password=sql_settings.password,
            database=sql_settings.database,
            cursorclass=sql_settings.cursorclass
        )
        logger.info(f"Connexion à la base établie")
    except pymysql.err.OperationalError as e:
        logger.error(f"[ERREUR_CONNEXION_BDD] Échec de connexion: {e}")
        raise

    try:
        with connection.cursor() as cursor:
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
                    model LONGBLOB,
                    scaler LONGBLOB
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
                model,
                scaler)
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
                    model = VALUES(model),
                    scaler = VALUES(scaler);
            """

            for _, row in df.iterrows():
                try:
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
                            row['model'],
                            row['scaler']
                        )
                    )
                except Exception as e:
                    logger.exception(f"[ERREUR_INSERTION_BDD] Échec insertion ligne id_model={row['id_model']}: {e}")
                    raise

        connection.commit()
    except Exception as e:
        logger.exception(f"[ERREUR_EXECUTION_SQL] Échec des opérations SQL: {e}")
        raise
    finally:
        logger.info("Connexion fermée")
        connection.close()


# Exécution principale
try:
    df_model_store, df_model = transomer_AI()
    TRANSFORMER_param_load(df_model_store)
    model_load(df_model)
except Exception as e:
    logger.exception(f"[ERREUR_PIPELINE] Échec complet du pipeline: {e}")
