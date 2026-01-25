import numpy as np
import pandas as pd
import pymysql
import logging
import uuid
import joblib

from src.config.sql_config import sql_settings

from datetime import datetime
from src.data_access.sql_reader import call_data_sql

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

from scikeras.wrappers import KerasRegressor

logger = logging.getLogger(__name__)


def create_dataset(dataset, look_back):
    try:
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:i+look_back, :])
            y.append(dataset[i+look_back, :])
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"Erreur lors de la création du dataset : {e}")
        return None, None

def build_model(units=64, learning_rate=0.001, look_back=20):
<<<<<<< HEAD:src/ML/lstm.py
    model = Sequential([
        tf.keras.layers.Input(shape=(look_back, 3)),
        LSTM(units),
        Dense(3)
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model

def lstm_model():
=======
    try:
        model = Sequential()
        model.add(LSTM(units, input_shape=(look_back, 3)))
        model.add(Dense(3))
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mse"
        )
        return model
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle LSTM : {e}")
        return None

def LSTM_model():
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
        logger.error(f"Erreur récupération des données SQL : {e}")
        return None, None, None, None
>>>>>>> e44f567472de3b0582bcfd6c4e3a55e44ee5fa47:AI_Model/LSTM_AI.py

    if df is None or df.empty:
        logger.error("Erreur : aucun résultat récupéré depuis SQL.")
        return None, None, None, None

    try:
        df["DAYTIME"] = pd.to_datetime(df["DAYTIME"])
    except Exception as e:
        logger.error(f"Erreur conversion DAYTIME en datetime : {e}")
        return None, None, None, None

    df = df.sort_values("DAYTIME").reset_index(drop=True)

    try:
        values = df[["OIL_VOL", "GAS_VOL", "WAT_VOL"]].values.astype("float32")
    except KeyError as e:
        logger.error(f"Colonnes attendues absentes : {e}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"Erreur préparation des données : {e}")
        return None, None, None, None

    try:
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values)
    except Exception as e:
        logger.error(f"Erreur lors du scaling des données : {e}")
        return None, None, None, None

    look_back = 20

    try:
        X, y = create_dataset(values_scaled, look_back)
        if X is None or y is None:
            return None, None, None, None
    except Exception as e:
        logger.error(f"Erreur création du dataset LSTM : {e}")
        return None, None, None, None

    try:
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
    except Exception as e:
        logger.error(f"Erreur découpage train/test : {e}")
        return None, None, None, None

    try:
        regressor = KerasRegressor(model=build_model, verbose=0)
    except Exception as e:
        logger.error(f"Erreur création du KerasRegressor : {e}")
        return None, None, None, None

    param_grid = {
        "model__units": [32, 64],
        "model__learning_rate": [0.001, 0.005],
        "batch_size": [16,32],
        "epochs": [30,50]
    }

    tscv = TimeSeriesSplit(n_splits=3)

    try:
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
    except Exception as e:
        logger.error(f"Erreur lors du RandomizedSearchCV : {e}")
        return None, None, None, None

    try:
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test)
        time_test = df["DAYTIME"].iloc[look_back + split : look_back + split + len(y_pred)]
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction ou inverse scaling : {e}")
        return None, None, None, None

    try:
        best_params = grid.best_params_
        best_score = (-grid.best_score_) 
        id_model = str(uuid.uuid4())

        df_best = pd.DataFrame([best_params])
        df_best["mse"] = best_score
        df_best["look_back"] = look_back  
        df_best["timestamp"] = pd.Timestamp.now()
        df_best["id_model"]=id_model  

        df_model = pd.DataFrame(
            data_test=y_test_inv,
            data_pred=y_pred_inv,
            columns=["OIL_VOL_test", "GAS_VOL_test", "WAT_VOL_test","OIL_VOL_pred", "GAS_VOL_pred", "WAT_VOL_pred"]
        )

        df_model["DAYTIME"] = pd.to_datetime(time_test.values)
        df_model["id_model"] = id_model
        df_model["model_type"] = "LSTM"
    except Exception as e:
        logger.error(f"Erreur création des DataFrames de sortie : {e}")
        return None, None, None, None

<<<<<<< HEAD:src/ML/lstm.py
    os.makedirs("models", exist_ok=True)
    
    id_model = str(uuid.uuid4())
    
    # Sauvegarder le modèle et le scaler
    model_path = f"models/lstm_{id_model}.keras"
    scaler_path = f"models/scaler_{id_model}.pkl"
    
    best_model.model_.save(model_path)
    joblib.dump(scaler, scaler_path)

    df_best = pd.DataFrame([best_params])
    df_best["mse"] = best_score
    df_best["look_back"] = look_back  
    df_best["timestamp"] = pd.Timestamp.now()
    df_best["id_model"] = id_model
    df_best["model_path"] = model_path
    df_best["scaler_path"] = scaler_path

    df_model = pd.DataFrame(
        np.hstack([y_test_inv, y_pred_inv]),
        columns=["OIL_VOL_test", "GAS_VOL_test", "WAT_VOL_test","OIL_VOL_pred", "GAS_VOL_pred", "WAT_VOL_pred"])
    df_model["DAYTIME"] = pd.to_datetime(time_test.values)
    df_model["id_model"] = id_model
    df_model["model_type"] = "LSTM"
=======
    return df_best, df_model, best_model, scaler

>>>>>>> e44f567472de3b0582bcfd6c4e3a55e44ee5fa47:AI_Model/LSTM_AI.py

    LSTM_param_load(df_best)
    
    return df_model

def lstm_param_load(df) -> None:
    """
    Loads data into a MySQL database using pymysql.
    Messages d'erreur ajoutés pour le suivi de la connexion et des insertions.
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
        return
<<<<<<< HEAD:src/ML/lstm.py
=======
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la connexion MySQL : {e}")
        return
>>>>>>> e44f567472de3b0582bcfd6c4e3a55e44ee5fa47:AI_Model/LSTM_AI.py

    try:
        with connection.cursor() as cursor:
            # Créer la table avec les nouveaux champs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS LSTM_PARAM (
                    id_model VARCHAR(36) PRIMARY KEY,
                    timestamp DATETIME,
                    model__units INT,
                    model__learning_rate DECIMAL(10,10),
                    epochs INT,
                    batch_size INT,
                    mse DECIMAL(10,10),
                    look_back INT,
                    model_path VARCHAR(255),
                    scaler_path VARCHAR(255)
                ); 
            """)

            # Insertion avec les nouveaux champs
            insert_objects = """
                INSERT INTO LSTM_PARAM
                (id_model,
                    timestamp,
                    model__units,
                    model__learning_rate,
                    epochs,
                    batch_size,
                    mse,
                    look_back,
                    model_path,
                    scaler_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    timestamp = VALUES(timestamp),
                    model__units = VALUES(model__units),
                    model__learning_rate = VALUES(model__learning_rate),
                    epochs = VALUES(epochs),
                    batch_size = VALUES(batch_size),
                    mse = VALUES(mse),
                    look_back = VALUES(look_back),
                    model_path = VALUES(model_path),
                    scaler_path = VALUES(scaler_path);
            """

            for _, row in df.iterrows():
<<<<<<< HEAD:src/ML/lstm.py
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
                        row['model_path'],
                        row['scaler_path']
                    )
                )
=======
                try:
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
                            row['look_back']
                        )
                    )
                except Exception as e:
                    logger.error(f"Erreur insertion ligne {row['id_model']} : {e}")

>>>>>>> e44f567472de3b0582bcfd6c4e3a55e44ee5fa47:AI_Model/LSTM_AI.py
        connection.commit()
    except Exception as e:
        logger.error(f"Erreur lors de la création ou insertion dans la table LSTM_PARAM : {e}")
    finally:
        try:
            connection.close()
            logger.info("Connection closed")
        except Exception as e:
            logger.error(f"Erreur fermeture connexion MySQL : {e}")
