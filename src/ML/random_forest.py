# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 23:02:07 2026

@author: ASUS Vivibook

Modifier by JPJanssen : support the insertion in the Database with main_AI.py
"""

import pandas as pd
import numpy as np
import logging
import uuid
import pymysql
import joblib

try:
    from src.config.sql_config import sql_settings
except ImportError:
    pass 

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

from SQL_connect_data import Call_data_sql


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 1. MODÈLE DE PRÉDICTION
# -------------------------------------------------------------------------
def random_forest_model():
    logger.info("Démarrage RF (Multi-Output : Huile, Gaz, Eau à J+1)...")

    # A. Connexion
    try:
        connection = pymysql.connect(
            user=sql_settings.user,
            password=sql_settings.password,
            database=sql_settings.database,
            cursorclass=sql_settings.cursorclass
        )
        query = "SELECT * FROM PWELL_DATA WHERE ON_STREAM_HRS > 0"
        df = pd.read_sql(query, connection)
        connection.close()

        df=Call_data_sql('SELECT * FROM PWELL_DATA WHERE ON_STREAM_HRS > 0')
    except Exception as e:
        logger.error(f"Erreur SQL : {e}")
        return None, None

    if df.empty:
        logger.error("La base de données est vide.")
        return None, None

  # B. Préparation
    if 'DAYTIME' in df.columns:
        df['DAYTIME'] = pd.to_datetime(df['DAYTIME'])
    else:
        df['DAYTIME'] = pd.to_datetime(df['DATEPRD'])
        
    df = df.sort_values('DAYTIME')

    # --- 1. DÉFINITION DES FEATURES (ENTRÉES) ---
    features_cols = [
        'ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
        'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 
        'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE',
        'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL' # <--- Les infos du collègue sont ici
    ]

    # Conversion numérique 
    for col in features_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # On supprime les lignes qui ont des trous dans les données d'aujourd'hui
    df = df.dropna(subset=features_cols + ['DAYTIME'])

    # --- 2. CRÉATION DES CIBLES DU FUTUR (SORTIES J+1) ---
    df['NEXT_OIL'] = df['BORE_OIL_VOL'].shift(-1)
    df['NEXT_GAS'] = df['BORE_GAS_VOL'].shift(-1)
    df['NEXT_WAT'] = df['BORE_WAT_VOL'].shift(-1)
    
    df_model = df.dropna(subset=['NEXT_OIL', 'NEXT_GAS', 'NEXT_WAT'])

    # --- 3. DÉFINITION X et y ---
    X = df_model[features_cols]                       # Tout ce qu'on sait aujourd'hui
    y = df_model[['NEXT_OIL', 'NEXT_GAS', 'NEXT_WAT']] # Ce qui arrive demain
    dates = df_model['DAYTIME']

    # D. Split Temporel (80% passé / 20% futur)
    split_index = int(len(df_model) * 0.8)

    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]
    dates_test = dates.iloc[split_index:]

    # E. Entraînement (MultiOutputRegressor pour gérer 3 cibles)
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    # F. Prédictions
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Prédiction J+1 terminée. R2 Global : {r2:.4f}")

    # G. Préparation Résultats
    id_model = str(uuid.uuid4())

    model_path = f"models/rf_{id_model}.pkl"
    joblib.dump(model, model_path)

    # H. Préparation des paramètres
    df_params = pd.DataFrame([{
        "id_model": id_model,
        "timestamp": pd.Timestamp.now(),
        "n_estimators": 100,
        "random_state": 42,
        "r2_score": r2,
        "mse": mse,
        "algo": "RF_MultiOutput_J+1",
        "model_path": model_path,
        "features": ",".join(features_cols)
    }])

    df_predictions = pd.DataFrame(
        np.hstack([y_test.values, y_pred]),
        columns=["OIL_VOL_test", "GAS_VOL_test", "WAT_VOL_test","OIL_VOL_pred", "GAS_VOL_pred", "WAT_VOL_pred"])
    df_predictions["DAYTIME"] = pd.to_datetime(dates_test.values)
    df_predictions["id_model"] = id_model
    df_predictions["model_type"] = "RF"

    RF_param_load(df_params)
   
    return df_predictions

<<<<<<< HEAD:src/ML/random_forest.py
# -------------------------------------------------------------------------
# 2. SAUVEGARDES
# -------------------------------------------------------------------------
def random_forest_param_load(df):
=======
def RF_param_load(df):
    """
    Loads Random Forest parameters into a MySQL database using pymysql.
    """
>>>>>>> 8591ab6be55cb0696defaf4fb4031201ac06fd61:AI_Model/RF_AI.py
    try:
        connection = pymysql.connect(
            user=sql_settings.user,
            password=sql_settings.password,
            database=sql_settings.database,
            cursorclass=sql_settings.cursorclass
        )
        logger.info("Connection established!")
        
        with connection.cursor() as cursor:
            # Créer la table avec les nouveaux champs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS RF_PARAM (
                    id_model VARCHAR(36) PRIMARY KEY,
                    timestamp DATETIME,
                    n_estimators INT,
                    random_state INT,
                    r2_score DECIMAL(10,5),
                    mse DECIMAL(20,5),
                    algo VARCHAR(50),
                    model_path VARCHAR(255),
                    features TEXT
                );
            """)
            
            # Insertion avec les nouveaux champs
            insert_query = """
                INSERT INTO RF_PARAM 
                (id_model, timestamp, n_estimators, random_state, r2_score, mse, algo, model_path, features)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) 
                ON DUPLICATE KEY UPDATE 
                    timestamp = VALUES(timestamp),
                    r2_score = VALUES(r2_score),
                    mse = VALUES(mse),
                    model_path = VALUES(model_path),
                    features = VALUES(features)
            """
            
            for _, row in df.iterrows():
                cursor.execute(
                    insert_query,
                    (
                        row['id_model'],
                        row['timestamp'],
                        row['n_estimators'],
                        row['random_state'],
                        row['r2_score'],
                        row['mse'],
                        row['algo'],
                        row['model_path'],
                        row['features']
                    )
                )
        
        connection.commit()
        logger.info(f"Model parameters saved successfully with id: {df['id_model'].values[0]}")
        
    except Exception as e:
        logger.error(f"Erreur Param: {e}")
        if 'connection' in locals():
            connection.rollback()
        raise
    finally:
        if 'connection' in locals():
            connection.close()
            logger.info("Connection closed")

if __name__ == "__main__":
    df_params, df_preds = random_forest_model()
    if df_params is not None:
<<<<<<< HEAD:src/ML/random_forest.py
        random_forest_param_load(df_params)
        model_load(df_preds)
=======
        RF_param_load(df_params)
>>>>>>> 8591ab6be55cb0696defaf4fb4031201ac06fd61:AI_Model/RF_AI.py
