
import pandas as pd
import numpy as np
import logging
import sys
import os
import uuid
import pymysql
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from src.config.sql_config import sql_settings
except ImportError:
    pass 

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 1. MODÈLE DE PRÉDICTION
# -------------------------------------------------------------------------
def RF_model():
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

    df_params = pd.DataFrame([{
        "id_model": id_model,
        "timestamp": pd.Timestamp.now(),
        "n_estimators": 100,
        "r2_score": r2,
        "mse": mse,
        "algo": "RF_MultiOutput_J+1"
    }])

    df_predictions = pd.DataFrame(y_pred, columns=['OIL_VOL', 'GAS_VOL', 'WAT_VOL'])
    
    df_predictions['BORE_OIL_VOL'] = y_test['NEXT_OIL'].values
    df_predictions['BORE_GAS_VOL'] = y_test['NEXT_GAS'].values
    df_predictions['BORE_WAT_VOL'] = y_test['NEXT_WAT'].values
    
    df_predictions["DAYTIME"] = dates_test.values + pd.Timedelta(days=1)
    df_predictions["id_model"] = id_model
    df_predictions["model_type"] = "RF"

    return df_params, df_predictions

# -------------------------------------------------------------------------
# 2. SAUVEGARDES
# -------------------------------------------------------------------------
def RF_param_load(df):
    try:
        connection = pymysql.connect(
            user=sql_settings.user, password=sql_settings.password,
            database=sql_settings.database, cursorclass=sql_settings.cursorclass
        )
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS RF_PARAM (
                    id_model VARCHAR(36) PRIMARY KEY,
                    timestamp DATETIME, n_estimators INT,
                    r2_score DECIMAL(10,5), mse DECIMAL(20,5)
                );
             """)
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO RF_PARAM (id_model, timestamp, n_estimators, r2_score, mse)
                    VALUES (%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE r2_score=VALUES(r2_score)
                """, (row['id_model'], row['timestamp'], row['n_estimators'], row['r2_score'], row['mse']))
        connection.commit()
        connection.close()
    except Exception as e:
        logger.error(f"Erreur Param: {e}")

def model_load(df):
    try:
        connection = pymysql.connect(
            user=sql_settings.user, password=sql_settings.password,
            database=sql_settings.database, cursorclass=sql_settings.cursorclass
        )
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS MODEL_AI (
                    id_model VARCHAR(36), model_type VARCHAR(32),
                    DAYTIME DATE, OIL_VOL DOUBLE, GAS_VOL DOUBLE, WAT_VOL DOUBLE,
                    PRIMARY KEY(id_model, DAYTIME)
                );
             """)
            for _, row in df.iterrows():
                d = pd.to_datetime(row['DAYTIME']).date()
                cursor.execute("""
                    INSERT INTO MODEL_AI (id_model, model_type, DAYTIME, OIL_VOL, GAS_VOL, WAT_VOL)
                    VALUES (%s, %s, %s, %s, %s, %s) 
                    ON DUPLICATE KEY UPDATE 
                        OIL_VOL = VALUES(OIL_VOL),
                        GAS_VOL = VALUES(GAS_VOL),
                        WAT_VOL = VALUES(WAT_VOL)
                """, (row['id_model'], row['model_type'], d, row['OIL_VOL'], row['GAS_VOL'], row['WAT_VOL']))
        connection.commit()
        connection.close()
        logger.info("Résultats sauvegardés dans MODEL_AI.")
    except Exception as e:
        logger.error(f"Erreur Load: {e}")

if __name__ == "__main__":
    df_params, df_preds = RF_model()
    if df_params is not None:
        RF_param_load(df_params)
        model_load(df_preds)