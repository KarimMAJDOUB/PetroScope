# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 15:25:54 2026

@author: RHEDDAD
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
import uuid
import pymysql

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# CONFIGURATION LOGGING
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# CONFIG SQL
# -----------------------------
try:
    from src.config.sql_config import sql_settings
except ImportError:
    logger.error("Impossible d'importer sql_settings. Vérifie src/config/sql_config.py")
    exit()

# -----------------------------
# DOSSIER POUR LES PLOTS
# -----------------------------
plots_folder = "AI/AI_plots"
os.makedirs(plots_folder, exist_ok=True)

# -----------------------------
# FONCTION MODÈLE
# -----------------------------
def RF_model():
    logger.info("Démarrage RF Multi-Output : Huile, Gaz, Eau à J+1")

    # --- A. Connexion SQL
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
        logger.info("Données récupérées depuis SQL")
    except Exception as e:
        logger.error(f"Erreur SQL : {e}")
        return None, None

    if df.empty:
        logger.error("La base de données est vide.")
        return None, None

    # --- B. Préparation des données
    df['DAYTIME'] = pd.to_datetime(df.get('DAYTIME', df.get('DATEPRD')))
    df = df.sort_values('DAYTIME')

    features_cols = [
        'ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
        'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 
        'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE',
        'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL'
    ]

    for col in features_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=features_cols + ['DAYTIME'])

    # --- C. Création des cibles J+1
    df['NEXT_OIL'] = df['BORE_OIL_VOL'].shift(-1)
    df['NEXT_GAS'] = df['BORE_GAS_VOL'].shift(-1)
    df['NEXT_WAT'] = df['BORE_WAT_VOL'].shift(-1)
    df_model = df.dropna(subset=['NEXT_OIL', 'NEXT_GAS', 'NEXT_WAT'])

    X = df_model[features_cols]
    y = df_model[['NEXT_OIL', 'NEXT_GAS', 'NEXT_WAT']]
    dates_test = df_model['DAYTIME']

    # --- D. Split temporel 80/20
    split_index = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    dates_test = dates_test.iloc[split_index:]

    # --- E. Entraînement du modèle
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    # --- F. Prédictions
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Prédictions J+1 terminées. R2 Global : {r2:.4f}, MSE : {mse:.4f}")

    # --- G. Génération des plots
    targets = ['OIL', 'GAS', 'WAT']
    for i, col in enumerate(targets):
        plt.figure(figsize=(10,6))
        plt.plot(dates_test, y_test.iloc[:,i], label=f"Réel {col}", color='blue')
        plt.plot(dates_test, y_pred[:,i], label=f"Prédit {col}", color='red')
        plt.xlabel("Date")
        plt.ylabel(f"Volume {col}")
        plt.title(f"Prédictions {col} J+1")
        plt.legend()
        plot_path = os.path.join(plots_folder, f"predictions_{col.lower()}.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Plot sauvegardé : {plot_path}")

    return y_test, y_pred, dates_test

# -----------------------------
# EXECUTION
# -----------------------------
if __name__ == "__main__":
    y_test, y_pred, dates_test = RF_model()
    if y_test is not None:
        print("Script terminé. Les plots sont sauvegardés")
