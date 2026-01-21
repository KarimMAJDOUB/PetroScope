
# -*- coding: utf-8 -*-
"""
AI_Model/LR.py
Régression linéaire multi-sorties (J+1) avec split temporel 80/20.
- Tente d'abord de lire depuis MySQL (PWELL_DATA)
- Si la connexion échoue, lit automatiquement le CSV le plus récent dans Data/Data_Ingested
- Sauvegarde dans la DB si disponible, sinon en CSV (AI_Model/outputs)
"""

import os
import uuid
import glob
import logging
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

# sklearn / pymysql (assurez-vous qu'ils sont installés)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

try:
    import pymysql
    HAS_PYMYSQL = True
except Exception:
    HAS_PYMYSQL = False

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LR_AI")

# -----------------------------------------------------------------------------
# 0) .env : charge en priorité celui de la racine du projet (…/PetroScope/.env),
#            sinon AI_Model/.env. Ne dépend d’aucun autre module.
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()          # .../PetroScope/AI_Model/LR.py
AI_MODEL_DIR = THIS_FILE.parent               # .../PetroScope/AI_Model
PROJECT_ROOT = AI_MODEL_DIR.parent            # .../PetroScope
DATA_INGESTED_DIR = PROJECT_ROOT / "Data" / "Data_Ingested"
OUTPUTS_DIR = AI_MODEL_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

ENV_CANDIDATES = [
    PROJECT_ROOT / ".env",        # ← ton souhait "une fois pour toute"
    AI_MODEL_DIR / ".env",        # fallback
]

def _load_env():
    loaded = False
    try:
        from dotenv import load_dotenv
        for p in ENV_CANDIDATES:
            if p.exists():
                load_dotenv(p, override=False)
                print(f"[LR.py] .env chargé depuis : {p}")
                loaded = True
                break
    except Exception:
        pass

    if not loaded:
        # Fallback manuel
        for p in ENV_CANDIDATES:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
                print(f"[LR.py] .env (manuel) chargé depuis : {p}")
                loaded = True
                break

    if not loaded:
        print("[LR.py] ⚠️ Aucun .env trouvé ; on utilisera les variables d'environnement système si elles existent.")

_load_env()

# -----------------------------------------------------------------------------
# 1) Connexion DB (si possible)
# -----------------------------------------------------------------------------
def try_connect_db():
    """Retourne une connexion PyMySQL ou None si indisponible."""
    if not HAS_PYMYSQL:
        logger.warning("PyMySQL indisponible : mode fichiers.")
        return None

    host = os.getenv("DB_HOST", "127.0.0.1")
    port = int(os.getenv("DB_PORT", "3306"))
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "")
    database = os.getenv("DB_NAME", "PetroScope")
    cursorclass_name = os.getenv("SQL_CURSORCLASS", "DictCursor")
    cursorclass = getattr(pymysql.cursors, cursorclass_name, pymysql.cursors.DictCursor)

    try:
        conn = pymysql.connect(
            host=host, port=port, user=user, password=password,
            database=database, cursorclass=cursorclass
        )
        logger.info(f"Connexion DB OK ({host}:{port}/{database})")
        return conn
    except Exception as e:
        logger.warning(f"Connexion DB indisponible ({host}:{port}/{database}) → {e}")
        return None

# -----------------------------------------------------------------------------
# 2) Chargement données : DB → fallback fichiers
# -----------------------------------------------------------------------------
REQUIRED_FEATURES = [
    "ON_STREAM_HRS", "AVG_DOWNHOLE_PRESSURE", "AVG_DOWNHOLE_TEMPERATURE",
    "AVG_DP_TUBING", "AVG_ANNULUS_PRESS", "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P", "AVG_WHT_P", "DP_CHOKE_SIZE",
    "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"
]

def load_from_db():
    conn = try_connect_db()
    if conn is None:
        return None
    try:
        df = pd.read_sql("SELECT * FROM PWELL_DATA WHERE ON_STREAM_HRS > 0", conn)
        return df
    finally:
        try:
            conn.close()
        except Exception:
            pass

def latest_csv_in_data_ingested():
    # On prend le CSV le plus récent (ex : volve_rate_*.csv)
    candidates = sorted(glob.glob(str(DATA_INGESTED_DIR / "*.csv")), key=os.path.getmtime, reverse=True)
    return Path(candidates[0]) if candidates else None

def load_from_files():
    csv_path = latest_csv_in_data_ingested()
    if not csv_path:
        raise FileNotFoundError(f"Aucun CSV trouvé dans {DATA_INGESTED_DIR}")

    # Tentative lecture auto (détecte séparateur)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep=None, engine="python")

    print(f"[LR.py] Données chargées depuis le fichier : {csv_path.name}")
    return df

def load_pwell_data():
    df = load_from_db()
    if df is not None and not df.empty:
        print("[LR.py] Source utilisée : Base MySQL")
        return df

    df = load_from_files()
    print("[LR.py] Source utilisée : Fichier CSV (fallback)")
    return df

# -----------------------------------------------------------------------------
# 3) Entraînement & Évaluation — LR 80/20 (J -> J+1)
# -----------------------------------------------------------------------------
def LR_model():
    """
    Entraîne une régression linéaire (80%/20% temporel) pour prédire
    OIL/GAS/WAT à J+1 à partir des features du jour.
    """
    logger.info("Démarrage LR (Multi-sorties : Huile, Gaz, Eau à J+1)...")

    # A. Lecture données
    df = load_pwell_data()
    if df.empty:
        raise RuntimeError("Aucune donnée disponible (DB et fichiers vides).")

    # B. Date
    if "DAYTIME" in df.columns:
        df["DAYTIME"] = pd.to_datetime(df["DAYTIME"])
    elif "DATEPRD" in df.columns:
        df["DAYTIME"] = pd.to_datetime(df["DATEPRD"])
    else:
        raise ValueError("Colonne DAYTIME/DATEPRD manquante dans la source de données.")
    df = df.sort_values("DAYTIME").reset_index(drop=True)

    # C. Assurer les features
    for c in REQUIRED_FEATURES:
        if c not in df.columns:
            df[c] = np.nan  # si absent, on met NaN (sera filtré par dropna)
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED_FEATURES + ["DAYTIME"])
    if df.empty:
        raise RuntimeError("Après nettoyage, plus de lignes complètes pour les features requises.")

    # D. Cibles J+1
    df["NEXT_OIL"] = df["BORE_OIL_VOL"].shift(-1)
    df["NEXT_GAS"] = df["BORE_GAS_VOL"].shift(-1)
    df["NEXT_WAT"] = df["BORE_WAT_VOL"].shift(-1)
    dfm = df.dropna(subset=["NEXT_OIL", "NEXT_GAS", "NEXT_WAT"]).copy()

    X = dfm[REQUIRED_FEATURES].values
    y = dfm[["NEXT_OIL", "NEXT_GAS", "NEXT_WAT"]].values
    dates = dfm["DAYTIME"].values

    # E. Split temporel 80/20
    split_idx = int(len(dfm) * 0.8)
    if split_idx == 0 or split_idx == len(dfm):
        raise RuntimeError("Pas assez d'échantillons pour un split 80/20.")
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = pd.to_datetime(dates[split_idx:]) + timedelta(days=1)

    # F. Entraînement LR
    model = LinearRegression()
    model.fit(X_train, y_train)

    # G. Prédictions & scores
    y_pred = model.predict(X_test)

    r2_global = r2_score(y_test, y_pred)
    mse_global = mean_squared_error(y_test, y_pred)

    r2_oil = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_gas = r2_score(y_test[:, 1], y_pred[:, 1])
    r2_wat = r2_score(y_test[:, 2], y_pred[:, 2])

    mse_oil = mean_squared_error(y_test[:, 0], y_pred[:, 0])
    mse_gas = mean_squared_error(y_test[:, 1], y_pred[:, 1])
    mse_wat = mean_squared_error(y_test[:, 2], y_pred[:, 2])

    logger.info(f"LR terminé — R2 global: {r2_global:.4f} | MSE global: {mse_global:.4f}")

    # H. Résultats
    id_model = str(uuid.uuid4())
    df_params = pd.DataFrame([{
        "id_model": id_model,
        "timestamp": pd.Timestamp.now(),
        "algo": "LinearRegression",
        "r2_global": float(r2_global),
        "mse_global": float(mse_global),
        "r2_oil": float(r2_oil), "r2_gas": float(r2_gas), "r2_wat": float(r2_wat),
        "mse_oil": float(mse_oil), "mse_gas": float(mse_gas), "mse_wat": float(mse_wat),
        "source": "DB" if HAS_PYMYSQL else "CSV"
    }])

    df_predictions = pd.DataFrame({
        "OIL_VOL":  y_pred[:, 0],
        "GAS_VOL":  y_pred[:, 1],
        "WAT_VOL":  y_pred[:, 2],
        "BORE_OIL_VOL": y_test[:, 0],
        "BORE_GAS_VOL": y_test[:, 1],
        "BORE_WAT_VOL": y_test[:, 2],
        "DAYTIME": dates_test,
        "id_model": id_model,
        "model_type": "LR",
    })

    return df_params, df_predictions

# -----------------------------------------------------------------------------
# 4) Sauvegardes : DB si dispo, sinon CSV
# -----------------------------------------------------------------------------
def save_params(df_params: pd.DataFrame):
    conn = try_connect_db()
    if conn is None:
        # Fichier
        out = OUTPUTS_DIR / "LR_PARAM.csv"
        mode = "a" if out.exists() else "w"
        header = not out.exists()
        df_params.to_csv(out, index=False, mode=mode, header=header)
        logger.info(f"Paramètres LR sauvegardés → {out}")
        return

    # DB
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS LR_PARAM (
                    id_model   VARCHAR(36) PRIMARY KEY,
                    timestamp  DATETIME,
                    algo       VARCHAR(64),
                    r2_global  DOUBLE,
                    mse_global DOUBLE,
                    r2_oil DOUBLE, r2_gas DOUBLE, r2_wat DOUBLE,
                    mse_oil DOUBLE, mse_gas DOUBLE, mse_wat DOUBLE,
                    source VARCHAR(10)
                );
            """)
            for _, row in df_params.iterrows():
                cur.execute("""
                    INSERT INTO LR_PARAM
                    (id_model, timestamp, algo, r2_global, mse_global,
                     r2_oil, r2_gas, r2_wat, mse_oil, mse_gas, mse_wat, source)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        timestamp=VALUES(timestamp),
                        algo=VALUES(algo),
                        r2_global=VALUES(r2_global),
                        mse_global=VALUES(mse_global),
                        r2_oil=VALUES(r2_oil),
                        r2_gas=VALUES(r2_gas),
                        r2_wat=VALUES(r2_wat),
                        mse_oil=VALUES(mse_oil),
                        mse_gas=VALUES(mse_gas),
                        mse_wat=VALUES(mse_wat),
                        source=VALUES(source);
                """, (
                    row["id_model"], row["timestamp"], row["algo"],
                    row["r2_global"], row["mse_global"],
                    row["r2_oil"], row["r2_gas"], row["r2_wat"],
                    row["mse_oil"], row["mse_gas"], row["mse_wat"],
                    row.get("source", "CSV")
                ))
        conn.commit()
        logger.info("Paramètres LR sauvegardés dans LR_PARAM (DB)")
    finally:
        try:
            conn.close()
        except Exception:
            pass

def save_predictions(df_preds: pd.DataFrame):
    conn = try_connect_db()
    if conn is None:
        out = OUTPUTS_DIR / "LR_MODEL_AI.csv"
        mode = "a" if out.exists() else "w"
        header = not out.exists()
        df_preds.to_csv(out, index=False, mode=mode, header=header)
        logger.info(f"Prédictions LR sauvegardées → {out}")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS MODEL_AI (
                    id_model   VARCHAR(36),
                    model_type VARCHAR(32),
                    DAYTIME    DATE,
                    OIL_VOL    DOUBLE,
                    GAS_VOL    DOUBLE,
                    WAT_VOL    DOUBLE,
                    PRIMARY KEY (id_model, DAYTIME)
                );
            """)
            for _, row in df_preds.iterrows():
                d = pd.to_datetime(row["DAYTIME"]).date()
                cur.execute("""
                    INSERT INTO MODEL_AI
                    (id_model, model_type, DAYTIME, OIL_VOL, GAS_VOL, WAT_VOL)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        OIL_VOL=VALUES(OIL_VOL),
                        GAS_VOL=VALUES(GAS_VOL),
                        WAT_VOL=VALUES(WAT_VOL);
                """, (
                    row["id_model"], row["model_type"], d,
                    float(row["OIL_VOL"]), float(row["GAS_VOL"]), float(row["WAT_VOL"])
                ))
        conn.commit()
        logger.info("Prédictions LR sauvegardées dans MODEL_AI (DB)")
    finally:
        try:
            conn.close()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# 5) Run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df_params, df_preds = LR_model()
    save_params(df_params)
    save_predictions(df_preds)

    print("\n=== LR_PARAM (aperçu) ===")
    print(df_params.round(4))
    print("\n=== LR_MODEL_AI (aperçu) ===")
    print(df_preds.head())
