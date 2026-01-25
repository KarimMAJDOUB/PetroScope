import pandas as pd
import numpy as np
import uuid
import logging
import pymysql

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from src.data_access.sql_reader import call_data_sql
from src.config.sql_config import sql_settings
from src.ml.model_load import model_load

logger = logging.getLogger(__name__)

def xgb_model():
    df = call_data_sql("""
        SELECT DAYTIME,
            SUM(BORE_OIL_VOL) AS OIL_VOL,
            SUM(BORE_GAS_VOL) AS GAS_VOL,
            SUM(BORE_WAT_VOL) AS WAT_VOL
        FROM PWELL_DATA
        GROUP BY DAYTIME
        ORDER BY DAYTIME ASC
    """)
    df["DAYTIME"] = pd.to_datetime(df["DAYTIME"])
    df = df.reset_index(drop=True)

    values = df[["OIL_VOL", "GAS_VOL", "WAT_VOL"]].values.astype("float32")

    X = values[:-1]
    y = values[1:]
    time_test = df["DAYTIME"].iloc[1:]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1]
    }

    y_pred_test = np.zeros_like(y_test)
    best_params = {}
    best_score = {}

    for i, target in enumerate(["OIL_VOL", "GAS_VOL", "WAT_VOL"]):
        search = RandomizedSearchCV(
            XGBRegressor(objective='reg:squarederror', random_state=42),
            param_distributions=param_grid,
            n_iter=5,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_mean_squared_error",
            verbose=0,
            random_state=42
        )
        search.fit(X_train, y_train[:, i])
        y_pred_test[:, i] = search.predict(X_test)
        best_params[target] = search.best_params_
        best_score[target] = -search.best_score_

    id_model = str(uuid.uuid4())

    df_best = pd.DataFrame([best_params])
    df_best["mse_OIL"] = best_score["OIL_VOL"]
    df_best["mse_GAS"] = best_score["GAS_VOL"]
    df_best["mse_WAT"] = best_score["WAT_VOL"]
    df_best["timestamp"] = pd.Timestamp.now()
    df_best["id_model"] = id_model

    df_model = pd.DataFrame(
        np.hstack([y_test, y_pred_test]),
        columns=["OIL_VOL_test","GAS_VOL_test","WAT_VOL_test","OIL_VOL_pred","GAS_VOL_pred","WAT_VOL_pred"]
    )
    df_model["DAYTIME"] = time_test.values
    df_model["id_model"] = id_model
    df_model["model_type"] = "XGBOOST"

    return df_best, df_model

def xgb_param_load(df) -> None:
    try:
        connection = pymysql.connect(
            user=sql_settings.user,
            password=sql_settings.password,
            database=sql_settings.database,
            cursorclass=sql_settings.cursorclass
        )
        logger.info("Connection established")
    except pymysql.err.OperationalError as e:
        logger.error(f"Connection failed: {e}")
        return

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS XGB_PARAM (
                    id_model VARCHAR(36) PRIMARY KEY,
                    timestamp DATETIME,
                    n_estimators_OIL INT,
                    max_depth_OIL INT,
                    learning_rate_OIL FLOAT,
                    n_estimators_GAS INT,
                    max_depth_GAS INT,
                    learning_rate_GAS FLOAT,
                    n_estimators_WAT INT,
                    max_depth_WAT INT,
                    learning_rate_WAT FLOAT,
                    mse_OIL FLOAT,
                    mse_GAS FLOAT,
                    mse_WAT FLOAT
                ); 
            """)

            insert_objects = """
                INSERT INTO XGB_PARAM
                (id_model, timestamp,
                 n_estimators_OIL, max_depth_OIL, learning_rate_OIL,
                 n_estimators_GAS, max_depth_GAS, learning_rate_GAS,
                 n_estimators_WAT, max_depth_WAT, learning_rate_WAT,
                 mse_OIL, mse_GAS, mse_WAT)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                    timestamp=VALUES(timestamp),
                    n_estimators_OIL=VALUES(n_estimators_OIL),
                    max_depth_OIL=VALUES(max_depth_OIL),
                    learning_rate_OIL=VALUES(learning_rate_OIL),
                    n_estimators_GAS=VALUES(n_estimators_GAS),
                    max_depth_GAS=VALUES(max_depth_GAS),
                    learning_rate_GAS=VALUES(learning_rate_GAS),
                    n_estimators_WAT=VALUES(n_estimators_WAT),
                    max_depth_WAT=VALUES(max_depth_WAT),
                    learning_rate_WAT=VALUES(learning_rate_WAT),
                    mse_OIL=VALUES(mse_OIL),
                    mse_GAS=VALUES(mse_GAS),
                    mse_WAT=VALUES(mse_WAT);
            """

            for _, row in df.iterrows():
                cursor.execute(insert_objects, (
                    row["id_model"], row["timestamp"],
                    row["OIL_VOL"]["n_estimators"], row["OIL_VOL"]["max_depth"], row["OIL_VOL"]["learning_rate"],
                    row["GAS_VOL"]["n_estimators"], row["GAS_VOL"]["max_depth"], row["GAS_VOL"]["learning_rate"],
                    row["WAT_VOL"]["n_estimators"], row["WAT_VOL"]["max_depth"], row["WAT_VOL"]["learning_rate"],
                    row["mse_OIL"], row["mse_GAS"], row["mse_WAT"]
                ))

        connection.commit()
    finally:
        logger.info("Connection closed")
        connection.close()


if __name__ == "__main__":
    df_best, df_model = xgb_model()
    xgb_param_load(df_best)
    model_load(df_model)
