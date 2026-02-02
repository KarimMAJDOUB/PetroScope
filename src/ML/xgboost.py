import pandas as pd
import numpy as np
import uuid
import logging
import pymysql
import datetime
import time

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from data_access.sql_reader import call_data_sql
from config.sql_config import sql_settings
from ml.model_load import model_load

from util.cleaning import fill_na, normalize

logger = logging.getLogger(__name__)

def xgb_model():
    df = call_data_sql("""
        SELECT  UNIX_TIMESTAMP(DAYTIME) AS DAYTIME,
				AVG(BORE_OIL_VOL) AS OIL_VOL,
                AVG(BORE_GAS_VOL) AS GAS_VOL,
                AVG(BORE_WAT_VOL) AS WAT_VOL,
                AVG(ON_STREAM_HRS) AS ON_STREAM_HRS,
                AVG(AVG_DOWNHOLE_PRESSURE) AS AVG_DOWNHOLE_PRESSURE,
                AVG(AVG_DOWNHOLE_TEMPERATURE) AS AVG_DOWNHOLE_TEMPERATURE,
                AVG(AVG_DP_TUBING) AS AVG_DP_TUBING,
                AVG(AVG_ANNULUS_PRESS) AS AVG_ANNULUS_PRESS,
                AVG(AVG_CHOKE_SIZE_P) AS AVG_CHOKE_SIZE_P,
                AVG(AVG_WHP_P) AS AVG_WHP_P,
                AVG(AVG_WHT_P) AS AVG_WHT_P,
                AVG(DP_CHOKE_SIZE) AS DP_CHOKE_SIZE
            FROM PWELL_DATA
            GROUP BY UNIX_TIMESTAMP(DAYTIME)
            ORDER BY UNIX_TIMESTAMP(DAYTIME)
    """)

    df_fill_na=fill_na(df,5)

    # df_norm=normalize(df_fill_na)
    
    features_cols = df_fill_na.drop(columns=['OIL_VOL','GAS_VOL','WAT_VOL']).columns

    for col in features_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X_norm, scaler_X = normalize(df_fill_na[features_cols], columns=features_cols)

    y_norm, scaler_y = normalize(df_fill_na[['OIL_VOL','GAS_VOL','WAT_VOL']], columns=['OIL_VOL','GAS_VOL','WAT_VOL'])
    
    X_train, X_test, y_train, y_test = train_test_split(
    X_norm,
    y_norm,
    test_size=0.2,
    shuffle=False
    )

    X_test_dates = df_fill_na.loc[X_test.index, "DAYTIME"]
    X_test_dates = pd.to_datetime(X_test_dates, unit="s")

    param_grid = {
    "estimator__n_estimators": [200, 400, 600],
    "estimator__max_depth": [3, 5, 7],
    "estimator__learning_rate": [0.01, 0.05, 0.1],
    "estimator__subsample": [0.8, 1.0],
    "estimator__colsample_bytree": [0.8, 1.0]
    }

    best_params = {}
    best_score = {}

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42
    )

    model = MultiOutputRegressor(base_model)

    search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=5,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_mean_squared_error",
            verbose=0,
            random_state=42
        )
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test.values)

    mse_per_output = mean_squared_error(
        y_test,
        y_pred,
        multioutput="raw_values"
    )

    mse_OIL, mse_GAS, mse_WAT = mse_per_output


    best_params = search.best_params_
    best_score = search.best_score_
    mse = mean_squared_error(y_test, y_pred)

    id_model = str(uuid.uuid4())

    df_best = pd.DataFrame([best_params])

    df_best["mse_OIL"] = mse_OIL
    df_best["mse_GAS"] = mse_GAS
    df_best["mse_WAT"] = mse_WAT
    df_best["timestamp"] = pd.Timestamp.now()
    df_best["id_model"] = id_model

    xgb_param_load(df_best)

    df_model = pd.DataFrame(
        np.hstack([y_test_inv, y_pred_inv]),
        columns=["OIL_VOL_test","GAS_VOL_test","WAT_VOL_test","OIL_VOL_pred","GAS_VOL_pred","WAT_VOL_pred"]
    )

    df_model["DAYTIME"] = X_test_dates.values
    df_model["id_model"] = id_model
    df_model["model_type"] = "XGBOOST"

    return df_model

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
                    row["estimator__n_estimators"], row["estimator__max_depth"], row["estimator__learning_rate"],
                    row["estimator__n_estimators"], row["estimator__max_depth"], row["estimator__learning_rate"],
                    row["estimator__n_estimators"], row["estimator__max_depth"], row["estimator__learning_rate"],

                    row["mse_OIL"], row["mse_GAS"], row["mse_WAT"]
                ))


        connection.commit()
    finally:
        logger.info("Connection closed")
        connection.close()