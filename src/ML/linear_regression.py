import pandas as pd
import numpy as np
import logging
import uuid
import pymysql
import joblib


from config.sql_config import sql_settings
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

from data_access.sql_reader import call_data_sql
from util.cleaning import fill_na, normalize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LR_AI")

def lr_model():
    try:
        query="""SELECT  UNIX_TIMESTAMP(DAYTIME) AS DAYTIME,
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
            """
        df=call_data_sql(query)
        df.columns = [c.upper() for c in df.columns]

    except Exception as e:
        logger.error(f"Erreur SQL : {e}")
        return None, None
   
    if df.empty:
        logger.error("La base de données est vide.")
        return None, None

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
    "estimator__fit_intercept": [True, False]
    }   

    base_model = LinearRegression()

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

    # G. Prédictions & scores
    y_pred = search.predict(X_test)

    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test.values)

    r2_global = r2_score(y_test, y_pred)
    mse_global = mean_squared_error(y_test, y_pred)

    y_test_arr = y_test.values

    r2_oil = r2_score(y_test_arr[:, 0], y_pred[:, 0])
    r2_gas = r2_score(y_test_arr[:, 1], y_pred[:, 1])
    r2_wat = r2_score(y_test_arr[:, 2], y_pred[:, 2])

    mse_oil = mean_squared_error(y_test_arr[:, 0], y_pred[:, 0])
    mse_gas = mean_squared_error(y_test_arr[:, 1], y_pred[:, 1])
    mse_wat = mean_squared_error(y_test_arr[:, 2], y_pred[:, 2])


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
        "mse_oil": float(mse_oil), "mse_gas": float(mse_gas), "mse_wat": float(mse_wat)
    }])

    lr_param_load(df_params)

    df_predictions = pd.DataFrame(
        np.hstack([y_test_inv, y_pred_inv]),
        columns=["OIL_VOL_test", "GAS_VOL_test", "WAT_VOL_test","OIL_VOL_pred", "GAS_VOL_pred", "WAT_VOL_pred"])
    df_predictions["DAYTIME"] = X_test_dates.values

    df_predictions["id_model"] = id_model
    df_predictions["model_type"] = "LR"

    return df_predictions

def lr_param_load(df):
    """
    Loads Linear Regression parameters into a MySQL database using pymysql.
    """
    try:
        connection = pymysql.connect(
            user=sql_settings.user,
            password=sql_settings.password,
            database=sql_settings.database,
            cursorclass=sql_settings.cursorclass
        )
        logger.info("Connection established!")

        with connection.cursor() as cursor:

            # Création de la table adaptée à LinearRegression
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS LR_PARAM (
                    id_model VARCHAR(36) PRIMARY KEY,
                    timestamp DATETIME,
                    algo VARCHAR(50),
                    r2_global DECIMAL(10,5),
                    mse_global DECIMAL(20,5),
                    r2_oil DECIMAL(10,5),
                    r2_gas DECIMAL(10,5),
                    r2_wat DECIMAL(10,5),
                    mse_oil DECIMAL(20,5),
                    mse_gas DECIMAL(20,5),
                    mse_wat DECIMAL(20,5)
                );
            """)

            insert_query = """
                INSERT INTO LR_PARAM (
                    id_model, timestamp, algo,
                    r2_global, mse_global,
                    r2_oil, r2_gas, r2_wat,
                    mse_oil, mse_gas, mse_wat
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    timestamp = VALUES(timestamp),
                    algo = VALUES(algo),
                    r2_global = VALUES(r2_global),
                    mse_global = VALUES(mse_global),
                    r2_oil = VALUES(r2_oil),
                    r2_gas = VALUES(r2_gas),
                    r2_wat = VALUES(r2_wat),
                    mse_oil = VALUES(mse_oil),
                    mse_gas = VALUES(mse_gas),
                    mse_wat = VALUES(mse_wat)
            """

            logger.info(type(df))
            for _, row in df.iterrows():
                cursor.execute(
                    insert_query,
                    (
                        row["id_model"],
                        row["timestamp"],
                        row["algo"],
                        row["r2_global"],
                        row["mse_global"],
                        row["r2_oil"],
                        row["r2_gas"],
                        row["r2_wat"],
                        row["mse_oil"],
                        row["mse_gas"],
                        row["mse_wat"]
                    )
                )

        connection.commit()
        logger.info(
            f"Linear Regression parameters saved successfully with id: {df['id_model'].values[0]}"
        )

    except Exception as e:
        logger.error(f"Erreur Param LR: {e}")
        if 'connection' in locals():
            connection.rollback()
        raise

    finally:
        if 'connection' in locals():
            connection.close()
            logger.info("Connection closed")
