import pandas as pd
import logging
import sys
import os
import numpy as np
from math import sqrt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_access.sql_reader import call_data_sql
from ml_helpers import create_ml_tables, create_ml_run, save_metrics, save_params, save_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# Load training data
# ------------------------
def load_training_data():
    query = """
    SELECT
        ON_STREAM_HRS,
        AVG_DP_TUBING,
        AVG_ANNULUS_PRESS,
        AVG_CHOKE_SIZE_P,
        AVG_WHP_P,
        AVG_WHT_P,
        DP_CHOKE_SIZE,
        BORE_WI_VOL,
        BORE_OIL_VOL,
        BORE_GAS_VOL,
        BORE_WAT_VOL
    FROM pwell_data
    """
    return call_data_sql(query)

# ------------------------
# Main pipeline
# ------------------------
if __name__ == "__main__":
    logger.info("XGBoost ML pipeline started")

    #Tables
    create_ml_tables()

    #New run
    run_id = create_ml_run(comment="XGBoost multi-target v1")
    logger.info(f"Run ID: {run_id}")

    #Data
    df = load_training_data()
    X = df.drop(columns=["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]).apply(pd.to_numeric)
    y = df[["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]].apply(pd.to_numeric)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Hyperparameter tuning
    params = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.01, 0.1]
    }

    best_models = {}
    metrics_dict = {}
    params_dict = {}
    y_pred_test = pd.DataFrame()

    for target in y.columns:
        logger.info(f"Tuning {target}")
        search = RandomizedSearchCV(
            XGBRegressor(random_state=42),
            params,

            verbose=1,
            random_state=42,
        )

        #log-transform
        y_train_log = np.log1p(y_train[target])
        search.fit(X_train, y_train_log)
        model = search.best_estimator_
        best_models[target] = model
        params_dict[target] = search.best_params_

        #Predict and inverse transform
        preds = np.expm1(model.predict(X_test))
        preds = np.clip(preds, 0, None)
        y_pred_test[target] = preds

        #Metrics
        metrics_dict[target] = {
            "rmse": sqrt(mean_squared_error(y_test[target], preds)),
            "mae": mean_absolute_error(y_test[target], preds),
            "r2": r2_score(y_test[target], preds)
        }

    #Save to SQL
    save_params(run_id, params_dict)
    save_metrics(run_id, metrics_dict)
    save_predictions(run_id, y_test, y_pred_test, sample_type="test")

    logger.info("ML run saved successfully")
