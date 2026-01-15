import pymysql
import logging

logger = logging.getLogger(__name__)

from config.sql_config import sql_settings

def create_ml_tables():
    """Créer les tables ML si elles n'existent pas"""
    connection = pymysql.connect(
        user=sql_settings.user,
        password=sql_settings.password,
        database=sql_settings.database,
        cursorclass=sql_settings.cursorclass
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS xgboost_ml_runs (
                    run_id INT AUTO_INCREMENT PRIMARY KEY,
                    run_datetime DATETIME DEFAULT CURRENT_TIMESTAMP,
                    comment VARCHAR(255)
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS xgboost_ml_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    run_id INT,
                    target VARCHAR(50),
                    rmse FLOAT,
                    mae FLOAT,
                    r2 FLOAT,
                    FOREIGN KEY (run_id) REFERENCES xgboost_ml_runs(run_id)
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS xgboost_ml_params (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    run_id INT,
                    target VARCHAR(50),
                    n_estimators INT,
                    max_depth INT,
                    learning_rate FLOAT,
                    FOREIGN KEY (run_id) REFERENCES xgboost_ml_runs(run_id)
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS xgboost_ml_predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    run_id INT NOT NULL,
                    target VARCHAR(50),
                    y_true FLOAT,
                    y_pred FLOAT,
                    sample_type VARCHAR(10),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
        connection.commit()
        logger.info("ML tables created / verified")
    finally:
        connection.close()


def create_ml_run(comment: str = None) -> int:
    """Crée un run et retourne son ID"""
    connection = pymysql.connect(
        user=sql_settings.user,
        password=sql_settings.password,
        database=sql_settings.database,
        cursorclass=sql_settings.cursorclass
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO xgboost_ml_runs (comment) VALUES (%s)", (comment,))
            run_id = cursor.lastrowid
        connection.commit()
        return run_id
    finally:
        connection.close()


def save_metrics(run_id: int, metrics_dict: dict):
    connection = pymysql.connect(
        user=sql_settings.user,
        password=sql_settings.password,
        database=sql_settings.database,
        cursorclass=sql_settings.cursorclass
    )

    try:
        with connection.cursor() as cursor:
            for target, m in metrics_dict.items():
                cursor.execute("""
                    INSERT INTO xgboost_ml_results
                    (run_id, target, rmse, mae, r2)
                    VALUES (%s, %s, %s, %s, %s)
                """, (run_id, target, m["rmse"], m["mae"], m["r2"]))
        connection.commit()
    finally:
        connection.close()


def save_params(run_id: int, params_dict: dict):
    connection = pymysql.connect(
        user=sql_settings.user,
        password=sql_settings.password,
        database=sql_settings.database,
        cursorclass=sql_settings.cursorclass
    )
    try:
        with connection.cursor() as cursor:
            for target, p in params_dict.items():
                cursor.execute("""
                    INSERT INTO xgboost_ml_params
                    (run_id, target, n_estimators, max_depth, learning_rate)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    run_id,
                    target,
                    p["n_estimators"],
                    p["max_depth"],
                    p["learning_rate"]
                ))
        connection.commit()
    finally:
        connection.close()


def save_predictions(run_id, y_true, y_pred, sample_type="test"):
    connection = pymysql.connect(
        user=sql_settings.user,
        password=sql_settings.password,
        database=sql_settings.database,
        cursorclass=sql_settings.cursorclass
    )
    try:
        with connection.cursor() as cursor:
            insert_query = """
                INSERT INTO xgboost_ml_predictions
                (run_id, target, y_true, y_pred, sample_type)
                VALUES (%s, %s, %s, %s, %s)
            """
            for target in y_pred:
                for true_val, pred_val in zip(y_true[target], y_pred[target]):
                    cursor.execute(insert_query, (run_id, target, float(true_val), float(pred_val), sample_type))
        connection.commit()
    finally:
        connection.close()
