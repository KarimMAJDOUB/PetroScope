import pandas as pd
import pymysql
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.sql_config import sql_settings

logger = logging.getLogger(__name__)

def model_load(df) -> None:
    """
    Loads data into a MySQL database using pymysql.

    --------------------------------------------

    The df MUST HAVE THE NEXT SCHEMA !!!!!!!!

    --------------------------------------------
    =================================================================
    | id_model | model_type | DAYTIME | OIL_VOL | GAS_VOL | WAT_VOL |
    =================================================================

    Where :
    - id_model : str(uuid.uuid4())
    - model_type : your model (RF, LR, LSTM...)
    - DAYTIME : x_test
    - OIL_VOL : y_pred
    - GAS_VOL : y_pred
    - WAT_VOL : y_pred

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
    

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS MODEL_AI (
                    id_model VARCHAR(36),
                    model_type VARCHAR(32),
                    DAYTIME DATE,
                    OIL_VOL DOUBLE,
                    GAS_VOL DOUBLE,
                    WAT_VOL DOUBLE,
                    PRIMARY KEY(id_model, DAYTIME)
                ); 
            """)

            insert_objects = """
                INSERT INTO MODEL_AI
                (id_model,
                    model_type  ,
                    DAYTIME  ,
                    OIL_VOL  ,
                    GAS_VOL  ,
                    WAT_VOL)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    model_type = VALUES(model_type),
                    DAYTIME = VALUES(DAYTIME),
                    OIL_VOL = VALUES(OIL_VOL),
                    GAS_VOL = VALUES(GAS_VOL),
                    WAT_VOL = VALUES(WAT_VOL);
            """

            for _, row in df.iterrows():
                daytime = row["DAYTIME"]
                if pd.notna(daytime):
                    daytime = daytime.date()
                
                cursor.execute(
                    insert_objects,
                    (
                        row['id_model'],
                        row['model_type'],
                        daytime,
                        row['OIL_VOL'],
                        row['GAS_VOL'],
                        row['WAT_VOL']
                    )
                )

        connection.commit()
    finally:
        logger.info("Connection closed")
        connection.close()