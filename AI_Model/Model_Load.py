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
    - OIL_VOL_test : y_test
    - GAS_VOL_test : y_test
    - WAT_VOL_test : y_test
    - OIL_VOL_pred : y_pred
    - GAS_VOL_pred : y_pred
    - WAT_VOL_pred : y_pred

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
                    OIL_VOL_test DOUBLE,
                    GAS_VOL_test DOUBLE,
                    WAT_VOL_test DOUBLE,
                    OIL_VOL_pred DOUBLE,
                    GAS_VOL_pred DOUBLE,
                    WAT_VOL_pred DOUBLE,
                    PRIMARY KEY(id_model, DAYTIME)
                ); 
            """)

            insert_objects = """
                INSERT INTO MODEL_AI
                (id_model,
                    model_type  ,
                    DAYTIME  ,
                    OIL_VOL_test  ,
                    GAS_VOL_test  ,
                    WAT_VOL_test,
                    OIL_VOL_pred  ,
                    GAS_VOL_pred  ,
                    WAT_VOL_pred)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    model_type = VALUES(model_type),
                    DAYTIME = VALUES(DAYTIME),
                    OIL_VOL_test  = VALUES(OIL_VOL_test),
                    GAS_VOL_test  = VALUES(GAS_VOL_test),
                    WAT_VOL_test  = VALUES(WAT_VOL_test),
                    OIL_VOL_pred = VALUES(OIL_VOL_pred),
                    GAS_VOL_pred = VALUES(GAS_VOL_pred),
                    WAT_VOL_pred = VALUES(WAT_VOL_pred);
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
                        row['OIL_VOL_test'],
                        row['GAS_VOL_test'],
                        row['WAT_VOL_test'],
                        row['OIL_VOL_pred'],
                        row['GAS_VOL_pred'],
                        row['WAT_VOL_pred']
                    )
                )

        connection.commit()
    finally:
        logger.info("Connection closed")
        connection.close()

def PF_Load(df):
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
    - DAYTIME : x_pf
    - OIL_VOL : y_pf
    - GAS_VOL : y_pf
    - WAT_VOL : y_pf


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
                CREATE TABLE IF NOT EXISTS MODEL_PF (
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
                INSERT INTO MODEL_PF
                (id_model,
                    model_type  ,
                    DAYTIME  ,
                    OIL_VOL,
                    GAS_VOL,
                    WAT_VOL)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    model_type = VALUES(model_type),
                    DAYTIME = VALUES(DAYTIME),
                    OIL_VOL  = VALUES(OIL_VOL),
                    GAS_VOL  = VALUES(GAS_VOL),
                    WAT_VOL  = VALUES(WAT_VOL);
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
                        row['OIL_VOL_pred'],
                        row['GAS_VOL_pred'],
                        row['WAT_VOL_pred']
                    )
                )

        connection.commit()
    finally:
        logger.info("Connection closed")
        connection.close()

