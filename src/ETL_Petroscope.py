import pandas as pd
import os
import pymysql
import logging
from config.sql_config import sql_settings

logger = logging.getLogger(__name__)



def data_extract(file_name):
    """
    Transforme les données brutes en DataFrame Pandas propre.
    """
    path_file=os.path.join('Data', file_name)
    df = pd.DataFrame(pd.read_csv(path_file))

    return df

def data_transform(df):
    numeric_cols = [
                'ON_STREAM_HRS',
                'AVG_DOWNHOLE_PRESSURE',
                'AVG_DOWNHOLE_TEMPERATURE',
                'AVG_DP_TUBING',
                'AVG_ANNULUS_PRESS',
                'AVG_CHOKE_SIZE_P',
                'AVG_WHP_P',
                'AVG_WHT_P',
                'DP_CHOKE_SIZE',
                'BORE_OIL_VOL',
                'BORE_GAS_VOL',
                'BORE_WAT_VOL',
                'BORE_WI_VOL'
            ]

    df[numeric_cols] = (
        df[numeric_cols]
        .astype(object)
        .where(pd.notnull(df[numeric_cols]), None)
    )

    text_cols = [
        'AVG_CHOKE_UOM',
        'FLOW_KIND',
        'NPD_WELL_BORE_CODE',
        'NPD_WELL_BORE_NAME',
        'NPD_FIELD_CODE',
        'NPD_FIELD_NAME',
        'WELL_TYPE'
    ]

    df[text_cols] = df[text_cols].where(pd.notnull(df[text_cols]), None)
    return df

def load(df) -> None:
    """
    Loads data into a MySQL database using pymysql.
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
                CREATE TABLE IF NOT EXISTS OBJECTS (
                    DAYTIME DATE,
                    WELL_BORE_CODE VARCHAR(50),
                    NPD_WELL_BORE_CODE VARCHAR(50),
                    NPD_WELL_BORE_NAME VARCHAR(255),
                    NPD_FIELD_CODE VARCHAR(50),
                    NPD_FIELD_NAME VARCHAR(255),
                    WELL_TYPE VARCHAR(255),
                    PRIMARY KEY (DAYTIME, WELL_BORE_CODE)
                );
            """)

            insert_objects = """
                INSERT INTO OBJECTS
                (DAYTIME, WELL_BORE_CODE, NPD_WELL_BORE_CODE,
                 NPD_WELL_BORE_NAME, NPD_FIELD_CODE, NPD_FIELD_NAME, WELL_TYPE)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    NPD_WELL_BORE_CODE = VALUES(NPD_WELL_BORE_CODE),
                    NPD_WELL_BORE_NAME = VALUES(NPD_WELL_BORE_NAME),
                    NPD_FIELD_CODE = VALUES(NPD_FIELD_CODE),
                    NPD_FIELD_NAME = VALUES(NPD_FIELD_NAME),
                    WELL_TYPE = VALUES(WELL_TYPE);
            """

            for _, row in df.iterrows():
                cursor.execute(
                    insert_objects,
                    (
                        row['DATEPRD'],
                        row['WELL_BORE_CODE'],
                        row['NPD_WELL_BORE_CODE'],
                        row['NPD_WELL_BORE_NAME'],
                        row['NPD_FIELD_CODE'],
                        row['NPD_FIELD_NAME'],
                        row['WELL_TYPE']
                    )
                )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS PWELL_DATA (
                    DAYTIME DATE,
                    WELL_BORE_CODE VARCHAR(50),

                    ON_STREAM_HRS DECIMAL(10,2),
                    AVG_DOWNHOLE_PRESSURE DECIMAL(10,2),
                    AVG_DOWNHOLE_TEMPERATURE DECIMAL(10,2),
                    AVG_DP_TUBING DECIMAL(10,2),
                    AVG_ANNULUS_PRESS DECIMAL(10,2),
                    AVG_CHOKE_SIZE_P DECIMAL(10,2),
                    AVG_CHOKE_UOM VARCHAR(20),
                    AVG_WHP_P DECIMAL(10,2),
                    AVG_WHT_P DECIMAL(10,2),
                    DP_CHOKE_SIZE DECIMAL(10,2),

                    BORE_OIL_VOL DECIMAL(14,3),
                    BORE_GAS_VOL DECIMAL(14,3),
                    BORE_WAT_VOL DECIMAL(14,3),
                    BORE_WI_VOL DECIMAL(14,3),

                    FLOW_KIND VARCHAR(50),

                    PRIMARY KEY (DAYTIME, WELL_BORE_CODE)
                );
            """)

            insert_pwell = """
                INSERT INTO PWELL_DATA
                (DAYTIME, WELL_BORE_CODE,
                 ON_STREAM_HRS, AVG_DOWNHOLE_PRESSURE, AVG_DOWNHOLE_TEMPERATURE,
                 AVG_DP_TUBING, AVG_ANNULUS_PRESS, AVG_CHOKE_SIZE_P,
                 AVG_CHOKE_UOM, AVG_WHP_P, AVG_WHT_P, DP_CHOKE_SIZE,
                 BORE_OIL_VOL, BORE_GAS_VOL, BORE_WAT_VOL, BORE_WI_VOL,
                 FLOW_KIND)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    ON_STREAM_HRS = VALUES(ON_STREAM_HRS),
                    AVG_DOWNHOLE_PRESSURE = VALUES(AVG_DOWNHOLE_PRESSURE),
                    AVG_DOWNHOLE_TEMPERATURE = VALUES(AVG_DOWNHOLE_TEMPERATURE),
                    AVG_DP_TUBING = VALUES(AVG_DP_TUBING),
                    AVG_ANNULUS_PRESS = VALUES(AVG_ANNULUS_PRESS),
                    AVG_CHOKE_SIZE_P = VALUES(AVG_CHOKE_SIZE_P),
                    AVG_CHOKE_UOM = VALUES(AVG_CHOKE_UOM),
                    AVG_WHP_P = VALUES(AVG_WHP_P),
                    AVG_WHT_P = VALUES(AVG_WHT_P),
                    DP_CHOKE_SIZE = VALUES(DP_CHOKE_SIZE),
                    BORE_OIL_VOL = VALUES(BORE_OIL_VOL),
                    BORE_GAS_VOL = VALUES(BORE_GAS_VOL),
                    BORE_WAT_VOL = VALUES(BORE_WAT_VOL),
                    BORE_WI_VOL = VALUES(BORE_WI_VOL),
                    FLOW_KIND = VALUES(FLOW_KIND);
            """

            for _, row in df.iterrows():
                cursor.execute(
                    insert_pwell,
                    (
                        row['DATEPRD'],
                        row['WELL_BORE_CODE'],
                        row['ON_STREAM_HRS'],
                        row['AVG_DOWNHOLE_PRESSURE'],
                        row['AVG_DOWNHOLE_TEMPERATURE'],
                        row['AVG_DP_TUBING'],
                        row['AVG_ANNULUS_PRESS'],
                        row['AVG_CHOKE_SIZE_P'],
                        row['AVG_CHOKE_UOM'],
                        row['AVG_WHP_P'],
                        row['AVG_WHT_P'],
                        row['DP_CHOKE_SIZE'],
                        row['BORE_OIL_VOL'],
                        row['BORE_GAS_VOL'],
                        row['BORE_WAT_VOL'],
                        row['BORE_WI_VOL'],
                        row['FLOW_KIND']
                    )
                )

        connection.commit()
    finally:
        logger.info("Connection closed")
        connection.close()


data_extracted=data_extract('volve_rate_20260106121832444_02.csv')

data_transformed=data_transform(data_extracted)

load(data_transformed)