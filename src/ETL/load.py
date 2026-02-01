import pandas as pd
import pymysql
import logging
from config.sql_config import sql_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("DataFrame vide ou None, impossible de charger en base")

    df = df.where(pd.notnull(df), None)

    # Supprimer les colonnes entièrement vides pour éviter les erreurs MySQL
    for col in df.columns:
        if df[col].isnull().all():
            df = df.drop(columns=[col])
            logger.warning(f"Colonne '{col}' entièrement vide, supprimée avant insertion")

    try:
        connection = pymysql.connect(
            user=sql_settings.user,
            password=sql_settings.password,
            database=sql_settings.database,
            cursorclass=sql_settings.cursorclass
        )
        logger.info("Connexion MySQL réussie")
    except Exception as e:
        raise ConnectionError(f"Impossible de se connecter à MySQL : {e}")

    try:
        with connection.cursor() as cursor:
            # -------------------- Table OBJECTS --------------------
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
            logger.info("Table OBJECTS prête")

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

            for i, row in df.iterrows():
                try:
                    cursor.execute(
                        insert_objects,
                        (
                            row['DAYTIME'],
                            row['WELL_BORE_CODE'],
                            row['NPD_WELL_BORE_CODE'],
                            row['NPD_WELL_BORE_NAME'],
                            row['NPD_FIELD_CODE'],
                            row['NPD_FIELD_NAME'],
                            row['WELL_TYPE']
                        )
                    )
                except KeyError as e:
                    raise KeyError(f"Colonne manquante OBJECTS (ligne {i}) : {e}")
                except Exception as e:
                    raise RuntimeError(f"Erreur insertion OBJECTS (ligne {i}) : {e}")

            # -------------------- Table PWELL_DATA --------------------
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS PWELL_DATA (
                    DAYTIME DATE,
                    WELL_BORE_CODE VARCHAR(50),
                    ON_STREAM_HRS DECIMAL(10,2) NULL,
                    AVG_DOWNHOLE_PRESSURE DECIMAL(10,2) NULL,
                    AVG_DOWNHOLE_TEMPERATURE DECIMAL(10,2) NULL,
                    AVG_DP_TUBING DECIMAL(10,2) NULL,
                    AVG_ANNULUS_PRESS DECIMAL(10,2) NULL,
                    AVG_CHOKE_SIZE_P DECIMAL(10,2) NULL,
                    AVG_CHOKE_UOM VARCHAR(20),
                    AVG_WHP_P DECIMAL(10,2) NULL,
                    AVG_WHT_P DECIMAL(10,2) NULL,
                    DP_CHOKE_SIZE DECIMAL(10,2) NULL,
                    BORE_OIL_VOL DECIMAL(14,3) NULL,
                    BORE_GAS_VOL DECIMAL(14,3) NULL,
                    BORE_WAT_VOL DECIMAL(14,3) NULL,
                    BORE_WI_VOL DECIMAL(14,3) NULL,
                    FLOW_KIND VARCHAR(50),
                    PRIMARY KEY (DAYTIME, WELL_BORE_CODE)
                );
            """)
            logger.info("Table PWELL_DATA prête")

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

            for i, row in df.iterrows():
                try:
                    cursor.execute(
                        insert_pwell,
                        (
                            row['DAYTIME'],
                            row['WELL_BORE_CODE'],
                            row.get('ON_STREAM_HRS'),
                            row.get('AVG_DOWNHOLE_PRESSURE'),
                            row.get('AVG_DOWNHOLE_TEMPERATURE'),
                            row.get('AVG_DP_TUBING'),
                            row.get('AVG_ANNULUS_PRESS'),
                            row.get('AVG_CHOKE_SIZE_P'),
                            row.get('AVG_CHOKE_UOM'),
                            row.get('AVG_WHP_P'),
                            row.get('AVG_WHT_P'),
                            row.get('DP_CHOKE_SIZE'),
                            row.get('BORE_OIL_VOL'),
                            row.get('BORE_GAS_VOL'),
                            row.get('BORE_WAT_VOL'),
                            row.get('BORE_WI_VOL'),
                            row.get('FLOW_KIND')
                        )
                    )
                except KeyError as e:
                    raise KeyError(f"Colonne manquante PWELL_DATA (ligne {i}) : {e}")
                except Exception as e:
                    raise RuntimeError(f"Erreur insertion PWELL_DATA (ligne {i}) : {e}")

        connection.commit()
        logger.info("Chargement MySQL terminé")

    finally:
        try:
            connection.close()
            logger.info("Connexion MySQL fermée")
        except Exception as e:
            logger.warning(f"Erreur fermeture connexion MySQL: {e}")
