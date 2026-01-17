# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:56:38 2026

@author: RHEDDAD
"""
import pandas as pd 
import os
import pymysql
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def data_extract(file_name):
    """
    Transforme les données brutes en DataFrame Pandas propre.
    """
    try:
        path_file=os.path.join('Data', file_name)
        df = pd.DataFrame(pd.read_csv(path_file))
        logger.info(f"Extraction réussie: {file_name} (lignes: {len(df)})")
        return df
    except FileNotFoundError:
        logger.error(f"Fichier introuvable: {file_name}")
    except pd.errors.EmptyDataError:
        logger.error(f"Fichier vide: {file_name}")
    except pd.errors.ParserError as e:
        logger.error(f"Erreur parsing CSV {file_name}: {e}")
    except Exception as e:
        logger.error(f"Erreur inconnue lors de l'extraction: {e}")


def data_transform(df):
    if df is None:
        logger.error("Aucune donnée à transformer")
        return None
    try:
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

        try:
            df[numeric_cols] = (
                df[numeric_cols]
                .astype(object)
                .where(pd.notnull(df[numeric_cols]), None)
            )
        except KeyError as e:
            logger.error(f"Colonne(s) numérique(s) manquante(s): {e}")
        except Exception as e:
            logger.error(f"Erreur conversion colonnes numériques: {e}")

        text_cols = [
            'AVG_CHOKE_UOM',
            'FLOW_KIND',
            'NPD_WELL_BORE_CODE',
            'NPD_WELL_BORE_NAME',
            'NPD_FIELD_CODE',
            'NPD_FIELD_NAME',
            'WELL_TYPE'
        ]

        try:
            df[text_cols] = df[text_cols].where(pd.notnull(df[text_cols]), None)
        except KeyError as e:
            logger.error(f"Colonne(s) texte manquante(s): {e}")
        except Exception as e:
            logger.error(f"Erreur conversion colonnes texte: {e}")

        logger.info("Transformation réussie")
        return df

    except Exception as e:
        logger.error(f"Erreur inconnue lors de la transformation: {e}")
        return df


def load(df) -> None:
    """
    Loads data into a MySQL database using pymysql.
    """
    db_user = "root"
    db_password = "Jujuaurugby10!"
    db_name = "Petroscope"

    try:
        connection = pymysql.connect(
            user=db_user,
            password=db_password,
            database=db_name,
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.info(f"Connexion MySQL réussie")
    except pymysql.err.OperationalError as e:
        logger.error(f"Connexion MySQL échouée: {e}")
        return
    except Exception as e:
        logger.error(f"Erreur inconnue lors de la connexion MySQL: {e}")
        return

    try:
        with connection.cursor() as cursor:
            # Table OBJECTS
            try:
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
            except Exception as e:
                logger.error(f"Erreur création table OBJECTS: {e}")

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
                            row['DATEPRD'],
                            row['WELL_BORE_CODE'],
                            row['NPD_WELL_BORE_CODE'],
                            row['NPD_WELL_BORE_NAME'],
                            row['NPD_FIELD_CODE'],
                            row['NPD_FIELD_NAME'],
                            row['WELL_TYPE']
                        )
                    )
                except KeyError as e:
                    logger.error(f"Colonne manquante OBJECTS (ligne {i}): {e}")
                except Exception as e:
                    logger.error(f"Erreur insertion OBJECTS (ligne {i}): {e}")

            # Table PWELL_DATA
            try:
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
                logger.info("Table PWELL_DATA prête")
            except Exception as e:
                logger.error(f"Erreur création table PWELL_DATA: {e}")

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
                except KeyError as e:
                    logger.error(f"Colonne manquante PWELL_DATA (ligne {i}): {e}")
                except Exception as e:
                    logger.error(f"Erreur insertion PWELL_DATA (ligne {i}): {e}")

        connection.commit()
        logger.info("Chargement MySQL terminé")
    except Exception as e:
        logger.error(f"Erreur lors de la création des tables ou insertion: {e}")
    finally:
        try:
            connection.close()
            logger.info("Connexion MySQL fermée")
        except Exception as e:
            logger.error(f"Erreur fermeture connexion MySQL: {e}")


data_extracted = data_extract('volve_rate_20260106121832444_02.csv')
data_transformed = data_transform(data_extracted)
load(data_transformed)

print("Done")
