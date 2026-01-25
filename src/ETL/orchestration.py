"""
ETL Orchestration

Ce fichier sert à lancer le pipeline ETL complet :
extract -> transform -> load.

Pour exécuter le pipeline, lancer le code depuis ce fichier.
"""

import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.sql_config import sql_settings
from etl.extract import extract
from etl.transform import transform
from etl.load import load

logger = logging.getLogger(__name__)

def etl_orchestration(file_name: str) -> bool:
    """
    Orchestrate ETL process: extract -> transform -> load.
    Returns True if ETL succeeded, False otherwise.
    """
    try:
        #Construct full path to the ingested file
        path_file = os.path.join('Data', 'Data_Ingested', file_name)
        if not os.path.exists(path_file):
            logger.error(f"File not found: {path_file}")
            return False

        #Extraction
        df = extract(file_name)
        logger.info(f"Extraction completed: {len(df)} rows loaded")

        #Transformation
        df_transformed = transform(df)
        logger.info(f"Transformation completed: {len(df_transformed)} rows processed")

        #Loading
        load(df_transformed)
        logger.info(f"Loading completed: data from {file_name} inserted into DB")

        return True

    except Exception as e:
        logger.error(f"ETL orchestration failed for {file_name}: {e}")
        return False