"""
ETL Orchestration

Ce fichier sert à lancer le pipeline ETL complet :
extract -> transform -> load.

Pour exécuter le pipeline, lancer le code depuis ce fichier.
"""

from ETL.extract import extract
from ETL.transform import transform
from ETL.load import load
import os
import logging

logger = logging.getLogger(__name__)

def ETL_orchestration(file_name: str):
    """
    Orchestration ETL : extract -> transform -> load
    """
    logger.info(f"ETL started for file: {file_name}")

    #Checking that the file exists
    path_file = os.path.join('Data', 'Data_Ingested', file_name)
    if not os.path.exists(path_file):
        logger.error(f"File not found: {path_file}")
        return None

    #Extraction
    df = extract(file_name)
    logger.info(f"Extraction done: {len(df)} rows loaded")

    #Transformation
    df_transformed = transform(df)
    logger.info(f"Transformation done: {len(df_transformed)} rows processed")

    #Loading
    load(df_transformed)
    logger.info(f"Loading done: data from {file_name} inserted into DB")

    return df_transformed