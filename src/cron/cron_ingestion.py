from apscheduler.schedulers.blocking import BlockingScheduler
from ingestion.main_ingestion import ingestion
from ETL.orchestration import ETL_orchestration

import apscheduler
import os
import logging

logger = logging.getLogger(__name__)

def get_latest_file():
    """
    Finds the most recent file in the 'Data' folder
    Returns None if no file is found
    """

    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(root_folder, "Data")
    
    files = [f for f in os.listdir(data_folder) if f.endswith(('.xlsx','.xls','.csv','.json','.txt'))]

    if not files:
        return None
    
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(data_folder, f)))
    return latest_file

def cron_ingestion():
    """
    Function called every 2 hours
    Retrieves the latest file, prepares it, then runs the full ETL pipeline
    """

    logger.info("Ingestion task started")
    try:
        file_name = get_latest_file()

        if file_name is None:
            logger.debug("No file found, skipping this run")
            return
        
        ingestion(file_name)
        logger.info(f"Ingestion completed for {file_name}")
    
        csv_file_name = f"{file_name.rsplit('.',1)[0]}_ready.csv"
        logger.info(f"Starting ETL pipeline for {csv_file_name}")

        ETL_orchestration(csv_file_name)
        logger.info(f"ETL pipeline completed for {csv_file_name}")

    except Exception as e:
        logger.error(f"Error during ingestion : {e}")

scheduler = BlockingScheduler()

scheduler.add_job(cron_ingestion, 'cron', hour='*/2')

scheduler.start()