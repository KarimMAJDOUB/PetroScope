from apscheduler.schedulers.blocking import BlockingScheduler
from main_ingestion import ingestion

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
    It retrieves the latest file and sends it to the ingestion function
    If no file is found, it simply skips the execution
    """

    logger.info("Ingestion task started")
    try:
        file_name = get_latest_file()

        if file_name is None:
            logger.debug("No file found, skipping this run")
            return
        
        ingestion(file_name)
        print("done")
        logger.info(f"Ingestion completed for {file_name}")
    except Exception as e:
        logger.error(f"Error during ingestion : {e}")


scheduler = BlockingScheduler()

scheduler.add_job(cron_ingestion, 'cron', second='*/10')

scheduler.start()