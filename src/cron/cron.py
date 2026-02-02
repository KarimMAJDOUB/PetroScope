import os
import sys
import time
import logging
from apscheduler.schedulers.blocking import BlockingScheduler

# --------------------------------------------------
# Logger configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Add src folder to sys.path for imports
# --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

print(src_dir)

# --------------------------------------------------
# ETL / Ingestion imports
# --------------------------------------------------
from ingestion.main_ingestion import ingestion
from etl.orchestration import etl_orchestration

# --------------------------------------------------
# Machine Learning model imports
# Each model must provide:
# - model function
# - parameter saving function
# - prediction saving function
from ml.random_forest import random_forest_model
from ml.xgboost import xgb_model
from ml.linear_regression import lr_model
from ml.model_load import model_load
from ml.output_generator import output_gen
# --------------------------------------------------
MODEL_INFOS = []

try:
    MODEL_INFOS.append({
        "name": "Random Forest",
        "available": True,
        "model_func": random_forest_model,
        "save_func": model_load
    })
except ImportError as e:
    logger.warning(f"Random Forest not available: {e}")

try:
    MODEL_INFOS.append({
        "name": "XGBoost",
        "available": True,
        "model_func": xgb_model,
        "save_func": model_load
    })
except ImportError as e:
    logger.warning(f"XGBoost not available: {e}")

try:
    MODEL_INFOS.append({
        "name": "Linear Regression",
        "available": True,
        "model_func": lr_model,
        "save_func": model_load
    })
except ImportError as e:
    logger.warning(f"XGBoost not available: {e}")
# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def get_latest_file():
    """
    Returns the latest file in the Raw_Data folder.
    If no file is found, returns None.
    """
    data_folder = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "Data")
    files = [f for f in os.listdir(data_folder) if f.endswith(('.xlsx','.xls','.csv','.json','.txt'))]
    if not files:
        return None
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(data_folder, f)))
    return latest_file


def run_etl(file_name: str) -> bool:
    """
    Run ingestion and ETL pipeline for the given file.
    Returns True if ETL completed successfully, False otherwise.
    """
    try:
        logger.info(f"Starting ingestion for {file_name}")
        ingestion(file_name)

        csv_file = f"{file_name.rsplit('.', 1)[0]}.csv"

        success = etl_orchestration(csv_file)

        if success:
            logger.info(f"ETL completed successfully for {csv_file}")
        else:
            logger.warning(f"ETL failed or skipped for {csv_file}")

        return success

    except Exception as e:
        logger.error(f"Error in run_etl for {csv_file}: {e}")
        return False

def run_model(model_info: dict):
    """
    Run a single ML model, save its parameters and predictions.

    Parameters:
        model_info (dict): Dictionary containing keys:
            - name (str): Model name for logging
            - available (bool): Flag if model can be run
            - model_func (callable): Function returning (params_df, predictions_df)
            - param_func (callable): Function to save model parameters to DB
            - save_func (callable): Function to save predictions to DB
    """
    name = model_info.get("name", "Unknown Model")
    
    if not model_info.get("available", False):
        logger.warning(f"{name} : Not available, skipping")
        return
    
    logger.info(f"Starting {name}...")
    start_time = time.time()

    try:
        df_preds = model_info["model_func"]()
        if df_preds is not None:
            model_info["save_func"](df_preds)

        logger.info(f"{name} : Completed successfully")

    except Exception as e:
        logger.error(f"{name} : Error during execution ({e})")

    duration = time.time() - start_time
    logger.info(f"{name} execution time: {duration:.2f} s")


# --------------------------------------------------
# Main cron function
# --------------------------------------------------
def cron():
    """
    Main cron function executed every 2 hours.
    Steps:
    1. Find the latest data file
    2. Run ingestion and ETL pipeline
    3. Execute all available ML models
    """
    try:
        file_name = get_latest_file()
        if not file_name:
            logger.warning("No file found, skipping this run")
            return
       
        #Run ETL
        run_etl(file_name)

        #Run ML models
        logger.info("Starting ML model computations")
        for model_info in MODEL_INFOS:
            run_model(model_info)

        logger.info("Starting output generation")
        output_gen()

    except Exception as e:
        logger.error(f"Error in cron execution: {e}")

# -----------------------------------------
# Scheduler
# -----------------------------------------
if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(cron, 'cron', second='*/10')
    logger.info("Scheduler started...")
    scheduler.start()
