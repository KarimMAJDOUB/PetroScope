from apscheduler.schedulers.blocking import BlockingScheduler
import apscheduler
import os
import sys
import logging
import time 
import sys

# Import de modèle (RF) ---

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # On importe tes fonctions depuis ton fichier RF_model.py
    from RF_model import RF_model, RF_param_load, model_load as model_save
    RF_AVAILABLE = True
except ImportError as e:
    print(f"Attention: RF_model.py non trouvé ou erreur: {e}")
    RF_AVAILABLE = False
# -----------------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.main_ingestion import ingestion
from ETL.orchestration import ETL_orchestration

logger = logging.getLogger(__name__)

def get_latest_file():
    """
    Finds the most recent file in the 'Data' folder
    Returns None if no file is found
    """
    root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(root_folder, "..", "Data", "Raw_Data")

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
    print("ingestion started")
    logger.info("Ingestion task started")
    try:
        file_name = get_latest_file()
        
        if file_name is None:
            logger.debug("No file found, skipping this run")
            return
        
        ingestion(file_name)
        logger.info(f"Ingestion completed for {file_name}")
        
        logger.info("--- Démarrage calcul IA ---")
        
        t0_rf = time.time()
        if RF_AVAILABLE:
            try:
                df_params, df_preds = RF_model()
                if df_params is not None:
                    RF_param_load(df_params)
                    model_save(df_preds)
                logger.info(" Random Forest : Succès")
            except Exception as e:
                logger.error(f" Random Forest : Erreur ({e})")
        else:
            logger.warning("Random Forest : Fichier introuvable")
        t1_rf = time.time()
        
        duration_rf = t1_rf - t0_rf
        logger.info(f" TEMPS DE CALCUL - Random Forest : {duration_rf:.4f} secondes")
        
        logger.info("--- Fin calcul IA ---")

    except Exception as e:
        logger.error(f"Error in execution: {e}")

    
    csv_file_name = f"{file_name.rsplit('.',1)[0]}_ready.csv"
    logger.info(f"Starting ETL pipeline for {csv_file_name}")

    ETL_orchestration(csv_file_name)
    logger.info(f"ETL pipeline completed for {csv_file_name}")

scheduler = BlockingScheduler()

scheduler.add_job(cron_ingestion, 'cron', second='*/5')

scheduler.start()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    scheduler = BlockingScheduler()
    scheduler.add_job(cron_ingestion, 'cron', hour='*/2')
    
    logger.info("Scheduler démarré...")
    
    scheduler.start()
