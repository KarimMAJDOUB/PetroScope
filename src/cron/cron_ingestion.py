
from apscheduler.schedulers.blocking import BlockingScheduler
import apscheduler
import os
import sys
import logging
import time 
import sys

# >>> ADD: permettre les imports depuis AI_Model (sinon "No module named 'AI_Model'")
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
AI_MODEL_DIR = os.path.join(PROJECT_ROOT, "AI_Model")
if AI_MODEL_DIR not in sys.path:
    sys.path.append(AI_MODEL_DIR)

# Import de modèle (RF) et (LSTM) ---

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)                 
sys.path.append(src_dir)

# --- 1. IMPORT DU PLOTTER
try:
    from AI_Model.AI_Plotter import save_performance_plot
    PLOTTER_AVAILABLE = True
except ImportError as e:
    print(f"Attention: AI_Plotter.py non trouvé: {e}")
    PLOTTER_AVAILABLE = False

try:
    from AI_Model.RF_AI import RF_model, RF_param_load, model_load as model_save
    RF_AVAILABLE = True
except ImportError as e:
    print(f"Attention: RF_AI.py non trouvé ou erreur: {e}")
    RF_AVAILABLE = False
    

try:
    from AI_Model.LSTM_AI import LSTM_model, LSTM_param_load
    from AI_Model.Model_Load import model_load as LSTM_save_preds
    LSTM_AVAILABLE = True
except ImportError as e:
    print(f"Attention: LSTM_AI.py non trouvé ou erreur: {e}")
    LSTM_AVAILABLE = False

# >>> ADD: IA 3 — LR (ta méthode autonome)
try:
    from AI_Model.LR import (
        LR_model,                      # entraîne + renvoie (df_params_lr, df_preds_lr)
        save_params as LR_param_load,  # stocke métriques (DB si dispo, sinon CSV)
        save_predictions as LR_save_preds  # stocke MODEL_AI (DB si dispo, sinon CSV)
    )
    LR_AVAILABLE = True
except ImportError as e:
    print(f"Attention: LR.py non trouvé ou erreur: {e}")
    LR_AVAILABLE = False
# -----------------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.main_ingestion import ingestion
from ETL.orchestration import ETL_orchestration

logger = logging.getLogger(__name__)

# >>> ADD: infra copie des plots vers AI/AI_plots (sans modifier le plotter)
import shutil
from glob import glob

AI_DIR = os.path.join(PROJECT_ROOT, "AI")
AI_PLOTS_DIR = os.path.join(AI_DIR, "AI_plots")
os.makedirs(AI_PLOTS_DIR, exist_ok=True)

AI_MODEL_PLOTS_DIR = os.path.join(PROJECT_ROOT, "AI_Model", "AI_plots")
os.makedirs(AI_MODEL_PLOTS_DIR, exist_ok=True)

def _copy_new_plots(model_name: str):
    pattern = os.path.join(AI_MODEL_PLOTS_DIR, f"{model_name}_*.png")
    for src_path in glob(pattern):
        try:
            dest_path = os.path.join(AI_PLOTS_DIR, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
            logger.info(f"Plot {model_name} copié → {dest_path}")
        except Exception as e:
            logger.warning(f"Plot {model_name} : copie AI/AI_plots échouée ({e})")

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
        print("done")
        logger.info(f"Ingestion completed for {file_name}")
        
        logger.info("--- Démarrage calcul IA ---")
        
        # === IA 1 : RF ===
        t0_rf = time.time()
        if RF_AVAILABLE:
            try:
                df_params, df_preds = RF_model()
                if df_params is not None:
                    RF_param_load(df_params)
                    model_save(df_preds)
                    
                    # --- DESSIN RF ---
                    if PLOTTER_AVAILABLE:
                        save_performance_plot(df_preds, "RF")
                        # >>> ADD: copie vers AI/AI_plots
                        _copy_new_plots("RF")
                        
                logger.info(" Random Forest : Succès")
            except Exception as e:
                logger.error(f" Random Forest : Erreur ({e})")
        else:
            logger.warning("Random Forest : Fichier introuvable")
        t1_rf = time.time()
        
        # === IA 2 : LSTM ===
        t0_lstm = time.time()
        if LSTM_AVAILABLE:
            try:
                df_best, df_model_lstm, _, _ = LSTM_model()              
                if df_best is not None:
                    LSTM_param_load(df_best)
                    LSTM_save_preds(df_model_lstm)
                    
                    # --- DESSIN LSTM ---
                    if PLOTTER_AVAILABLE:
                        save_performance_plot(df_model_lstm, "LSTM")
                        # >>> ADD: copie vers AI/AI_plots
                        _copy_new_plots("LSTM")
                        
                logger.info(" LSTM : Succès")
            except Exception as e:
                logger.error(f" LSTM : Erreur ({e})")
        else:
            logger.info(" LSTM : Non disponible")
        t1_lstm = time.time()
        
        # >>> ADD: IA 3 — LR (exécution + stockage + plots + temps)
        t0_lr = time.time()
        if LR_AVAILABLE:
            try:
                df_params_lr, df_preds_lr = LR_model()
                if df_params_lr is not None:
                    LR_param_load(df_params_lr)
                    LR_save_preds(df_preds_lr)

                    if PLOTTER_AVAILABLE:
                        save_performance_plot(df_preds_lr, "LR")
                        _copy_new_plots("LR")

                logger.info(" LR : Succès")
            except Exception as e:
                logger.error(f" LR : Erreur ({e})")
        else:
            logger.warning("LR : Non disponible")
        t1_lr = time.time()
        
        # ======= TEMPS D'EXÉCUTION =======
        duration_rf = t1_rf - t0_rf
        duration_lstm = t1_lstm - t0_lstm
        logger.info(f" TEMPS RF   : {duration_rf:.4f} s")
        if LSTM_AVAILABLE:
            logger.info(f" TEMPS LSTM : {duration_lstm:.4f} s")

        # >>> ADD: temps LR
        duration_lr = t1_lr - t0_lr
        if LR_AVAILABLE:
            logger.info(f" TEMPS LR   : {duration_lr:.4f} s")

    except Exception as e:
        logger.error(f"Error in execution: {e}")

    
    csv_file_name = f"{file_name.rsplit('.',1)[0]}_ready.csv"
    logger.info(f"Starting ETL pipeline for {csv_file_name}")

    ETL_orchestration(csv_file_name)
    logger.info(f"ETL pipeline completed for {csv_file_name}")

scheduler = BlockingScheduler()

scheduler.add_job(cron_ingestion, 'cron', second='*/5')

scheduler = BlockingScheduler()

scheduler.add_job(cron_ingestion, 'cron', second='*/10')

scheduler.start()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    scheduler = BlockingScheduler()
    scheduler.add_job(cron_ingestion, 'cron', hour='*/2')
    
    logger.info("Scheduler démarré...")
    
    scheduler.start()
