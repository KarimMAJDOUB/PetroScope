import pandas as pd
import json 
import os
import csv
import logging
import getpass

# Setup logging
logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Identify user
user = getpass.getuser()
logging.info(f"Script lancé par l'utilisateur : {user}")

def ingestion(file_name):
    """
    Docstring for ingestion
    
    :param file_name
    :Import the last file of the file Data
    
    Supported formats : csv, txt, xlsx, xls, json
    """

    name_write = f"{file_name.rsplit('.', 1)[0]}.csv"

    current_file=os.path.dirname(os.path.abspath(__file__))
    path_read = os.path.join(current_file,'..', '..', "Data", file_name)
    path_read=os.path.abspath(path_read)
    path_write = os.path.join(current_file,'..','..', "Data","Data_Ingested", name_write)
    
    df = None
    logging.info(f"Tentative d'ingestion du fichier : {file_name}")

    try:
        if os.path.exists(path_write):
            raise ValueError(f"Le fichier '{name_write}' existe déjà. Ingestion annulée.")

        # Excel files
        if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            logging.info("Format fichier détecté : Excel")
            if os.path.getsize(path_read) == 0:
                raise ValueError(f"Le fichier '{file_name}' est vide.")
            raw_file = pd.read_excel(path_read)
            raw_file.to_csv(path_write, index=False)
            logging.info(f"Fichier Excel converti et sauvegardé en : {name_write}")

        # JSON files
        elif file_name.endswith(".json"):
            logging.info("Format fichier détecté : JSON")
            if os.path.getsize(path_read) == 0:
                raise ValueError(f"Le fichier '{file_name}' est vide.")
            with open(path_read, 'r', encoding="utf-8") as json_file:
                data_json = json.load(json_file)

            if isinstance(data_json, list):
                df = pd.DataFrame(data_json)
            elif isinstance(data_json, dict):
                df = pd.DataFrame.from_dict(data_json)
            else:
                raise ValueError("Structure JSON non supportée")

            df.to_csv(path_write, index=False)
            logging.info(f"Fichier JSON converti et sauvegardé en : {name_write}")

        # CSV or TXT files
        elif file_name.endswith(".csv") or file_name.endswith(".txt"):
            logging.info("Format fichier détecté : CSV ou TXT")
            if os.path.getsize(path_read) == 0:
                raise ValueError(f"Le fichier '{file_name}' est vide.")
            with open(path_read, 'r') as raw_file:
                raw_file = raw_file.read()
            with open(path_write, 'w') as net_file:
                net_file.write(raw_file)
            logging.info(f"Fichier texte copié vers : {name_write}")

        else:
            raise ValueError(
                f"Type de fichier non reconnu pour '{file_name}'. "
                "Types supportés : .csv, .txt, .xlsx, .xls, .json"
            )

    except Exception as e:
        logging.error(f"Échec de l'ingestion du fichier '{file_name}'. Détails : {e}")
    else:
        logging.info(f"Ingestion de '{file_name}' terminée avec succès.")
    finally:
        if os.path.exists(path_read):
            os.remove(path_read)
            logging.info(f"Fichier original supprimé : {file_name}")

root_file = os.path.dirname(os.path.abspath(__file__))
