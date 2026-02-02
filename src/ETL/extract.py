import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract(file_name):
    """
    Transforme les données brutes en DataFrame Pandas propre.
    """
    current_file=os.path.dirname(os.path.abspath(__file__))

    path_file = os.path.join(current_file,'..', '..','Data','Data_Ingested', file_name)
    path_file=os.path.abspath(path_file)

    try:
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
