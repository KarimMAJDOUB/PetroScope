import pandas as pd
import os
from Database_creation import load

import logging

logger = logging.getLogger(__name__)

nom_fichier = "volve_rate_20260106121832444_02.csv"


# Si tu es à la racine du projet :
chemin_complet = os.path.join('Data', nom_fichier)


def dataExtraction(file_name):
    """
    Transforme les données brutes en DataFrame Pandas propre.
    """
    path_file=os.path.join('Data', nom_fichier)

    # 1. Création du DataFrame

    df = pd.read_csv(chemin_complet)
    load(df)
    logger.info("DataFrame created with success")

dataExtraction(nom_fichier)






