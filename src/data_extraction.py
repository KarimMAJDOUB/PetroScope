import pandas as pd
import os

# On construit le chemin. 
# ATTENTION : J'ai ajouté ".csv" à la fin, vérifie si c'est bien le cas !
nom_fichier = "volve_rate_20251207161634.csv"

# Si tu es à la racine du projet :
chemin_complet = os.path.join('Data', nom_fichier)

print(f"Chargement de : {chemin_complet}")
def dataExtraction():
    """
    Transforme les données brutes en DataFrame Pandas propre.
    """
    # 1. Création du DataFrame
    df = pd.read_csv(chemin_complet)
        
        
    return df
df = dataExtraction()
print(df)
print(df.head())




