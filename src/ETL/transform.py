import pandas as pd
import logging
from util.cleaning import fill_na

logger = logging.getLogger(__name__)

def transform(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Aucune donnée à transformer")

    # -------------------- DATE --------------------
    df['DAYTIME'] = pd.to_datetime(df['DATEPRD'], errors='coerce').dt.date
    df = df.dropna(subset=['DAYTIME'])
    logger.info(f"Lignes avec DAYTIME valide: {len(df)}")

    # -------------------- NUMERIC --------------------
    numeric_cols = [
        'ON_STREAM_HRS', 'AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE',
        'AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P',
        'AVG_WHP_P', 'AVG_WHT_P', 'DP_CHOKE_SIZE',
        'BORE_OIL_VOL', 'BORE_GAS_VOL', 'BORE_WAT_VOL', 'BORE_WI_VOL'
    ]

    # Garder seulement les colonnes existantes
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]

    if existing_numeric_cols:
        # Convertir en float pour l'imputation
        df[existing_numeric_cols] = df[existing_numeric_cols].astype(float)
        # Remplir les NaN
        df[existing_numeric_cols] = fill_na(df[existing_numeric_cols], n_neighbors=5)
        # Logger le résultat
        for col in existing_numeric_cols:
            nb_na = df[col].isnull().sum()
            if nb_na > 0:
                logger.warning(f"Colonne '{col}' contient encore {nb_na} NaN après fill_na")
            else:
                logger.info(f"Colonne '{col}' OK, plus de NaN")
    else:
        logger.warning("Aucune colonne numérique existante pour l'imputation")

    # -------------------- TEXTE --------------------
    text_cols = [
        'AVG_CHOKE_UOM', 'FLOW_KIND', 'NPD_WELL_BORE_CODE',
        'NPD_WELL_BORE_NAME', 'NPD_FIELD_CODE', 'NPD_FIELD_NAME',
        'WELL_TYPE', 'WELL_BORE_CODE'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).where(pd.notnull(df[col]), None)
        else:
            logger.warning(f"Colonne texte manquante : {col}")

    logger.info("Transformation réussie après remplissage des NaN")
    return df
