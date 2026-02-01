import pandas as pd
import logging

from sqlalchemy import create_engine
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from category_encoders.cat_boost import CatBoostEncoder

from config.sql_config import SQLConfig, sql_settings

def create_sql(sql_settings: SQLConfig):
    return create_engine (
        f"mysql+mysqlconnector://"
        f"{sql_settings.user}"
        f":{sql_settings.password}"
        f"/{sql_settings.database}"   
    )

def objects(sql) -> pd.DataFrame:
    query = "SELECT * FROM objects;"
    return pd.read_sql(query, con=sql)


def pwell_data(sql) -> pd.DataFrame:
    query = "SELECT * FROM pwell_data;"
    return pd.read_sql(query, con=sql)

def encode(
    df: pd.DataFrame,
    text_columns: list[str],
    target: str
) -> pd.DataFrame:
    encoder = CatBoostEncoder(cols=text_columns, return_df=True)
    return encoder.fit_transform(df, df[target])

def fill_na(df: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
    """
    Remplit les NaN dans les colonnes numériques avec KNNImputer
    tout en conservant exactement les mêmes colonnes et index,
    même si certaines colonnes sont entièrement vides.
    """
    import logging
    import pandas as pd
    from sklearn.impute import KNNImputer

    logger = logging.getLogger(__name__)

    df_float = df.astype(float)

    # Identifier colonnes entièrement vides
    empty_cols = [col for col in df_float.columns if df_float[col].isnull().all()]

    # Colonnes partiellement remplies pour KNN
    partial_cols = [col for col in df_float.columns if col not in empty_cols]

    if partial_cols:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_float[partial_cols] = imputer.fit_transform(df_float[partial_cols])

    # Les colonnes entièrement vides restent NaN
    result = pd_float = df_float.copy()

    return result

def normalize(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    scaler=StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled, scaler