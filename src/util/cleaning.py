import pandas as pd

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
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(df)
    return pd.DataFrame(imputed, columns=df.columns)

def normalize(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
