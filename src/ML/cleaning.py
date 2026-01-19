import pandas as pd

from functools import cached_property
from sqlalchemy import create_engine
from sklearn.impute import KNNImputer
from category_encoders.cat_boost import CatBoostEncoder

from config.sql_config import SQLConfig, sql_settings

class Cleaner:

    def __init__(self, sql_settings: SQLConfig, n_neighbors: int = 3):
        self.sql = create_engine(
            f"mysql+mysqlconnector://"
            f"{sql_settings.user}"
            f":{sql_settings.password}"
            f"@{sql_settings.host}"
            f":{sql_settings.port}"
            f"/{sql_settings.database}"   
        )
        self.n_neighbors = n_neighbors
        
    @cached_property
    def objects(self) -> pd.DataFrame:
        query = "SELECT * FROM objects;"
        return pd.read_sql(query, con=self.sql)
    
    @cached_property
    def pwell_data(self) -> pd.DataFrame:
        query = "SELECT * FROM pwell_data;"
        return pd.read_sql(query, con=self.sql)
    
    def encode(self,
        df: pd.DataFrame,
        text_columns: list[str],
        target: str
    ) -> pd.DataFrame:
        encoder = CatBoostEncoder(cols=text_columns, return_df=True)
        return encoder.fit_transform(df, df[target])
    
    def fill_na(self, df: pd.DataFrame) -> pd.DataFrame:
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        imputed = imputer.fit_transform(df)
        cols = df.columns[~df.isna().all()]
        return pd.DataFrame(
            imputed,
            columns=cols
        )
    

