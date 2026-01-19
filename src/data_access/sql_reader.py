import pandas as pd
import logging
from sqlalchemy import create_engine
from config.sql_config import sql_settings

logger = logging.getLogger(__name__)

def call_data_sql(sql_query: str) -> pd.DataFrame:
    """
    Docstring for Call_data_sql
<<<<<<< HEAD
=======

>>>>>>> origin/master
    Read data in the SQL Database according to the sql query and write it directly in a pandas Dataframe.
    
    :param sql_query: Write the query you want to use in the database
    """
    try:
        engine = create_engine(
            f"mysql+mysqlconnector://"
            f"{sql_settings.user}:"
            f"{sql_settings.password}"
            f"@localhost/{sql_settings.database}"
        )
        df = pd.read_sql(sql_query, engine)
        return df
<<<<<<< HEAD

=======
        
>>>>>>> origin/master
    except Exception as e:
        logger.error(f"SQL query failed: {e}")
        raise