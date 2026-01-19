import pandas as pd
from sqlalchemy import create_engine

def Call_data_sql(sql_query):
    """
    Docstring for Call_data_sql

    Read data in the SQL Database according to the sql query and write it directly in a pandas Dataframe.
    
    :param sql_query: Write the query you want to use in the database
    """
    db_user = "root"
    db_password = "Jujuaurugby10!"
    db_name = "Petroscope"

    engine = create_engine(
        f"mysql+mysqlconnector://{db_user}:{db_password}@localhost/{db_name}"
    )

    df = pd.read_sql(sql_query, engine)

    return df