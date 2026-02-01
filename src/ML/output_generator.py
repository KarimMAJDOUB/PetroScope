import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

logging.basicConfig(
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

from config.sql_config import sql_settings

from data_access.sql_reader import call_data_sql

def output_gen():
    query="""select * from model_ai
            """
    df=call_data_sql(query)

    name_write = f"ml_predictions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv"

    current_file=os.path.dirname(os.path.abspath(__file__))
    path_write = os.path.join(current_file,'..','..', "output", name_write)
    path_write=os.path.abspath(path_write)

    df.to_csv(path_write, index=False)
    return logging.info(f"Output csv généré {name_write}")
