import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def extract(file_name: str) -> pd.DataFrame:
    """
    Transforms raw data into a clean Pandas DataFrame.
    """
    path_file = os.path.join('Data', 'Data_Ingested', file_name)
    df = pd.read_csv(path_file)
    logger.info("DataFrame created with success")
