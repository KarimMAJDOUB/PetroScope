import pandas as pd

def ingestion(file_name):
    """
    Docstring for ingestion
    
    :param file_name
    :Import the last file of the file Data
    """

    raw_file_name=file_name.rsplit(".", 1)[0]

    path_read=f'Petroscope/Data/{file_name}'
    path_write=f"Petroscope/Raw_Data/{raw_file_name}.csv"
    if file_name[-5:]==".xlsx":
        raw_file=pd.read_excel(path_read)
        raw_file.to_csv(path_write, index=False)
    else:
        with open(path_read,'r') as raw_file:
            raw_file=raw_file.read()
        with open(path_write,'w') as net_file:
            net_file.write(raw_file)

file_name="volve_rate_20251207161634.xlsx" #To be changed with the CRON

ingestion(file_name)
