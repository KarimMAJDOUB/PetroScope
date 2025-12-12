import pandas as pd
import json 
import os
import csv

def ingestion(file_name):
    """
    Docstring for ingestion
    
    :param file_name
    :Import the last file of the file Data
    
    Supported formats : csv, txt, xlsx, xls, json
    """

    #path_read='Petroscope/Data/{file_name}'
    root_file=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(root_file)
    
    name_write=f"{file_name.rsplit(".", 1)[0]}.csv"

    path_read=os.path.join(root_file, "Data", file_name)
    path_write=os.path.join(root_file, "Data", "Raw_Data",name_write)
    if file_name[-5:]==".xlsx" or file_name[-4:]==".xls":
        raw_file=pd.read_excel(path_read)
        raw_file.to_csv(path_write, index=False)
    elif file_name[-5:]==".json":
        with open(path_read,'r', encoding="utf-8") as json_file:
            data_json=json_file.read().split()
        
        if isinstance(data_json,list):
            df= pd.DataFrame(data_json)
        elif isinstance(data_json, dict):
            df= pd.DataFrame.from_dict(data_json)

        df.to_csv(path_write,index=False)
            
    else:
        with open(path_read,'r') as raw_file:
            raw_file=raw_file.read()
        with open(path_write,'w') as net_file:
            net_file.write(raw_file)
    
    os.remove(path_read)

file_name="volve_rate_20251207161634.xlsx" #To be changed with the CRON
ingestion(file_name)
