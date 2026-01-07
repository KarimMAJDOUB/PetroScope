#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 16:29:10 2025

@author: mdbouras
"""

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

    name_write = f"{file_name.rsplit('.', 1)[0]}.csv"

    current_file=os.path.dirname(os.path.abspath(__file__))
    path_read = os.path.join(current_file,'..', "Data", file_name)
    path_read=os.path.abspath(path_read)
    print(path_read)
    path_write = os.path.join(root_file, "Data", name_write)

    try:
        # NEW: check if output file already exists
        if os.path.exists(path_write):
            raise ValueError(f"Not acceptable: the file '{name_write}' already exists in Raw_Data")

        # Excel files
        if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            if os.path.getsize(path_read) == 0:
                raise ValueError(f"Not acceptable: the file '{file_name}' is empty")
            raw_file = pd.read_excel(path_read)
            raw_file.to_csv(path_write, index=False)

        # JSON files
        if file_name.endswith(".json"):
            if os.path.getsize(path_read) == 0:
                raise ValueError(f"Not acceptable: the file '{file_name}' is empty")
            with open(path_read, 'r', encoding="utf-8") as json_file:
                data_json = json.load(json_file)

            if isinstance(data_json, list):
                df = pd.DataFrame(data_json)
            elif isinstance(data_json, dict):
                df = pd.DataFrame.from_dict(data_json)
            else:
                raise ValueError("Unsupported JSON structure")

            df.to_csv(path_write, index=False)

        # CSV or TXT files
        if file_name.endswith(".csv") or file_name.endswith(".txt"):
            if os.path.getsize(path_read) == 0:
                raise ValueError(f"Not acceptable: the file '{file_name}' is empty")
            with open(path_read, 'r') as raw_file:
                raw_file = raw_file.read()
            with open(path_write, 'w') as net_file:
                net_file.write(raw_file)

        # If none of the above matched
        else:
            raise ValueError(
                f"Unrecognized input type for '{file_name}'. "
                "Supported types are: .csv, .txt, .xlsx, .xls, .json"
            )
    
    except Exception as e:
        print(f"[ERROR] Data ingestion failed for '{file_name}'. Details: {e}")
    os.remove(path_read)

