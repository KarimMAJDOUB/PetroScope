import pandas as pd

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms numeric and text columns, replaces NaN with None.
    """
    if df is None:
        raise ValueError("Received None instead of a DataFrame")
    
    numeric_cols = [
        'ON_STREAM_HRS','AVG_DOWNHOLE_PRESSURE','AVG_DOWNHOLE_TEMPERATURE',
        'AVG_DP_TUBING','AVG_ANNULUS_PRESS','AVG_CHOKE_SIZE_P','AVG_WHP_P',
        'AVG_WHT_P','DP_CHOKE_SIZE','BORE_OIL_VOL','BORE_GAS_VOL',
        'BORE_WAT_VOL','BORE_WI_VOL'
    ]

    #Keep only existing columns
    existing_cols = [col for col in numeric_cols if col in df.columns]

    #Replace NaN with None
    df[existing_cols] = df[existing_cols].astype(object).where(pd.notna(df[existing_cols]), None)

    return df
