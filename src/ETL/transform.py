import pandas as pd

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms numeric and text columns, replaces NaN with None.
    """
    numeric_cols = [
                'ON_STREAM_HRS',
                'AVG_DOWNHOLE_PRESSURE',
                'AVG_DOWNHOLE_TEMPERATURE',
                'AVG_DP_TUBING',
                'AVG_ANNULUS_PRESS',
                'AVG_CHOKE_SIZE_P',
                'AVG_WHP_P',
                'AVG_WHT_P',
                'DP_CHOKE_SIZE',
                'BORE_OIL_VOL',
                'BORE_GAS_VOL',
                'BORE_WAT_VOL',
                'BORE_WI_VOL'
            ]

    df[numeric_cols] = (
        df[numeric_cols]
        .astype(object)
        .where(pd.notnull(df[numeric_cols]), None)
    )

    text_cols = [
        'AVG_CHOKE_UOM',
        'FLOW_KIND',
        'NPD_WELL_BORE_CODE',
        'NPD_WELL_BORE_NAME',
        'NPD_FIELD_CODE',
        'NPD_FIELD_NAME',
        'WELL_TYPE'
    ]

    df[text_cols] = df[text_cols].where(pd.notnull(df[text_cols]), None)
    return df