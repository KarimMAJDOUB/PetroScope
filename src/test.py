from ML.cleaning import Cleaner
from config.sql_config import sql_settings


if __name__ == "__main__":
    cleaner = Cleaner(sql_settings)
    pwell = cleaner.pwell_data.drop(columns=["DAYTIME"])
    pwell = cleaner.encode(
        pwell,
        text_columns=["WELL_BORE_CODE", "AVG_CHOKE_UOM", "FLOW_KIND"],
        target="BORE_OIL_VOL"
    )
    pwell = cleaner.fill_na(pwell)
    end = 1