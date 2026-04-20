"""
build_dim_location.py
=====================
Builds the location dimension table (Dim_Location) for the star schema.

For each of the seven weather stations used in the project, this script
extracts the relevant metadata from the MeteoSwiss station metadata file
and enriches it with a human-readable canton name.

The resulting table is the spatial dimension that joins the weather and
electricity fact tables via the `canton` key.

Output: ../data_processed/dim_location.csv
"""

import pandas as pd
from pathlib import Path

# --- Configuration -----------------------------------------------------------

WEATHER_FOLDER     = Path("../data_raw/weather")
OUTPUT_FOLDER      = Path("../data_processed")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

STATIONS_META_FILE = WEATHER_FOLDER / "ogd-smn_meta_stations.csv"
OUTPUT_FILE        = OUTPUT_FOLDER / "dim_location.csv"

# One representative station per canton group (same mapping as in build_weather_dataset.py).
STATION_TO_CANTON = {
    "BER": "BE_JU",
    "GVE": "GE_VD",
    "DAV": "GR",
    "STG": "SG",
    "LUG": "TI",
    "EVO": "VS",
    "SMA": "ZH_SH",
}

# Human-readable canton names for reporting and dashboards.
CANTON_NAME_MAP = {
    "BE_JU": "Bern/Jura",
    "GE_VD": "Genf/Waadt",
    "GR":    "Graubünden",
    "SG":    "St. Gallen",
    "TI":    "Tessin",
    "VS":    "Wallis",
    "ZH_SH": "Zürich/Schaffhausen",
}

# Final column order of the exported dimension table.
ORDERED_COLUMNS = [
    "canton", "canton_name",
    "station_abbr", "station_name", "station_canton",
    "latitude", "longitude",
    "height_masl", "height_barometer_masl",
    "lv95_east", "lv95_north",
    "station_type", "station_dataowner",
]

# MeteoSwiss raw column -> our target column.
RENAME_MAP = {
    "station_name":                   "station_name",
    "station_canton":                 "station_canton",
    "station_height_masl":            "height_masl",
    "station_height_barometer_masl":  "height_barometer_masl",
    "station_coordinates_wgs84_lat":  "latitude",
    "station_coordinates_wgs84_lon":  "longitude",
    "station_coordinates_lv95_east":  "lv95_east",
    "station_coordinates_lv95_north": "lv95_north",
    "station_type_en":                "station_type",
    "station_dataowner":              "station_dataowner",
}

NUMERIC_COLUMNS = [
    "latitude", "longitude",
    "height_masl", "height_barometer_masl",
    "lv95_east", "lv95_north",
]


# --- Helpers -----------------------------------------------------------------

def load_station_metadata(file_path: Path) -> pd.DataFrame:
    """Load the MeteoSwiss station metadata file, falling back to latin1 if needed."""
    try:
        return pd.read_csv(
            file_path, sep=";", encoding="utf-8",
            na_values=["", " ", "NA", "nan"],
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            file_path, sep=";", encoding="latin1",
            na_values=["", " ", "NA", "nan"],
        )


def run_quality_checks(df: pd.DataFrame) -> None:
    """
    Guard against accidentally exporting a broken dimension:
      - exactly 7 rows (one per canton group)
      - unique canton and station keys
      - no missing values in mandatory fields
    """
    if len(df) != 7:
        raise ValueError(f"Expected 7 rows in Dim_Location, got {len(df)}.")

    if df["canton"].duplicated().any():
        raise ValueError(f"Duplicate canton values: {df.loc[df['canton'].duplicated(), 'canton'].tolist()}")

    if df["station_abbr"].duplicated().any():
        raise ValueError(f"Duplicate station_abbr values: {df.loc[df['station_abbr'].duplicated(), 'station_abbr'].tolist()}")

    required = ["canton", "canton_name", "station_abbr", "station_name", "latitude", "longitude"]
    missing  = df[required].isna().sum()
    if missing.any():
        raise ValueError(f"Missing values in required fields:\n{missing[missing > 0]}")


# --- Build -------------------------------------------------------------------

stations_df = load_station_metadata(STATIONS_META_FILE)

# Keep only the seven stations used in the project.
stations_df = stations_df[stations_df["station_abbr"].isin(STATION_TO_CANTON)].copy()

# Attach canton key and human-readable name.
stations_df["canton"]      = stations_df["station_abbr"].map(STATION_TO_CANTON)
stations_df["canton_name"] = stations_df["canton"].map(CANTON_NAME_MAP)

# Normalise column names and select the final schema.
stations_df = stations_df.rename(columns=RENAME_MAP)[ORDERED_COLUMNS].copy()

# Ensure numeric columns are actually numeric (coordinates, elevation, etc.).
for col in NUMERIC_COLUMNS:
    stations_df[col] = pd.to_numeric(stations_df[col], errors="coerce")

stations_df = stations_df.sort_values("canton").reset_index(drop=True)

run_quality_checks(stations_df)

# --- Export ------------------------------------------------------------------

stations_df.to_csv(
    OUTPUT_FILE,
    index=False, sep=";", decimal=",",
    encoding="utf-8-sig",  # BOM so Power BI / Excel render umlauts correctly
)

print(f"Dim_Location: {len(stations_df)} rows, {stations_df.shape[1]} columns")
print(f"Saved: {OUTPUT_FILE}")
