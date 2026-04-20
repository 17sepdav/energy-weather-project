"""
build_weather_dataset.py
========================
Builds the hourly weather dataset from the raw MeteoSwiss station CSV files.

Pipeline
--------
1. Discover all MeteoSwiss timeseries CSVs in ../data_raw/weather/
   (metadata files starting with "ogd-smn_meta_" are excluded)
2. For each file: parse timestamps, keep selected variables, restrict to the
   analysis window (2015-2025), map the station to its canton group,
   rename technical MeteoSwiss codes to readable names
3. Concatenate all stations, sort, and run quality checks
4. Export a single hourly CSV covering all seven regions

Output: ../data_processed/weather_dataset.csv
"""

import pandas as pd
from pathlib import Path

# --- Configuration -----------------------------------------------------------

WEATHER_FOLDER = Path("../data_raw/weather")
OUTPUT_FOLDER  = Path("../data_processed")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE    = OUTPUT_FOLDER / "weather_dataset.csv"

# Analysis window — matches the electricity dataset so both can be joined 1:1 later.
ANALYSIS_START = pd.Timestamp("2015-01-01 00:00:00")
ANALYSIS_END   = pd.Timestamp("2025-12-31 23:00:00")

# MeteoSwiss variable codes we actually need (all hourly measurements).
SELECTED_WEATHER_FEATURES = [
    "tre200h0",  # air temperature at 2m
    "rre150h0",  # precipitation
    "ure200h0",  # relative humidity
    "gre000h0",  # global radiation
    "sre000h0",  # sunshine duration
    "fkl010h0",  # wind speed
    "dkl010h0",  # wind direction
    "prestah0",  # station pressure
    "pp0qnhh0",  # pressure QNH (sea-level equivalent)
    "pva200h0",  # vapour pressure
    "tde200h0",  # dew point
]

# Maps a station abbreviation to the canton group it represents in our model.
# One representative station per region keeps the join to the electricity dataset simple.
STATION_TO_CANTON = {
    "BER": "BE_JU",
    "GVE": "GE_VD",
    "DAV": "GR",
    "STG": "SG",
    "LUG": "TI",
    "EVO": "VS",
    "SMA": "ZH_SH",
}
EXPECTED_CANTONS = sorted(STATION_TO_CANTON.values())

# Rename technical MeteoSwiss codes to readable column names for downstream work.
RENAME_MAP = {
    "reference_timestamp": "timestamp",
    "tre200h0": "temperature_2m",
    "rre150h0": "precipitation",
    "ure200h0": "humidity_rel",
    "gre000h0": "global_radiation",
    "sre000h0": "sunshine_duration",
    "fkl010h0": "wind_speed",
    "dkl010h0": "wind_direction",
    "prestah0": "pressure_station",
    "pp0qnhh0": "pressure_qnh",
    "pva200h0": "vapour_pressure",
    "tde200h0": "dew_point",
}

ORDERED_COLUMNS = [
    "timestamp", "canton", "station_abbr",
    "temperature_2m", "precipitation", "humidity_rel",
    "global_radiation", "sunshine_duration",
    "wind_speed", "wind_direction",
    "pressure_station", "pressure_qnh",
    "vapour_pressure", "dew_point",
]
NUMERIC_COLUMNS = [c for c in ORDERED_COLUMNS if c not in {"timestamp", "canton", "station_abbr"}]


# --- Helpers -----------------------------------------------------------------

def transform_weather_file(file_name: str) -> pd.DataFrame:
    """Load one MeteoSwiss file and transform it into the target schema."""
    file_path = WEATHER_FOLDER / file_name

    df = pd.read_csv(
        file_path, sep=";", encoding="utf-8",
        na_values=["", " ", "NA", "nan"],
    )
    df["reference_timestamp"] = pd.to_datetime(
        df["reference_timestamp"], format="%d.%m.%Y %H:%M", errors="coerce"
    )

    # Keep only the columns we need.
    df = df[["station_abbr", "reference_timestamp"] + SELECTED_WEATHER_FEATURES].copy()

    # Restrict to the analysis window.
    df = df[
        (df["reference_timestamp"] >= ANALYSIS_START) &
        (df["reference_timestamp"] <= ANALYSIS_END)
    ].copy()

    # Attach canton group; drop any rows from stations we don't use.
    df["canton"] = df["station_abbr"].map(STATION_TO_CANTON)
    df = df[df["canton"].notna()].copy()

    df = df.rename(columns=RENAME_MAP)[ORDERED_COLUMNS].copy()
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].astype(float)

    return df


def get_weather_files() -> list[str]:
    """Return all weather timeseries files (metadata files are excluded)."""
    return [
        f.name for f in sorted(WEATHER_FOLDER.glob("*.csv"))
        if not f.name.startswith("ogd-smn_meta_")
    ]


def run_quality_checks(weather_df: pd.DataFrame) -> None:
    """
    Validate the combined dataset before export. Raises on any problem so a
    broken build never silently overwrites the CSV.

    Checks:
      - no duplicate (timestamp, canton) keys
      - every expected canton is present
      - every canton has exactly the expected number of hourly rows
      - no missing hour x canton combinations in the analysis window
    """
    duplicates = weather_df.duplicated(subset=["timestamp", "canton"]).sum()
    if duplicates > 0:
        raise ValueError(f"{duplicates} duplicate rows on (timestamp, canton).")

    rows_per_canton = weather_df["canton"].value_counts().sort_index()
    expected_rows   = len(pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="h"))

    missing_cantons = sorted(set(EXPECTED_CANTONS) - set(rows_per_canton.index))
    if missing_cantons:
        raise ValueError(f"Missing cantons in dataset: {missing_cantons}")

    wrong_counts = rows_per_canton[rows_per_canton != expected_rows]
    if not wrong_counts.empty:
        raise ValueError(
            f"Cantons with wrong row count (expected {expected_rows}):\n{wrong_counts}"
        )

    # Check for missing hour x canton combinations by comparing against a full grid.
    expected_grid = (
        pd.MultiIndex
          .from_product(
              [EXPECTED_CANTONS, pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="h")],
              names=["canton", "timestamp"],
          )
          .to_frame(index=False)[["timestamp", "canton"]]
    )
    actual_grid = weather_df[["timestamp", "canton"]].drop_duplicates()

    missing_hours = expected_grid.merge(
        actual_grid, on=["timestamp", "canton"], how="left", indicator=True
    )
    missing_hours = missing_hours[missing_hours["_merge"] == "left_only"]
    if not missing_hours.empty:
        raise ValueError(
            "Missing hours by canton:\n"
            f"{missing_hours['canton'].value_counts().sort_index()}"
        )


# --- Build -------------------------------------------------------------------

weather_files = get_weather_files()
print(f"Processing {len(weather_files)} weather files ...")

all_dfs = [transform_weather_file(f) for f in weather_files]
weather_df = (
    pd.concat(all_dfs, ignore_index=True)
      .sort_values(by=["canton", "timestamp"])
      .reset_index(drop=True)
)

run_quality_checks(weather_df)

# --- Summary & export --------------------------------------------------------

print(f"Rows: {len(weather_df):,}")
print(f"Time range: {weather_df['timestamp'].min()} -> {weather_df['timestamp'].max()}")
print("Rows per canton:")
print(weather_df["canton"].value_counts().sort_index().to_string())

# Store timestamps as a readable string to keep the CSV stable across locales.
weather_df["timestamp"] = weather_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
weather_df.to_csv(OUTPUT_FILE, index=False, sep=";", decimal=",")

print(f"Saved: {OUTPUT_FILE}")
