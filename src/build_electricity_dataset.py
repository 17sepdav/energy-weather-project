"""
build_electricity_dataset.py
============================
Builds the hourly electricity consumption dataset from the raw Swissgrid
Excel files ("EnergieUebersichtCH-YYYY.xls[x]").

Pipeline
--------
1. Load each yearly Swissgrid file from ../data_raw/electricity/
2. Keep only the regional consumption columns we need for the project
3. Parse timestamps and restrict each file to rows belonging to its own year
   (Swissgrid files overlap at year boundaries, so we trim to avoid duplicates)
4. Reshape wide -> long (one row per timestamp x canton)
5. Aggregate 15-minute values to hourly sums (values are energy, so sum is correct)
6. Convert kWh -> MWh for better readability downstream
7. Concatenate all years and export as a single CSV

Output: ../data_processed/electricity_hourly_2015_2026.csv
"""

import re
import pandas as pd
from pathlib import Path

# --- Configuration -----------------------------------------------------------

ELECTRICITY_FOLDER = Path("../data_raw/electricity")
OUTPUT_FOLDER      = Path("../data_processed")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "electricity_hourly_2015_2026.csv"

# Mapping of original Swissgrid column names -> compact canton codes used downstream.
# Only these columns are kept from each Excel file; everything else is dropped.
SELECTED_COLUMNS = {
    "Unnamed: 0": "timestamp",
    "Verbrauch Kanton GR\nConsumption Canton GR":           "GR",
    "Verbrauch Kanton SG\nConsumption Canton SG":           "SG",
    "Verbrauch Kanton TI\nConsumption Canton TI":           "TI",
    "Verbrauch Kanton VS\nConsumption Canton VS":           "VS",
    "Verbrauch Kantone GE, VD\nConsumption Cantons GE, VD": "GE_VD",
    "Verbrauch Kantone SH, ZH\nConsumption Cantons SH, ZH": "ZH_SH",
    "Verbrauch Kantone BE, JU\nConsumption Cantons BE, JU": "BE_JU",
}

# --- Processing --------------------------------------------------------------

def process_electricity_file(file_path: Path) -> pd.DataFrame:
    """
    Read a single Swissgrid yearly file and return hourly consumption in long
    format with columns: timestamp, canton_code, consumption_mwh.
    """
    # Extract the year from the filename (e.g. "EnergieUebersichtCH-2024.xlsx" -> 2024).
    # Used below to drop rows that spill into neighbouring years.
    match = re.search(r"(\d{4})", file_path.stem)
    if not match:
        raise ValueError(f"No year found in filename: {file_path.name}")
    file_year = int(match.group(1))

    # Sheet 'Zeitreihen0h15' holds the 15-minute timeseries.
    # skiprows=[1] skips the units row (kWh) that sits between header and data.
    df = pd.read_excel(
        file_path,
        sheet_name="Zeitreihen0h15",
        header=0,
        skiprows=[1],
    )

    # Keep and rename only the regional columns of interest.
    df = df[list(SELECTED_COLUMNS.keys())].rename(columns=SELECTED_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y %H:%M")

    # Drop rows that belong to a different year than the file (avoids overlap duplicates).
    df = df[df["timestamp"].dt.year == file_year].copy()

    # Wide -> long: one row per (timestamp, canton_code).
    df_long = df.melt(
        id_vars="timestamp",
        var_name="canton_code",
        value_name="consumption_kwh",
    )

    # Floor timestamps to full hours so 00:15 / 00:30 / 00:45 all map to the 00:00 bucket.
    df_long["timestamp"] = df_long["timestamp"].dt.floor("h")

    # Aggregate the four quarter-hour values per hour. Sum is correct for energy quantities.
    df_hourly = (
        df_long
        .groupby(["timestamp", "canton_code"], as_index=False)
        .agg(consumption_kwh=("consumption_kwh", "sum"))
    )

    # Convert to MWh for downstream readability.
    df_hourly["consumption_mwh"] = df_hourly["consumption_kwh"] / 1000
    df_hourly = df_hourly.drop(columns=["consumption_kwh"])

    return df_hourly


# --- Build -------------------------------------------------------------------

all_files = sorted(
    list(ELECTRICITY_FOLDER.glob("*.xls")) + list(ELECTRICITY_FOLDER.glob("*.xlsx"))
)
if not all_files:
    raise FileNotFoundError(f"No Swissgrid files found in {ELECTRICITY_FOLDER}")

print(f"Processing {len(all_files)} Swissgrid files ...")

all_data = [process_electricity_file(f) for f in all_files]
electricity_final = pd.concat(all_data, ignore_index=True)

# Final ordering and column layout.
electricity_final = (
    electricity_final
    .sort_values(by=["timestamp", "canton_code"])
    .reset_index(drop=True)
)
electricity_final = electricity_final[["timestamp", "canton_code", "consumption_mwh"]]
electricity_final["consumption_mwh"] = electricity_final["consumption_mwh"].round(3)

# --- Quality check -----------------------------------------------------------
# (timestamp, canton_code) must be unique after the year-filter logic above.
duplicate_count = electricity_final.duplicated(subset=["timestamp", "canton_code"]).sum()

# --- Summary & export --------------------------------------------------------

print(f"Rows: {len(electricity_final):,}")
print(f"Time range: {electricity_final['timestamp'].min()} -> {electricity_final['timestamp'].max()}")
print(f"Duplicates on (timestamp, canton_code): {duplicate_count}")

electricity_final.to_csv(OUTPUT_FILE, index=False, sep=";", decimal=",")
print(f"Saved: {OUTPUT_FILE}")
