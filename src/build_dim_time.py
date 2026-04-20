"""
build_dim_time.py
=================
Builds the time dimension (Dim_Time) for the star schema.

One row per unique hour present in the analytical base. Each row carries
pre-computed time attributes (year, quarter, month, weekday, season, etc.)
so that Power BI can slice by these dimensions without re-deriving them.

Output: ../data_processed/dim_time.csv
"""

import pandas as pd
from pathlib import Path

# --- Configuration -----------------------------------------------------------

INPUT_FILE  = Path("../data_processed/analytical_base.csv")
OUTPUT_FILE = Path("../data_processed/dim_time.csv")


# --- Derived-attribute helpers -----------------------------------------------
# Kept as small functions for readability; all take a scalar and return a label.

def get_season(month: int) -> str:
    if month in (12, 1, 2):  return "Winter"
    if month in (3, 4, 5):   return "Frühling"
    if month in (6, 7, 8):   return "Sommer"
    return "Herbst"


def get_day_name(day_of_week: int) -> str:
    return {
        0: "Montag", 1: "Dienstag", 2: "Mittwoch", 3: "Donnerstag",
        4: "Freitag", 5: "Samstag", 6: "Sonntag",
    }[day_of_week]


def get_hour_bucket(hour: int) -> str:
    """Group 24 hours into four day-phases used in dashboards."""
    if 0  <= hour <= 5:  return "Nacht"
    if 6  <= hour <= 11: return "Morgen"
    if 12 <= hour <= 17: return "Nachmittag"
    return "Abend"


# --- Build -------------------------------------------------------------------

# Load analytical base and take the distinct set of hourly timestamps.
df = pd.read_csv(INPUT_FILE, sep=";", encoding="utf-8-sig")
df["timestamp"] = pd.to_datetime(df["timestamp"])

dim_time = (
    df[["timestamp"]]
      .drop_duplicates()
      .sort_values("timestamp")
      .reset_index(drop=True)
)

# Surrogate integer key YYYYMMDDHH — convenient for joins and sorting in the model.
dim_time["time_key"] = dim_time["timestamp"].dt.strftime("%Y%m%d%H").astype(int)

# Standard calendar attributes.
dim_time["year"]         = dim_time["timestamp"].dt.year
dim_time["quarter"]      = dim_time["timestamp"].dt.quarter
dim_time["month"]        = dim_time["timestamp"].dt.month
dim_time["week_of_year"] = dim_time["timestamp"].dt.isocalendar().week.astype(int)
dim_time["day_of_week"]  = dim_time["timestamp"].dt.dayofweek   # Monday = 0
dim_time["day_name"]     = dim_time["day_of_week"].apply(get_day_name)
dim_time["hour"]         = dim_time["timestamp"].dt.hour

# Analytical flags and groupings used across the report.
dim_time["is_weekend"]       = dim_time["day_of_week"].isin([5, 6]).astype(int)
dim_time["season"]           = dim_time["month"].apply(get_season)
dim_time["year_month"]       = dim_time["timestamp"].dt.strftime("%Y-%m")
dim_time["is_business_hour"] = dim_time["hour"].between(8, 17).astype(int)
dim_time["hour_bucket"]      = dim_time["hour"].apply(get_hour_bucket)

# --- Export ------------------------------------------------------------------

dim_time.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8-sig")

print(f"Dim_Time: {len(dim_time):,} rows")
print(f"Range: {dim_time['timestamp'].min()} -> {dim_time['timestamp'].max()}")
print(f"Saved: {OUTPUT_FILE}")
