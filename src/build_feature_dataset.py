"""
build_feature_dataset.py
========================
Extends the analytical base with additional explanatory features used in
correlation analysis, regression models and dashboards.

Feature groups
--------------
- Temperature-based
    * hdd  — Heating Degree Days (base 18°C): heating demand indicator
    * cdd  — Cooling Degree Days (base 18°C): cooling demand indicator
    * is_extreme_cold / is_extreme_heat — binary flags for extreme conditions

- Precipitation
    * precipitation_flag        — any measurable precipitation (>0 mm)
    * heavy_precipitation_flag  — heavy precipitation (>=5 mm/h)

- Temporal lag features (per canton, same hour)
    * consumption_lag_24h   — consumption 24 hours ago  (daily pattern)
    * consumption_lag_168h  — consumption 7 days ago    (weekly pattern)

Output: ../data_processed/feature_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration -----------------------------------------------------------

INPUT_FILE  = Path("../data_processed/analytical_base.csv")
OUTPUT_FILE = Path("../data_processed/feature_dataset.csv")

CSV_SEPARATOR = ";"
CSV_DECIMAL   = ","

# Thresholds for feature engineering. Kept in one place so they are easy to tune.
CONFIG = {
    "base_temperature_hdd":          18.0,  # reference temperature for heating demand
    "base_temperature_cdd":          18.0,  # reference temperature for cooling demand
    "extreme_cold_threshold":         0.0,  # °C
    "extreme_heat_threshold":        25.0,  # °C
    "heavy_precipitation_threshold":  5.0,  # mm/h
}


# --- Feature engineering -----------------------------------------------------

def add_temperature_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute HDD/CDD and extreme-temperature flags from `temperature_2m`.

    HDD and CDD are clipped at 0 because negative demand makes no sense. Rows
    without a temperature reading keep NaN everywhere instead of becoming 0 —
    this avoids biasing the flags toward "not extreme" when we simply have no
    measurement.
    """
    base_hdd = config["base_temperature_hdd"]
    base_cdd = config["base_temperature_cdd"]

    df["hdd"] = np.maximum(base_hdd - df["temperature_2m"], 0)
    df["cdd"] = np.maximum(df["temperature_2m"] - base_cdd, 0)
    df.loc[df["temperature_2m"].isna(), ["hdd", "cdd"]] = np.nan

    df["is_extreme_cold"] = (df["temperature_2m"] <= config["extreme_cold_threshold"]).astype(float)
    df["is_extreme_heat"] = (df["temperature_2m"] >= config["extreme_heat_threshold"]).astype(float)
    df.loc[df["temperature_2m"].isna(), ["is_extreme_cold", "is_extreme_heat"]] = np.nan

    return df


def add_precipitation_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Binary precipitation flags, NaN-preserving (same rationale as above)."""
    heavy_threshold = config["heavy_precipitation_threshold"]

    df["precipitation_flag"]       = (df["precipitation"] > 0).astype(float)
    df["heavy_precipitation_flag"] = (df["precipitation"] >= heavy_threshold).astype(float)
    df.loc[df["precipitation"].isna(), ["precipitation_flag", "heavy_precipitation_flag"]] = np.nan

    return df


def add_consumption_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add consumption lags computed within each canton. Sorting by
    (canton, timestamp) beforehand makes `shift(n)` correspond to the value
    from exactly n hours earlier in the same region.
    """
    df = df.sort_values(["canton", "timestamp"])
    df["consumption_lag_24h"]  = df.groupby("canton")["consumption_mwh"].shift(24)
    df["consumption_lag_168h"] = df.groupby("canton")["consumption_mwh"].shift(168)
    return df


# --- Main --------------------------------------------------------------------

def main() -> None:
    print("Loading analytical base ...")
    df = pd.read_csv(
        INPUT_FILE,
        sep=CSV_SEPARATOR, decimal=CSV_DECIMAL,
        parse_dates=["timestamp"],
    )

    df = add_temperature_features(df, CONFIG)
    df = add_precipitation_features(df, CONFIG)
    df = add_consumption_lag_features(df)

    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("Nulls in key feature columns:")
    print(
        df[[
            "temperature_2m", "hdd", "cdd",
            "precipitation", "precipitation_flag",
            "consumption_lag_24h", "consumption_lag_168h",
        ]].isna().sum().to_string()
    )

    df.to_csv(OUTPUT_FILE, sep=CSV_SEPARATOR, decimal=CSV_DECIMAL, index=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
