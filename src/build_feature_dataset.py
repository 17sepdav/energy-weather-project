import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1) Konfiguration
# ------------------------------------------------------------

INPUT_FILE = Path("../data_processed/analytical_base.csv")
OUTPUT_FILE = Path("../data_processed/feature_dataset.csv")

CSV_SEPARATOR = ";"
CSV_DECIMAL = ","

CONFIG = {
    "base_temperature_hdd": 18.0,
    "base_temperature_cdd": 18.0,
    "extreme_cold_threshold": 0.0,
    "extreme_heat_threshold": 25.0,
    "heavy_precipitation_threshold": 5.0
}

# ------------------------------------------------------------
# 2) Daten laden
# ------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL,
        parse_dates=["timestamp"]
    )

# ------------------------------------------------------------
# 3) Feature Engineering
# ------------------------------------------------------------

def add_temperature_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
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
    heavy_thr = config["heavy_precipitation_threshold"]

    df["precipitation_flag"] = (df["precipitation"] > 0).astype(float)
    df["heavy_precipitation_flag"] = (df["precipitation"] >= heavy_thr).astype(float)

    df.loc[df["precipitation"].isna(), ["precipitation_flag", "heavy_precipitation_flag"]] = np.nan

    return df


def add_consumption_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag-Features pro Kanton"""

    df = df.sort_values(["canton", "timestamp"])

    df["consumption_lag_24h"] = df.groupby("canton")["consumption_mwh"].shift(24)
    df["consumption_lag_168h"] = df.groupby("canton")["consumption_mwh"].shift(168)

    return df

# ------------------------------------------------------------
# 4) Qualitätscheck
# ------------------------------------------------------------

def run_quality_check(df: pd.DataFrame) -> None:
    print("\n--- Qualitätscheck ---")
    print("Zeilen:", len(df))
    print("Spalten:", len(df.columns))

    print("\nNullwerte (Auszug):")
    print(df[[
        "temperature_2m", "hdd", "cdd",
        "precipitation", "precipitation_flag",
        "consumption_lag_24h", "consumption_lag_168h"
    ]].isna().sum())

    print("\nSample (Wetter):")
    print(df[[
        "timestamp", "canton",
        "temperature_2m", "hdd", "cdd",
        "precipitation", "precipitation_flag"
    ]].head(5))

    print("\nSample (Lags):")
    print(df[[
        "timestamp", "canton",
        "consumption_mwh",
        "consumption_lag_24h",
        "consumption_lag_168h"
    ]].head(10))

# ------------------------------------------------------------
# 5) Main
# ------------------------------------------------------------

def main():
    print("Lade Daten ...")
    df = load_data(INPUT_FILE)

    print("Erzeuge Temperatur-Features ...")
    df = add_temperature_features(df, CONFIG)

    print("Erzeuge Niederschlags-Features ...")
    df = add_precipitation_features(df, CONFIG)

    print("Erzeuge Lag-Features ...")
    df = add_consumption_lag_features(df)

    run_quality_check(df)

    print("\nSpeichere Feature Dataset ...")
    df.to_csv(
        OUTPUT_FILE,
        sep=CSV_SEPARATOR,
        decimal=CSV_DECIMAL,
        index=False
    )

    print("Fertig.")


if __name__ == "__main__":
    main()