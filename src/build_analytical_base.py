"""
build_analytical_base.py
========================
Combines the processed electricity and weather datasets into a single
analytical base table at the granularity of (timestamp x canton).

Pipeline
--------
1. Load both processed datasets
2. Standardise the join keys (uppercase, trimmed canton; rename canton_code -> canton)
3. Restrict both sides to their common time window so the join cannot leak
   partial data from a period where only one source exists
4. Validate overlap on (timestamp, canton) and report what would be dropped
5. Inner join on (timestamp, canton) — keep only fully observed rows
6. Sort, run null summary, export

Output: ../data_processed/analytical_base.csv
"""

import pandas as pd
from pathlib import Path

# --- Configuration -----------------------------------------------------------

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
ELECTRICITY_FILE = PROJECT_ROOT / "data_processed" / "electricity_hourly_2015_2026.csv"
WEATHER_FILE     = PROJECT_ROOT / "data_processed" / "weather_dataset.csv"
OUTPUT_FILE      = PROJECT_ROOT / "data_processed" / "analytical_base.csv"


# --- Helpers -----------------------------------------------------------------

def load_data(file_path: Path) -> pd.DataFrame:
    """Load a semicolon-separated CSV that uses comma as decimal separator."""
    return pd.read_csv(file_path, sep=";", decimal=",", parse_dates=["timestamp"])


def standardize_keys(
    df_electricity: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align join keys between both datasets:
      - strip/upper-case canton on both sides
      - rename electricity's `canton_code` to `canton` so the merge key name matches
    """
    df_electricity = df_electricity.copy()
    df_weather     = df_weather.copy()

    df_electricity["canton_code"] = df_electricity["canton_code"].astype(str).str.strip().str.upper()
    df_weather["canton"]          = df_weather["canton"].astype(str).str.strip().str.upper()

    df_electricity = df_electricity.rename(columns={"canton_code": "canton"})
    return df_electricity, df_weather


def get_common_time_window(
    df_electricity: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the [start, end] range where both datasets have data."""
    start = max(df_electricity["timestamp"].min(), df_weather["timestamp"].min())
    end   = min(df_electricity["timestamp"].max(), df_weather["timestamp"].max())
    return start, end


def filter_to_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Keep only rows whose timestamp lies within the common window."""
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()


def report_join_coverage(df_e: pd.DataFrame, df_w: pd.DataFrame) -> int:
    """
    Print how many (timestamp, canton) keys match between the two sides and
    return the count of matching keys for post-merge validation.
    """
    keys_e = set(zip(df_e["timestamp"], df_e["canton"]))
    keys_w = set(zip(df_w["timestamp"], df_w["canton"]))
    matching = keys_e & keys_w

    print(f"Electricity keys: {len(keys_e):,}")
    print(f"Weather keys:     {len(keys_w):,}")
    print(f"Matching keys:    {len(matching):,}")
    print(f"Only electricity: {len(keys_e - keys_w):,}")
    print(f"Only weather:     {len(keys_w - keys_e):,}")
    return len(matching)


# --- Main --------------------------------------------------------------------

def main() -> None:
    print("Loading datasets ...")
    df_electricity = load_data(ELECTRICITY_FILE)
    df_weather     = load_data(WEATHER_FILE)
    print(f"Electricity: {df_electricity.shape} | Weather: {df_weather.shape}")

    # Align join keys before any further operation.
    df_electricity, df_weather = standardize_keys(df_electricity, df_weather)

    # Restrict both sides to the overlapping time window.
    start, end = get_common_time_window(df_electricity, df_weather)
    print(f"Common window: {start} -> {end}")

    df_electricity = filter_to_window(df_electricity, start, end)
    df_weather     = filter_to_window(df_weather,     start, end)

    # Diagnostic: how many keys will actually join?
    expected_rows = report_join_coverage(df_electricity, df_weather)

    # Inner join with validate="one_to_one" enforces that the join key is unique
    # on both sides — catches duplicate timestamps early instead of silently
    # multiplying rows.
    df_ab = pd.merge(
        df_electricity, df_weather,
        on=["timestamp", "canton"],
        how="inner",
        validate="one_to_one",
    )

    # Sanity check: inner-join row count must equal the number of matching keys.
    if len(df_ab) != expected_rows:
        raise ValueError(
            f"Row count mismatch: expected {expected_rows:,}, got {len(df_ab):,}"
        )

    # Final ordering for deterministic output.
    df_ab = df_ab.sort_values(by=["canton", "timestamp"]).reset_index(drop=True)

    print(f"Analytical base: {df_ab.shape}")
    print("Nulls per column:")
    print(df_ab.isna().sum().to_string())

    df_ab.to_csv(OUTPUT_FILE, sep=";", decimal=",", index=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
