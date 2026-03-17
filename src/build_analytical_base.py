import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ELECTRICITY_FILE = PROJECT_ROOT / "data_processed" / "electricity_hourly_2015_2026.csv"
WEATHER_FILE = PROJECT_ROOT / "data_processed" / "weather_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data_processed" / "analytical_base.csv"


# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def print_section(title: str) -> None:
    """Print a formatted section header to the console."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_data(file_path: Path) -> pd.DataFrame:
    """Load a semicolon-separated CSV file with comma decimals."""
    return pd.read_csv(
        file_path,
        sep=";",
        decimal=",",
        parse_dates=["timestamp"]
    )


def standardize_keys(
    df_electricity: pd.DataFrame,
    df_weather: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize join keys for both datasets."""
    df_electricity = df_electricity.copy()
    df_weather = df_weather.copy()

    df_electricity["canton_code"] = (
        df_electricity["canton_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df_weather["canton"] = (
        df_weather["canton"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df_electricity = df_electricity.rename(columns={"canton_code": "canton"})

    return df_electricity, df_weather


def get_common_time_window(
    df_electricity: pd.DataFrame,
    df_weather: pd.DataFrame
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Determine the common timestamp range across both datasets."""
    electricity_min = df_electricity["timestamp"].min()
    electricity_max = df_electricity["timestamp"].max()
    weather_min = df_weather["timestamp"].min()
    weather_max = df_weather["timestamp"].max()

    common_start = max(electricity_min, weather_min)
    common_end = min(electricity_max, weather_max)

    print(f"Electricity range: {electricity_min} -> {electricity_max}")
    print(f"Weather range:     {weather_min} -> {weather_max}")
    print(f"Common range:      {common_start} -> {common_end}")

    return common_start, common_end


def filter_to_common_window(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp
) -> pd.DataFrame:
    """Filter a dataframe to the common timestamp window."""
    return df[
        (df["timestamp"] >= start) &
        (df["timestamp"] <= end)
    ].copy()


def validate_join_coverage(
    df_electricity: pd.DataFrame,
    df_weather: pd.DataFrame
) -> set[tuple[pd.Timestamp, str]]:
    """Validate join-key coverage and return matching keys."""
    electricity_keys = set(zip(df_electricity["timestamp"], df_electricity["canton"]))
    weather_keys = set(zip(df_weather["timestamp"], df_weather["canton"]))

    matching_keys = electricity_keys & weather_keys
    only_electricity_keys = electricity_keys - weather_keys
    only_weather_keys = weather_keys - electricity_keys

    print(f"Electricity key count: {len(electricity_keys):,}")
    print(f"Weather key count:     {len(weather_keys):,}")
    print(f"Matching key count:    {len(matching_keys):,}")
    print(f"Only in electricity:   {len(only_electricity_keys):,}")
    print(f"Only in weather:       {len(only_weather_keys):,}")

    print("\nExample keys only in electricity:")
    for key in list(sorted(only_electricity_keys))[:5]:
        print(key)

    print("\nExample keys only in weather:")
    for key in list(sorted(only_weather_keys))[:5]:
        print(key)

    return matching_keys


def build_analytical_base(
    df_electricity: pd.DataFrame,
    df_weather: pd.DataFrame
) -> pd.DataFrame:
    """Create the analytical base by joining electricity and weather data."""
    df_merged = pd.merge(
        df_electricity,
        df_weather,
        on=["timestamp", "canton"],
        how="inner",
        validate="one_to_one"
    )

    return df_merged


def print_null_summary(df: pd.DataFrame) -> None:
    """Print null counts per column."""
    print(df.isna().sum())


def save_output(df: pd.DataFrame, output_file: Path) -> None:
    """Save analytical base to CSV."""
    df.to_csv(
        output_file,
        sep=";",
        decimal=",",
        index=False
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    print_section("LOAD DATA")
    df_electricity = load_data(ELECTRICITY_FILE)
    df_weather = load_data(WEATHER_FILE)

    print(f"Electricity shape: {df_electricity.shape}")
    print(f"Weather shape:     {df_weather.shape}")

    print_section("STANDARDIZE JOIN KEYS")
    df_electricity, df_weather = standardize_keys(df_electricity, df_weather)

    print("Electricity dtypes:")
    print(df_electricity[["timestamp", "canton"]].dtypes)

    print("\nWeather dtypes:")
    print(df_weather[["timestamp", "canton"]].dtypes)

    print_section("DEFINE COMMON TIME WINDOW")
    common_start, common_end = get_common_time_window(df_electricity, df_weather)

    print_section("FILTER TO COMMON TIME WINDOW")
    df_electricity_filtered = filter_to_common_window(df_electricity, common_start, common_end)
    df_weather_filtered = filter_to_common_window(df_weather, common_start, common_end)

    print(f"Filtered electricity shape: {df_electricity_filtered.shape}")
    print(f"Filtered weather shape:     {df_weather_filtered.shape}")

    print_section("JOIN KEY COVERAGE AFTER FILTERING")
    matching_keys = validate_join_coverage(df_electricity_filtered, df_weather_filtered)

    print_section("BUILD ANALYTICAL BASE")
    df_analytical_base = build_analytical_base(df_electricity_filtered, df_weather_filtered)
    print(f"Analytical base shape: {df_analytical_base.shape}")

    print_section("MERGE VALIDATION")
    expected_rows = len(matching_keys)
    actual_rows = len(df_analytical_base)

    print(f"Expected merged rows (matching keys): {expected_rows:,}")
    print(f"Actual merged rows:                   {actual_rows:,}")
    print(f"Row difference:                       {actual_rows - expected_rows:,}")

    print("\nColumns in analytical base:")
    print(df_analytical_base.columns.tolist())

    print_section("SAMPLE + NULL CHECK")
    print("Sample rows:")
    print(df_analytical_base.head(5).to_string(index=False))

    print("\nNull values per column:")
    print_null_summary(df_analytical_base)

    print_section("FINALIZE DATASET")
    df_analytical_base = df_analytical_base.sort_values(
        by=["canton", "timestamp"]
    ).reset_index(drop=True)
    print("Final dataset sorted.")

    print_section("SAVE ANALYTICAL BASE")
    save_output(df_analytical_base, OUTPUT_FILE)
    print(f"Analytical base saved to: {OUTPUT_FILE}")

    print_section("FINAL CHECK")
    print(f"Final shape: {df_analytical_base.shape}")

    print("\nTime range:")
    print(f"{df_analytical_base['timestamp'].min()} -> {df_analytical_base['timestamp'].max()}")

    print("\nNumber of cantons:")
    print(df_analytical_base['canton'].nunique())

    print("\nRows per canton:")
    print(df_analytical_base["canton"].value_counts().sort_index())

    print_section("DONE")
    print("Analytical Base successfully created.")


if __name__ == "__main__":
    main()