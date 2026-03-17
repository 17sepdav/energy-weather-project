import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1) Grundeinstellungen
# ------------------------------------------------------------

weather_folder = Path("../data_raw/weather")
output_folder = Path("../data_processed")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "weather_dataset.csv"

selected_weather_features = [
    "tre200h0",
    "rre150h0",
    "ure200h0",
    "gre000h0",
    "sre000h0",
    "fkl010h0",
    "dkl010h0",
    "prestah0",
    "pp0qnhh0",
    "pva200h0",
    "tde200h0",
]

station_to_canton = {
    "BER": "BE_JU",
    "GVE": "GE_VD",
    "DAV": "GR",
    "STG": "SG",
    "LUG": "TI",
    "EVO": "VS",
    "SMA": "ZH_SH",
}

expected_cantons = [
    "BE_JU",
    "GE_VD",
    "GR",
    "SG",
    "TI",
    "VS",
    "ZH_SH",
]

analysis_start = pd.Timestamp("2015-01-01 00:00:00")
analysis_end = pd.Timestamp("2025-12-31 23:00:00")

rename_map = {
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

ordered_columns = [
    "timestamp",
    "canton",
    "station_abbr",
    "temperature_2m",
    "precipitation",
    "humidity_rel",
    "global_radiation",
    "sunshine_duration",
    "wind_speed",
    "wind_direction",
    "pressure_station",
    "pressure_qnh",
    "vapour_pressure",
    "dew_point",
]

numeric_columns = [
    col for col in ordered_columns
    if col not in {"timestamp", "canton", "station_abbr"}
]

# ------------------------------------------------------------
# 2) Hilfsfunktionen
# ------------------------------------------------------------

def load_weather_file(file_name: str) -> pd.DataFrame:
    """Lädt eine Wetterdatei und parsed den Zeitstempel."""
    file_path = weather_folder / file_name

    df = pd.read_csv(
        file_path,
        sep=";",
        encoding="utf-8",
        na_values=["", " ", "NA", "nan"],
    )

    df["reference_timestamp"] = pd.to_datetime(
        df["reference_timestamp"],
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )

    return df


def transform_weather_file(file_name: str) -> pd.DataFrame:
    """Transformiert eine einzelne Wetterdatei in das gewünschte Zielschema."""
    df = load_weather_file(file_name)

    required_columns = ["station_abbr", "reference_timestamp"] + selected_weather_features
    df = df[required_columns].copy()

    df = df[
        (df["reference_timestamp"] >= analysis_start) &
        (df["reference_timestamp"] <= analysis_end)
    ].copy()

    df["canton"] = df["station_abbr"].map(station_to_canton)
    df = df[df["canton"].notna()].copy()

    df = df.rename(columns=rename_map)
    df = df[ordered_columns].copy()

    df[numeric_columns] = df[numeric_columns].astype(float)

    return df


def get_weather_files() -> list[str]:
    """Liefert alle echten Wetter-Zeitreihenfiles ohne Metadateien."""
    all_csv_files = sorted(weather_folder.glob("*.csv"))
    return [
        file.name
        for file in all_csv_files
        if not file.name.startswith("ogd-smn_meta_")
    ]


def build_expected_hour_grid(cantons: list[str]) -> pd.DataFrame:
    """Erzeugt die erwartete Stunden-Kanton-Kombination für den Analysezeitraum."""
    timestamps = pd.date_range(
        start=analysis_start,
        end=analysis_end,
        freq="h",
    )

    grid = pd.MultiIndex.from_product(
        [cantons, timestamps],
        names=["canton", "timestamp"]
    ).to_frame(index=False)

    return grid[["timestamp", "canton"]]


def run_quality_checks(weather_df: pd.DataFrame) -> None:
    """Führt zentrale Qualitätschecks durch und bricht bei kritischen Problemen ab."""
    duplicates = weather_df.duplicated(subset=["timestamp", "canton"]).sum()
    if duplicates > 0:
        raise ValueError(f"Es wurden {duplicates} Duplikate auf timestamp + canton gefunden.")

    rows_per_canton = weather_df["canton"].value_counts().sort_index()

    expected_rows_per_canton = len(
        pd.date_range(start=analysis_start, end=analysis_end, freq="h")
    )

    missing_cantons = sorted(set(expected_cantons) - set(rows_per_canton.index))
    if missing_cantons:
        raise ValueError(f"Folgende Kantone fehlen vollständig im Datensatz: {missing_cantons}")

    wrong_row_counts = rows_per_canton[rows_per_canton != expected_rows_per_canton]
    if not wrong_row_counts.empty:
        raise ValueError(
            "Nicht alle Kantone haben die erwartete Anzahl Stunden:\n"
            f"{wrong_row_counts.to_string()}"
        )

    expected_grid = build_expected_hour_grid(expected_cantons)
    actual_grid = weather_df[["timestamp", "canton"]].drop_duplicates()

    missing_hours = expected_grid.merge(
        actual_grid,
        on=["timestamp", "canton"],
        how="left",
        indicator=True
    )

    missing_hours = missing_hours[missing_hours["_merge"] == "left_only"].drop(columns="_merge")

    if not missing_hours.empty:
        missing_summary = missing_hours["canton"].value_counts().sort_index()
        raise ValueError(
            "Es fehlen Stunden im finalen Datensatz:\n"
            f"{missing_summary.to_string()}"
        )


# ------------------------------------------------------------
# 3) Build starten
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("BUILD WEATHER DATASET")
print("=" * 80)

weather_files = get_weather_files()
print(f"\nGefundene Wetterfiles: {len(weather_files)}")

all_dfs = []

for file_name in weather_files:
    print(f"Verarbeite: {file_name}")
    df_part = transform_weather_file(file_name)
    all_dfs.append(df_part)

weather_df = pd.concat(all_dfs, ignore_index=True)

weather_df = weather_df.sort_values(
    by=["canton", "timestamp"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 4) Qualitätschecks
# ------------------------------------------------------------

run_quality_checks(weather_df)

# ------------------------------------------------------------
# 5) Build-Zusammenfassung
# ------------------------------------------------------------

print("\nBuild-Zusammenfassung:")
print(f"- Zeilen gesamt: {len(weather_df):,}".replace(",", "'"))
print(f"- Spalten gesamt: {weather_df.shape[1]}")
print(f"- Zeitraum: {weather_df['timestamp'].min()} bis {weather_df['timestamp'].max()}")

print("\nZeilen pro Kanton:")
print(weather_df["canton"].value_counts().sort_index())

print("\nMissing Values gesamt:")
print(weather_df.isna().sum().sort_values(ascending=False))

# ------------------------------------------------------------
# 6) Export
# ------------------------------------------------------------

weather_df["timestamp"] = weather_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

weather_df.to_csv(
    output_file,
    index=False,
    sep=";",
    decimal=","
)

print(f"\nWeather-Dataset erfolgreich gespeichert unter: {output_file}")