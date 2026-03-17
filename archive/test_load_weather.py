import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1) Grundeinstellungen
# ------------------------------------------------------------

weather_folder = Path("../data_raw/weather")
output_folder = Path("../data_processed")
output_folder.mkdir(parents=True, exist_ok=True)

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

expected_cantons = [
    "BE_JU",
    "GE_VD",
    "GR",
    "SG",
    "TI",
    "VS",
    "ZH_SH",
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


# ------------------------------------------------------------
# 3) Relevante Wetterfiles finden
# ------------------------------------------------------------

weather_files = get_weather_files()

print("\n" + "=" * 80)
print("GEFUNDENE WETTERFILES")
print("=" * 80)
for file_name in weather_files:
    print(file_name)

# ------------------------------------------------------------
# 4) Alle Wetterfiles verarbeiten
# ------------------------------------------------------------

all_dfs = []

for file_name in weather_files:
    print("\n" + "-" * 80)
    print(f"VERARBEITE: {file_name}")
    print("-" * 80)

    df_raw = load_weather_file(file_name)
    station_values = df_raw["station_abbr"].dropna().unique()

    print("Station_abbr:", station_values)

    df_part = transform_weather_file(file_name)

    print("Shape:", df_part.shape)

    if not df_part.empty:
        print("Min timestamp:", df_part["timestamp"].min())
        print("Max timestamp:", df_part["timestamp"].max())
        print("Kantone:", df_part["canton"].unique())
    else:
        print("Datei liefert im Analysezeitraum keine gemappten Daten.")

    all_dfs.append(df_part)

# ------------------------------------------------------------
# 5) Zusammenführen
# ------------------------------------------------------------

weather_df = pd.concat(all_dfs, ignore_index=True)

weather_df = weather_df.sort_values(
    by=["canton", "timestamp"]
).reset_index(drop=True)

print("\n" + "=" * 80)
print("KOMBINIERTER WEATHER DATAFRAME")
print("=" * 80)
print("Shape:", weather_df.shape)

print("\nErste 10 Zeilen:")
print(weather_df.head(10))

# ------------------------------------------------------------
# 6) Qualitätschecks
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("QUALITÄTSCHECKS")
print("=" * 80)

print("\nZeitbereich gesamt:")
print("Min timestamp:", weather_df["timestamp"].min())
print("Max timestamp:", weather_df["timestamp"].max())

print("\nZeilen pro Kanton:")
rows_per_canton = weather_df["canton"].value_counts().sort_index()
print(rows_per_canton)

duplicates = weather_df.duplicated(subset=["timestamp", "canton"]).sum()
print("\nDuplikate auf timestamp + canton:", duplicates)

print("\nMissing Values gesamt:")
print(weather_df.isna().sum().sort_values(ascending=False))

print("\nMissing Values pro Kanton:")
missing_by_canton = weather_df.groupby("canton")[numeric_columns].apply(lambda x: x.isna().sum())
print(missing_by_canton)

# ------------------------------------------------------------
# 7) Vollständigkeitscheck auf Stundenebene
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("VOLLSTÄNDIGKEITSCHECK")
print("=" * 80)

expected_grid = build_expected_hour_grid(expected_cantons)
actual_grid = weather_df[["timestamp", "canton"]].drop_duplicates()

missing_hours = expected_grid.merge(
    actual_grid,
    on=["timestamp", "canton"],
    how="left",
    indicator=True
)

missing_hours = missing_hours[missing_hours["_merge"] == "left_only"].drop(columns="_merge")

expected_rows_per_canton = len(pd.date_range(start=analysis_start, end=analysis_end, freq="h"))
print("\nErwartete Zeilen pro Kanton:", expected_rows_per_canton)

print("\nFehlende Stunden pro Kanton:")
if missing_hours.empty:
    print("Keine fehlenden Stunden.")
else:
    print(missing_hours["canton"].value_counts().sort_index())

# ------------------------------------------------------------
# 8) Finaler Test-Output
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SORTIERTER FINALER TEST-DATAFRAME")
print("=" * 80)
print(weather_df.shape)
print(weather_df.head(20))

# Optionaler Test-Export
test_output_path = output_folder / "weather_dataset_test.csv"
weather_df.to_csv(
    test_output_path,
    index=False,
    sep=";",
    decimal=","
)
print(f"\nTest-Export gespeichert unter: {test_output_path}")