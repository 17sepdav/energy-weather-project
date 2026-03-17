import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1) Grundeinstellungen
# ------------------------------------------------------------

weather_folder = Path("../data_raw/weather")
output_folder = Path("../data_processed")
output_folder.mkdir(parents=True, exist_ok=True)

stations_meta_file = weather_folder / "ogd-smn_meta_stations.csv"
output_file = output_folder / "dim_location.csv"

station_to_canton = {
    "BER": "BE_JU",
    "GVE": "GE_VD",
    "DAV": "GR",
    "STG": "SG",
    "LUG": "TI",
    "EVO": "VS",
    "SMA": "ZH_SH",
}

canton_name_map = {
    "BE_JU": "Bern/Jura",
    "GE_VD": "Genf/Waadt",
    "GR": "Graubünden",
    "SG": "St. Gallen",
    "TI": "Tessin",
    "VS": "Wallis",
    "ZH_SH": "Zürich/Schaffhausen",
}

ordered_columns = [
    "canton",
    "canton_name",
    "station_abbr",
    "station_name",
    "station_canton",
    "latitude",
    "longitude",
    "height_masl",
    "height_barometer_masl",
    "lv95_east",
    "lv95_north",
    "station_type",
    "station_dataowner",
]

# ------------------------------------------------------------
# 2) Hilfsfunktionen
# ------------------------------------------------------------

def load_station_metadata(file_path: Path) -> pd.DataFrame:
    """Lädt die Stations-Metadaten."""
    try:
        df = pd.read_csv(
            file_path,
            sep=";",
            encoding="utf-8",
            na_values=["", " ", "NA", "nan"],
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            file_path,
            sep=";",
            encoding="latin1",
            na_values=["", " ", "NA", "nan"],
        )

    return df


def run_quality_checks(df: pd.DataFrame) -> None:
    """Führt einfache Qualitätschecks für die Location-Dimension aus."""
    if len(df) != 7:
        raise ValueError(f"Dim_Location sollte genau 7 Zeilen haben, aktuell: {len(df)}")

    if df["canton"].duplicated().any():
        duplicates = df.loc[df["canton"].duplicated(), "canton"].tolist()
        raise ValueError(f"Doppelte canton-Werte gefunden: {duplicates}")

    if df["station_abbr"].duplicated().any():
        duplicates = df.loc[df["station_abbr"].duplicated(), "station_abbr"].tolist()
        raise ValueError(f"Doppelte station_abbr-Werte gefunden: {duplicates}")

    required_not_null = ["canton", "canton_name", "station_abbr", "station_name", "latitude", "longitude"]
    missing_required = df[required_not_null].isna().sum()

    if missing_required.any():
        raise ValueError(
            "Pflichtfelder enthalten Missing Values:\n"
            f"{missing_required[missing_required > 0].to_string()}"
        )


# ------------------------------------------------------------
# 3) Metadaten laden
# ------------------------------------------------------------

stations_df = load_station_metadata(stations_meta_file)

# Falls nötig: einmal zur Kontrolle aktivieren
# print(stations_df.columns.tolist())
# print(stations_df.head())

# ------------------------------------------------------------
# 4) Auf relevante Stationen filtern
# ------------------------------------------------------------

stations_df = stations_df[stations_df["station_abbr"].isin(station_to_canton.keys())].copy()

# ------------------------------------------------------------
# 5) Kanton-Key und sprechende Namen ergänzen
# ------------------------------------------------------------

stations_df["canton"] = stations_df["station_abbr"].map(station_to_canton)
stations_df["canton_name"] = stations_df["canton"].map(canton_name_map)

# ------------------------------------------------------------
# 6) Spalten umbenennen
# ------------------------------------------------------------

rename_map = {
    "station_name": "station_name",
    "station_canton": "station_canton",
    "station_height_masl": "height_masl",
    "station_height_barometer_masl": "height_barometer_masl",
    "station_coordinates_wgs84_lat": "latitude",
    "station_coordinates_wgs84_lon": "longitude",
    "station_coordinates_lv95_east": "lv95_east",
    "station_coordinates_lv95_north": "lv95_north",
    "station_type_en": "station_type",
    "station_dataowner": "station_dataowner",
}

stations_df = stations_df.rename(columns=rename_map)

# ------------------------------------------------------------
# 7) Relevante Spalten auswählen
# ------------------------------------------------------------

stations_df = stations_df[ordered_columns].copy()

# Numerische Felder sauber setzen
numeric_columns = [
    "latitude",
    "longitude",
    "height_masl",
    "height_barometer_masl",
    "lv95_east",
    "lv95_north",
]

for col in numeric_columns:
    stations_df[col] = pd.to_numeric(stations_df[col], errors="coerce")

# Sortierung für sauberen Export
stations_df = stations_df.sort_values(by="canton").reset_index(drop=True)

# ------------------------------------------------------------
# 8) Qualitätschecks
# ------------------------------------------------------------

run_quality_checks(stations_df)

# ------------------------------------------------------------
# 9) Export
# ------------------------------------------------------------

stations_df.to_csv(
    output_file,
    index=False,
    sep=";",
    decimal=",",
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("BUILD DIM_LOCATION")
print("=" * 80)
print(f"Zeilen: {len(stations_df)}")
print(f"Spalten: {stations_df.shape[1]}")
print("\nInhalt:")
print(stations_df)
print(f"\nDim_Location erfolgreich gespeichert unter: {output_file}")