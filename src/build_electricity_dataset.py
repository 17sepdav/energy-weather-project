import re
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1) Grundeinstellungen
# ------------------------------------------------------------

# Ordner mit den Rohdateien der Stromdaten
electricity_folder = Path("../data_raw/electricity")

# Zielordner für die aufbereiteten Daten
output_folder = Path("../data_processed")
output_folder.mkdir(parents=True, exist_ok=True)

# Relevante Spalten aus den Swissgrid-Dateien:
# Links steht jeweils der Originalname in der Excel-Datei,
# rechts der kompakte Zielname für die weitere Verarbeitung.
selected_columns = {
    "Unnamed: 0": "timestamp",
    "Verbrauch Kanton GR\nConsumption Canton GR": "GR",
    "Verbrauch Kanton SG\nConsumption Canton SG": "SG",
    "Verbrauch Kanton TI\nConsumption Canton TI": "TI",
    "Verbrauch Kanton VS\nConsumption Canton VS": "VS",
    "Verbrauch Kantone GE, VD\nConsumption Cantons GE, VD": "GE_VD",
    "Verbrauch Kantone SH, ZH\nConsumption Cantons SH, ZH": "ZH_SH",
    "Verbrauch Kantone BE, JU\nConsumption Cantons BE, JU": "BE_JU",
}

# ------------------------------------------------------------
# 2) Alle relevanten Excel-Dateien finden
# ------------------------------------------------------------

all_files = sorted(list(electricity_folder.glob("*.xls")) + list(electricity_folder.glob("*.xlsx")))

print("Gefundene Stromdateien:")
for file in all_files:
    print(f" - {file.name}")

if not all_files:
    raise FileNotFoundError("Keine Stromdateien im Ordner '../data_raw/electricity' gefunden.")


# ------------------------------------------------------------
# 3) Funktion zur Verarbeitung einer einzelnen Datei
# ------------------------------------------------------------

def process_electricity_file(file_path: Path) -> pd.DataFrame:
    """
    Liest eine einzelne Swissgrid-Datei ein und transformiert sie in
    ein stündliches, langes Format.

    Ergebnisstruktur:
    timestamp | canton_code | consumption_mwh
    """

    # Jahr aus dem Dateinamen extrahieren, z. B. 2024 aus:
    # 'EnergieUebersichtCH-2024.xlsx'
    match = re.search(r"(\d{4})", file_path.stem)
    if not match:
        raise ValueError(f"Kein Jahr im Dateinamen gefunden: {file_path.name}")

    file_year = int(match.group(1))

    # Excel-Datei einlesen:
    # - Sheet 'Zeitreihen0h15' enthält die Viertelstunden-Zeitreihe
    # - header=0 verwendet die erste Zeile als Spaltennamen
    # - skiprows=[1] überspringt die Zeile mit den Einheiten (kWh)
    df = pd.read_excel(
        file_path,
        sheet_name="Zeitreihen0h15",
        header=0,
        skiprows=[1]
    )

    # Nur die für das Projekt relevanten Spalten behalten
    df = df[list(selected_columns.keys())].rename(columns=selected_columns)

    # Zeitstempel in echtes Datumsformat umwandeln
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y %H:%M")

    # Nur Daten behalten, die wirklich zum Jahr der Datei gehören.
    # Damit werden Überlappungen zwischen Jahresdateien vermieden.
    df = df[df["timestamp"].dt.year == file_year].copy()

    # Von breitem Format in langes Format umwandeln:
    # vorher: eine Spalte pro Region
    # nachher: eine Zeile pro Zeitstempel und Region
    df_long = df.melt(
        id_vars="timestamp",
        var_name="canton_code",
        value_name="consumption_kwh"
    )

    # Viertelstundenwerte auf volle Stunden runden
    # Beispiel: 00:15, 00:30, 00:45 -> 00:00
    df_long["timestamp"] = df_long["timestamp"].dt.floor("h")

    # Viertelstundenwerte je Stunde und Region aufsummieren
    # Summe ist korrekt, da es sich um Energiemengen handelt
    df_hourly = (
        df_long
        .groupby(["timestamp", "canton_code"], as_index=False)
        .agg(consumption_kwh=("consumption_kwh", "sum"))
    )

    # kWh in MWh umrechnen, damit die Werte später besser lesbar sind
    df_hourly["consumption_mwh"] = df_hourly["consumption_kwh"] / 1000

    # kWh-Spalte wird danach nicht mehr benötigt
    df_hourly = df_hourly.drop(columns=["consumption_kwh"])

    return df_hourly


# ------------------------------------------------------------
# 4) Alle Dateien verarbeiten und zusammenführen
# ------------------------------------------------------------

all_data = []

for file_path in all_files:
    print(f"\nVerarbeite Datei: {file_path.name}")
    df_processed = process_electricity_file(file_path)
    all_data.append(df_processed)

electricity_final = pd.concat(all_data, ignore_index=True)


# ------------------------------------------------------------
# 5) Finale Aufbereitung
# ------------------------------------------------------------

# Zur Sicherheit chronologisch sortieren
electricity_final = electricity_final.sort_values(
    by=["timestamp", "canton_code"]
).reset_index(drop=True)

# Finale Spaltenreihenfolge
electricity_final = electricity_final[["timestamp", "canton_code", "consumption_mwh"]]


# ------------------------------------------------------------
# 6) Qualitätschecks / Übersicht
# ------------------------------------------------------------

# Prüfen, ob nach dem Jahr-Filter noch Duplikate vorhanden sind.
# Die Kombination aus timestamp + canton_code sollte eindeutig sein.
duplicate_count = electricity_final.duplicated(
    subset=["timestamp", "canton_code"]
).sum()

print("\n--- Finale Übersicht Stromdaten ---")
print(f"Anzahl Zeilen: {len(electricity_final)}")
print(f"Anzahl Spalten: {electricity_final.shape[1]}")

print("\nZeitraum:")
print(f"Von: {electricity_final['timestamp'].min()}")
print(f"Bis: {electricity_final['timestamp'].max()}")

print("\nAnzahl Datensätze pro Region:")
print(electricity_final["canton_code"].value_counts().sort_index())

print(f"\nVerbleibende Duplikate (timestamp + canton_code): {duplicate_count}")

print("\nErste 10 Zeilen:")
print(electricity_final.head(10))


# ------------------------------------------------------------
# 7) Export
# ------------------------------------------------------------

output_path = output_folder / "electricity_hourly_2015_2026.csv"
electricity_final["consumption_mwh"] = electricity_final["consumption_mwh"].round(3)

output_path = output_folder / "electricity_hourly_2015_2026.csv"
electricity_final.to_csv(
    output_path,
    index=False,
    sep=";",
    decimal=","
)

print(f"\nDatei erfolgreich gespeichert: {output_path}")