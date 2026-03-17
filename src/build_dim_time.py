import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1) Pfade
# ------------------------------------------------------------
input_file = Path("../data_processed/analytical_base.csv")
output_file = Path("../data_processed/dim_time.csv")

# ------------------------------------------------------------
# 2) CSV einlesen
# ------------------------------------------------------------
df = pd.read_csv(input_file, sep=";", encoding="utf-8-sig")

# Timestamp in echtes Datumsformat umwandeln
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ------------------------------------------------------------
# 3) Eindeutige Zeitwerte extrahieren
# ------------------------------------------------------------
dim_time = (
    df[["timestamp"]]
    .drop_duplicates()
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# 4) Hilfsfunktionen
# ------------------------------------------------------------
def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Frühling"
    elif month in [6, 7, 8]:
        return "Sommer"
    else:
        return "Herbst"


def get_day_name(day_of_week: int) -> str:
    day_names = {
        0: "Montag",
        1: "Dienstag",
        2: "Mittwoch",
        3: "Donnerstag",
        4: "Freitag",
        5: "Samstag",
        6: "Sonntag"
    }
    return day_names[day_of_week]


def get_hour_bucket(hour: int) -> str:
    if 0 <= hour <= 5:
        return "Nacht"
    elif 6 <= hour <= 11:
        return "Morgen"
    elif 12 <= hour <= 17:
        return "Nachmittag"
    else:
        return "Abend"

# ------------------------------------------------------------
# 5) Zeitfeatures erzeugen
# ------------------------------------------------------------
dim_time["time_key"] = dim_time["timestamp"].dt.strftime("%Y%m%d%H").astype(int)

dim_time["year"] = dim_time["timestamp"].dt.year
dim_time["quarter"] = dim_time["timestamp"].dt.quarter
dim_time["month"] = dim_time["timestamp"].dt.month
dim_time["week_of_year"] = dim_time["timestamp"].dt.isocalendar().week.astype(int)
dim_time["day_of_week"] = dim_time["timestamp"].dt.dayofweek   # Montag=0, Sonntag=6
dim_time["day_name"] = dim_time["day_of_week"].apply(get_day_name)
dim_time["hour"] = dim_time["timestamp"].dt.hour

dim_time["is_weekend"] = dim_time["day_of_week"].isin([5, 6]).astype(int)
dim_time["season"] = dim_time["month"].apply(get_season)
dim_time["year_month"] = dim_time["timestamp"].dt.strftime("%Y-%m")
dim_time["is_business_hour"] = dim_time["hour"].between(8, 17).astype(int)
dim_time["hour_bucket"] = dim_time["hour"].apply(get_hour_bucket)

# ------------------------------------------------------------
# 6) Export
# ------------------------------------------------------------
dim_time.to_csv(output_file, sep=";", index=False, encoding="utf-8-sig")

print(f"Time-Dimension erstellt: {output_file}")
print(f"Anzahl Zeilen: {len(dim_time)}")
print(dim_time.head())
print(dim_time.dtypes)