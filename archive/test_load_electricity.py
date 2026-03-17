import pandas as pd
from pathlib import Path

file_path = Path("../data_raw/electricity/EnergieUebersichtCH-2025.xlsx")

df = pd.read_excel(
    file_path,
    sheet_name="Zeitreihen0h15",
    header=0,
    skiprows=[1]
)

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

df = df[list(selected_columns.keys())].rename(columns=selected_columns)

df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y %H:%M")

# Wide -> Long
df_long = df.melt(
    id_vars="timestamp",
    var_name="canton_code",
    value_name="consumption_kwh"
)

print("Shape breit:")
print(df.shape)

print("\nShape lang:")
print(df_long.shape)

print("\nErste 10 Zeilen lang:")
print(df_long.head(10))

print("\nDatentypen lang:")
print(df_long.dtypes)

print("\nAnzahl Werte pro Kanton:")
print(df_long["canton_code"].value_counts())


# --------------------------------
# 15min -> Stundenaggregation
# --------------------------------

df_long["timestamp_hour"] = df_long["timestamp"].dt.floor("h")

df_hourly = (
    df_long
    .groupby(["timestamp_hour", "canton_code"], as_index=False)
    .agg(consumption_kwh=("consumption_kwh", "sum"))
)

print("\nShape stündlich:")
print(df_hourly.shape)

print("\nErste 10 Zeilen stündlich:")
print(df_hourly.head(10))

print("\nWerte pro Kanton:")
print(df_hourly["canton_code"].value_counts())

print("\nMin / Max Zeitstempel 15min:")
print(df_long["timestamp"].min())
print(df_long["timestamp"].max())

print("\nMin / Max Zeitstempel stündlich:")
print(df_hourly["timestamp_hour"].min())
print(df_hourly["timestamp_hour"].max())

print("\nAnzahl eindeutige Stunden:")
print(df_hourly["timestamp_hour"].nunique())

print("\nErste 5 Stunden:")
print(df_hourly["timestamp_hour"].drop_duplicates().sort_values().head())

print("\nLetzte 5 Stunden:")
print(df_hourly["timestamp_hour"].drop_duplicates().sort_values().tail())