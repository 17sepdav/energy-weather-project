import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1) Pfade und Grundeinstellungen
# ------------------------------------------------------------

input_file  = Path("../data_processed/feature_dataset.csv")
output_file = Path("../data_processed/scenario_predictions_rf.csv")

target_candidates = [
    "consumption_mwh",
    "electricity_consumption_mwh",
    "stromverbrauch_mwh",
    "consumption"
]

numeric_columns = [
    "consumption_mwh",
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
    "hdd",
    "cdd",
    "is_extreme_cold",
    "is_extreme_heat",
    "precipitation_flag",
    "heavy_precipitation_flag",
    "consumption_lag_24h",
    "consumption_lag_168h"
]

# Scenario features: no lag variables (not interactively controllable in Power BI)
# This mirrors the logic of Model D (RF + Canton) from analyse_regression_extended.py
scenario_features = [
    "canton",
    "season",
    "day_type",
    "hour",
    "temperature_bucket"
]


# ------------------------------------------------------------
# 2) Hilfsfunktionen
# ------------------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def find_target_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Keine Zielvariable gefunden.\n"
        f"Vorhandene Spalten: {list(df.columns)}"
    )


def read_csv_robust(file_path: Path) -> pd.DataFrame:
    last_error = None

    for sep in [";", ","]:
        try:
            df = pd.read_csv(
                file_path,
                sep=sep,
                encoding="utf-8-sig",
                engine="python"
            )
            print(f"CSV erfolgreich gelesen mit Separator: '{sep}'")
            return df
        except Exception as e:
            last_error = e

    raise ValueError(f"CSV konnte nicht eingelesen werden:\n{last_error}")


def convert_decimal_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"]      = df["timestamp"].dt.hour.astype("Int64")
    df["month"]     = df["timestamp"].dt.month

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_type"] = df["day_of_week"].map({
        0: "Weekday",
        1: "Weekday",
        2: "Weekday",
        3: "Weekday",
        4: "Weekday",
        5: "Saturday",
        6: "Sunday"
    })

    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter",  2: "Winter",
         3: "Spring", 4: "Spring",  5: "Spring",
         6: "Summer", 7: "Summer",  8: "Summer",
         9: "Autumn", 10: "Autumn", 11: "Autumn"
    })

    return df


def add_temperature_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "temperature_2m" not in df.columns:
        raise ValueError("Spalte 'temperature_2m' fehlt.")

    df["temperature_bucket"] = pd.cut(
        df["temperature_2m"],
        bins=[-999, 0, 10, 20, 30, 999],
        labels=["very_cold", "cold", "mild", "warm", "hot"]
    )

    return df


# ------------------------------------------------------------
# 3) Daten laden und vorbereiten
# ------------------------------------------------------------

print_section("DATEI LADEN")

if not input_file.exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {input_file}")

df = read_csv_robust(input_file)
print(f"Datei geladen: {input_file}")
print(f"Shape: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

print_section("DATEN AUFBEREITEN")

df = convert_decimal_columns(df, numeric_columns)
df = add_time_features(df)
df = add_temperature_bucket(df)

target_col = find_target_column(df, target_candidates)

print("Datenaufbereitung abgeschlossen.")


# ------------------------------------------------------------
# 4) Modell-Datensatz vorbereiten
# ------------------------------------------------------------

print_section("MODELL-DATENSATZ")

required_cols = scenario_features + [target_col]
missing_required = [col for col in required_cols if col not in df.columns]

if missing_required:
    raise ValueError(f"Fehlende Spalten für Modellierung: {missing_required}")

model_df = df[required_cols].copy()
model_df = model_df.dropna(subset=[target_col])

# hour as string so the RF treats it as a categorical feature (realistic day profiles)
model_df["hour"] = model_df["hour"].astype("string")

print(f"Shape Modell-Datensatz: {model_df.shape[0]} Zeilen, {model_df.shape[1]} Spalten")


# ------------------------------------------------------------
# 5) Train/Test-Split
# ------------------------------------------------------------

print_section("TRAIN / TEST SPLIT")

X = model_df[scenario_features]
y = model_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Train Shape: {X_train.shape}")
print(f"Test Shape:  {X_test.shape}")


# ------------------------------------------------------------
# 6) Modell trainieren – Random Forest (analog zu Modell D)
#
#    Replaces the previous LinearRegression model.
#    Hyperparameters are aligned with analyse_regression_extended.py
#    (Model D: RF + Canton, no lag features) for consistency.
#    n_estimators=200 / max_depth=12 / min_samples_leaf=5 balance
#    accuracy and training time on ~670k hourly rows.
# ------------------------------------------------------------

print_section("MODELL TRAINIEREN (Random Forest)")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, scenario_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1          # use all available CPU cores
    ))
])

model.fit(X_train, y_train)

print("Modelltraining abgeschlossen.")


# ------------------------------------------------------------
# 7) Modell evaluieren
# ------------------------------------------------------------

print_section("MODELL EVALUATION")

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"MAE :  {mae:.3f}")
print(f"RMSE:  {rmse:.3f}")
print(f"R²  :  {r2:.4f}")


# ------------------------------------------------------------
# 8) Beispiel-Predictions
# ------------------------------------------------------------

print_section("BEISPIEL-PREDICTIONS")

example_scenarios = pd.DataFrame([
    {
        "canton":             "ZH_SH",
        "season":             "Winter",
        "day_type":           "Weekday",
        "hour":               "8",
        "temperature_bucket": "very_cold"
    },
    {
        "canton":             "ZH_SH",
        "season":             "Summer",
        "day_type":           "Sunday",
        "hour":               "15",
        "temperature_bucket": "warm"
    },
    {
        "canton":             "TI",
        "season":             "Winter",
        "day_type":           "Weekday",
        "hour":               "18",
        "temperature_bucket": "cold"
    }
])

example_scenarios["predicted_consumption_mwh"] = model.predict(example_scenarios)
print(example_scenarios.to_string(index=False))


# ------------------------------------------------------------
# 9) Szenario-Kombinationen erzeugen
# ------------------------------------------------------------

print_section("SZENARIO-KOMBINATIONEN ERZEUGEN")

canton_values      = sorted(df["canton"].dropna().unique().tolist())
season_values      = ["Winter", "Spring", "Summer", "Autumn"]
day_type_values    = ["Weekday", "Saturday", "Sunday"]
hour_values        = [str(h) for h in range(24)]
temperature_values = ["very_cold", "cold", "mild", "warm", "hot"]

scenario_predictions = pd.MultiIndex.from_product(
    [
        canton_values,
        season_values,
        day_type_values,
        hour_values,
        temperature_values
    ],
    names=[
        "canton",
        "season",
        "day_type",
        "hour",
        "temperature_bucket"
    ]
).to_frame(index=False)

print(f"Anzahl Szenarien: {len(scenario_predictions)}")


# ------------------------------------------------------------
# 10) Predictions für Szenario-Tabelle berechnen
# ------------------------------------------------------------

print_section("SZENARIO-PREDICTIONS BERECHNEN")

scenario_predictions["predicted_consumption_mwh"] = model.predict(scenario_predictions)

print("Beispiel der Szenario-Tabelle:")
print(scenario_predictions.head(10).to_string(index=False))


# ------------------------------------------------------------
# 11) Szenario-Tabelle speichern
# ------------------------------------------------------------

print_section("OUTPUT SPEICHERN")

scenario_predictions.to_csv(
    output_file,
    sep=";",
    index=False,
    decimal=","
)
print(f"Szenario-Datei gespeichert unter: {output_file}")
