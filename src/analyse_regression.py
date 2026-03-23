import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------
# 1) Daten laden
# ------------------------------------------------------------
input_file = Path("../data_processed/feature_dataset.csv")

df = pd.read_csv(input_file, sep=";", parse_dates=["timestamp"])

# ------------------------------------------------------------
# 2) Numerische Spalten konvertieren
# ------------------------------------------------------------
numeric_cols = [
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

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 3) Zeitfeatures
# ------------------------------------------------------------
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

# ------------------------------------------------------------
# 4) Regressionsfunktion
# ------------------------------------------------------------
def run_regression(df, features, target="consumption_mwh", use_canton=False):

    cols = features + [target]

    if use_canton:
        cols.append("canton")

    data = df[cols].dropna()

    # One-Hot-Encoding für canton
    if use_canton:
        data = pd.get_dummies(data, columns=["canton"], drop_first=True)

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_
    }).sort_values(by="coefficient", key=abs, ascending=False)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "coefficients": coef_df,
        "intercept": model.intercept_
    }

# ------------------------------------------------------------
# 5) MODELL A: MIT LAG-FEATURES
# ------------------------------------------------------------
features_with_lag = [
    "temperature_2m",
    "hdd",
    "cdd",
    "hour",
    "day_of_week",
    "month",
    "consumption_lag_24h",
    "consumption_lag_168h"
]

result_a = run_regression(df, features_with_lag)

print("\n" + "=" * 70)
print("MODELL A: MIT LAG-FEATURES")
print("=" * 70)

print(f"MAE:  {result_a['mae']:.2f}")
print(f"RMSE: {result_a['rmse']:.2f}")
print(f"R²:   {result_a['r2']:.4f}")

print("\nTop Features:")
print(result_a["coefficients"].head(10).to_string(index=False))

print(f"\nIntercept: {result_a['intercept']:.2f}")

# ------------------------------------------------------------
# 6) MODELL B: OHNE LAG (WETTER + ZEIT)
# ------------------------------------------------------------
features_without_lag = [
    "temperature_2m",
    "hdd",
    "cdd",
    "hour",
    "day_of_week",
    "month"
]

result_b = run_regression(df, features_without_lag)

print("\n" + "=" * 70)
print("MODELL B: OHNE LAG-FEATURES (WETTER + ZEIT)")
print("=" * 70)

print(f"MAE:  {result_b['mae']:.2f}")
print(f"RMSE: {result_b['rmse']:.2f}")
print(f"R²:   {result_b['r2']:.4f}")

print("\nTop Features:")
print(result_b["coefficients"].head(10).to_string(index=False))

print(f"\nIntercept: {result_b['intercept']:.2f}")

# ------------------------------------------------------------
# 7) MODELL C: OHNE LAG + CANTON
# ------------------------------------------------------------
result_c = run_regression(
    df,
    features_without_lag,
    use_canton=True
)

print("\n" + "=" * 70)
print("MODELL C: OHNE LAG + CANTON (REGION)")
print("=" * 70)

print(f"MAE:  {result_c['mae']:.2f}")
print(f"RMSE: {result_c['rmse']:.2f}")
print(f"R²:   {result_c['r2']:.4f}")

print("\nTop Features:")
print(result_c["coefficients"].head(10).to_string(index=False))

print(f"\nIntercept: {result_c['intercept']:.2f}")

# ------------------------------------------------------------
# 8) EXPORT FÜR POWER BI
# ------------------------------------------------------------
output_folder = Path("../data_processed")
output_folder.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 8.1 Modell-Metriken
# -----------------------------
metrics_df = pd.DataFrame([
    {
        "model": "A_with_lag",
        "mae": result_a["mae"],
        "rmse": result_a["rmse"],
        "r2": result_a["r2"]
    },
    {
        "model": "B_weather_time",
        "mae": result_b["mae"],
        "rmse": result_b["rmse"],
        "r2": result_b["r2"]
    },
    {
        "model": "C_weather_time_canton",
        "mae": result_c["mae"],
        "rmse": result_c["rmse"],
        "r2": result_c["r2"]
    }
])

metrics_df.to_csv(output_folder / "regression_model_metrics.csv", sep=";", decimal=",",index=False)

# -----------------------------
# 8.2 Koeffizienten (Feature Importance)
# -----------------------------
def prepare_coef_df(result, model_name):
    df_coef = result["coefficients"].copy()
    df_coef["model"] = model_name

    # Feature-Typ klassifizieren (für bessere Visualisierung)
    def classify_feature(f):
        if "canton_" in f:
            return "region"
        elif f in ["hdd", "cdd", "temperature_2m"]:
            return "weather"
        elif f in ["hour", "day_of_week", "month"]:
            return "time"
        elif "lag" in f:
            return "lag"
        else:
            return "other"

    df_coef["feature_type"] = df_coef["feature"].apply(classify_feature)

    return df_coef

coef_df = pd.concat([
    prepare_coef_df(result_a, "A_with_lag"),
    prepare_coef_df(result_b, "B_weather_time"),
    prepare_coef_df(result_c, "C_weather_time_canton")
])

coef_df.to_csv(output_folder / "regression_coefficients.csv", sep=";", decimal=",",index=False)

# -----------------------------
# 8.3 Predictions Sample (für Visuals)
# -----------------------------
# nur Modell C sinnvoll für Darstellung

features_c = [
    "temperature_2m",
    "hdd",
    "cdd",
    "hour",
    "day_of_week",
    "month"
]

sample_df = df[features_c + ["consumption_mwh", "canton", "timestamp"]].dropna().sample(5000, random_state=42)

sample_encoded = pd.get_dummies(sample_df, columns=["canton"], drop_first=True)

X_sample = sample_encoded.drop(columns=["consumption_mwh", "timestamp"])
y_sample = sample_encoded["consumption_mwh"]

# Modell neu fitten auf vollständigen Daten (für stabile Predictions)
full_data = df[features_c + ["consumption_mwh", "canton"]].dropna()
full_data = pd.get_dummies(full_data, columns=["canton"], drop_first=True)

X_full = full_data.drop(columns=["consumption_mwh"])
y_full = full_data["consumption_mwh"]

model_c = LinearRegression()
model_c.fit(X_full, y_full)

y_pred_sample = model_c.predict(X_sample)

sample_df["predicted_consumption"] = y_pred_sample

sample_df.to_csv(output_folder / "regression_predictions_sample.csv", sep=";", decimal=",", index=False)

# ------------------------------------------------------------
# 9) INFO
# ------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPORT ABGESCHLOSSEN")
print("=" * 70)

print("Erstellte Dateien:")
print("- regression_model_metrics.csv")
print("- regression_coefficients.csv")
print("- regression_predictions_sample.csv")