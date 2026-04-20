"""
analyse_regression.py
=====================
Baseline regression analysis of the hourly electricity consumption.

Three linear regression models are trained and compared to isolate the
contribution of different feature groups:

    Model A — weather + time + lag
              Best predictive accuracy (lag features capture autocorrelation).

    Model B — weather + time only
              Measures how much signal weather + calendar carry on their own.

    Model C — weather + time + canton (one-hot encoded)
              Adds regional structure; shows how much a region's baseline
              level explains without any weather/time interactions.

All three use the same train/test split (80/20, random_state=42) so their
metrics are directly comparable.

Outputs
-------
- regression_model_metrics.csv      MAE / RMSE / R² per model
- regression_coefficients.csv       Coefficients + feature type classification
- regression_predictions_sample.csv 5000-row sample of actual vs. predicted
                                    (based on Model C, for visual checks)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# --- Configuration -----------------------------------------------------------

INPUT_FILE    = Path("../data_processed/feature_dataset.csv")
OUTPUT_FOLDER = Path("../data_processed")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

TARGET = "consumption_mwh"

# All numeric columns that may contain comma decimals (legacy CSV format)
# and must be coerced to float before modelling.
NUMERIC_COLS = [
    "consumption_mwh",
    "temperature_2m", "precipitation", "humidity_rel",
    "global_radiation", "sunshine_duration",
    "wind_speed", "wind_direction",
    "pressure_station", "pressure_qnh",
    "vapour_pressure", "dew_point",
    "hdd", "cdd",
    "is_extreme_cold", "is_extreme_heat",
    "precipitation_flag", "heavy_precipitation_flag",
    "consumption_lag_24h", "consumption_lag_168h",
]

# Feature sets for the three models.
FEATURES_WITH_LAG = [
    "temperature_2m", "hdd", "cdd",
    "hour", "day_of_week", "month",
    "consumption_lag_24h", "consumption_lag_168h",
]
FEATURES_WITHOUT_LAG = [
    "temperature_2m", "hdd", "cdd",
    "hour", "day_of_week", "month",
]


# --- Helpers -----------------------------------------------------------------

def classify_feature(name: str) -> str:
    """Group features into thematic types for visualisation in Power BI."""
    if "canton_" in name:                                return "region"
    if name in ("hdd", "cdd", "temperature_2m"):         return "weather"
    if name in ("hour", "day_of_week", "month"):         return "time"
    if "lag" in name:                                    return "lag"
    return "other"


def run_regression(df: pd.DataFrame, features: list, use_canton: bool = False) -> dict:
    """
    Fit a linear regression on `features` (plus one-hot-encoded canton when
    requested) and return metrics plus the sorted coefficient table.

    Rows with NaN in any selected column are dropped — this keeps the three
    models on comparable "clean" data without having to impute missing values.
    """
    cols = features + [TARGET] + (["canton"] if use_canton else [])
    data = df[cols].dropna()

    if use_canton:
        # drop_first=True avoids the dummy-variable trap: one canton becomes
        # the implicit baseline, and the remaining coefficients show the
        # relative offset of each other canton.
        data = pd.get_dummies(data, columns=["canton"], drop_first=True)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    coef_df = (
        pd.DataFrame({"feature": X.columns, "coefficient": model.coef_})
          .sort_values(by="coefficient", key=abs, ascending=False)
    )

    return {
        "mae":          mean_absolute_error(y_test, y_pred),
        "rmse":         np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2":           r2_score(y_test, y_pred),
        "coefficients": coef_df,
        "intercept":    model.intercept_,
    }


def prepare_coef_df(result: dict, model_name: str) -> pd.DataFrame:
    """Attach model name and feature-type classification to a coefficient table."""
    out = result["coefficients"].copy()
    out["model"]        = model_name
    out["feature_type"] = out["feature"].apply(classify_feature)
    return out


def print_model_summary(name: str, result: dict) -> None:
    """Compact console summary for one model."""
    print(f"\n{name}")
    print(f"  MAE:  {result['mae']:.2f}")
    print(f"  RMSE: {result['rmse']:.2f}")
    print(f"  R²:   {result['r2']:.4f}")
    print("  Top coefficients:")
    print(result["coefficients"].head(5).to_string(index=False))


# --- Load & prepare data -----------------------------------------------------

print("Loading feature dataset ...")
df = pd.read_csv(INPUT_FILE, sep=";", parse_dates=["timestamp"])

# Coerce numeric columns (some come in with commas as decimal separator).
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

# Derive time features from the timestamp (same logic as the dim_time table
# but computed inline so this script can also be run standalone).
df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"]       = df["timestamp"].dt.month


# --- Train models ------------------------------------------------------------

result_a = run_regression(df, FEATURES_WITH_LAG,    use_canton=False)
print_model_summary("Model A — with lag features", result_a)

result_b = run_regression(df, FEATURES_WITHOUT_LAG, use_canton=False)
print_model_summary("Model B — weather + time only", result_b)

result_c = run_regression(df, FEATURES_WITHOUT_LAG, use_canton=True)
print_model_summary("Model C — weather + time + canton", result_c)


# --- Export model metrics ----------------------------------------------------

metrics_df = pd.DataFrame([
    {"model": "A_with_lag",            "mae": result_a["mae"], "rmse": result_a["rmse"], "r2": result_a["r2"]},
    {"model": "B_weather_time",        "mae": result_b["mae"], "rmse": result_b["rmse"], "r2": result_b["r2"]},
    {"model": "C_weather_time_canton", "mae": result_c["mae"], "rmse": result_c["rmse"], "r2": result_c["r2"]},
])
metrics_df.to_csv(OUTPUT_FOLDER / "regression_model_metrics.csv",
                  sep=";", decimal=",", index=False)


# --- Export coefficients (stacked across models) -----------------------------

coef_df = pd.concat([
    prepare_coef_df(result_a, "A_with_lag"),
    prepare_coef_df(result_b, "B_weather_time"),
    prepare_coef_df(result_c, "C_weather_time_canton"),
], ignore_index=True)

coef_df.to_csv(OUTPUT_FOLDER / "regression_coefficients.csv",
               sep=";", decimal=",", index=False)


# --- Export predictions sample (Model C) -------------------------------------
# Used by Power BI for scatter plots of actual vs. predicted consumption and
# to check residual patterns by canton / hour. Fitted on the full clean dataset
# (not just the train split) so predictions on the sample are stable.

features_c = FEATURES_WITHOUT_LAG

# Fit on all clean rows to produce stable predictions for the sample.
full_data = df[features_c + [TARGET, "canton"]].dropna()
full_data = pd.get_dummies(full_data, columns=["canton"], drop_first=True)

X_full = full_data.drop(columns=[TARGET])
y_full = full_data[TARGET]
model_c_full = LinearRegression().fit(X_full, y_full)

# Draw a reproducible 5000-row sample with full context columns (timestamp, canton).
sample_df = df[features_c + [TARGET, "canton", "timestamp"]].dropna().sample(
    5000, random_state=42
)
sample_encoded = pd.get_dummies(sample_df, columns=["canton"], drop_first=True)
X_sample = sample_encoded.drop(columns=[TARGET, "timestamp"])

sample_df["predicted_consumption"] = model_c_full.predict(X_sample)

sample_df.to_csv(OUTPUT_FOLDER / "regression_predictions_sample.csv",
                 sep=";", decimal=",", index=False)

print("\nExports written:")
print("  regression_model_metrics.csv")
print("  regression_coefficients.csv")
print("  regression_predictions_sample.csv")
