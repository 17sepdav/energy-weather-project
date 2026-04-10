"""
analyse_regression_extended.py
================================
Extends analyse_regression.py with two Random Forest models and
prepares all results for Power BI visualisation.

Run order: execute AFTER analyse_regression.py, which produces the
baseline CSVs (regression_model_metrics.csv, regression_coefficients.csv,
regression_predictions_sample.csv) that this script reads and extends.

New / updated output files in data_processed/:
  - regression_model_metrics.csv      updated  (models D + E appended, algorithm column added)
  - regression_coefficients.csv       updated  (RF feature importances appended)
  - regression_residuals.csv          NEW      (residual analysis of best model for Power BI)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────

INPUT_FILE    = Path("../data_processed/feature_dataset.csv")
OUTPUT_FOLDER = Path("../data_processed")

TARGET = "consumption_mwh"

# ── Feature sets (identical to analyse_regression.py for comparability) ──────

# Model D: weather + time + canton  →  compared against Model C (LinReg)
FEATURES_WITHOUT_LAG = [
    "temperature_2m", "hdd", "cdd",
    "hour", "day_of_week", "month"
]

# Model E: weather + time + lag  →  compared against Model A (LinReg)
FEATURES_WITH_LAG = [
    "temperature_2m", "hdd", "cdd",
    "hour", "day_of_week", "month",
    "consumption_lag_24h", "consumption_lag_168h"
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load feature dataset, convert decimal format, derive time features."""
    df = pd.read_csv(path, sep=";", parse_dates=["timestamp"])

    numeric_cols = [
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )

    # Time features derived from timestamp (same logic as analyse_regression.py)
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month

    return df


def prepare_xy(df: pd.DataFrame, features: list, use_canton: bool = False):
    """
    Select relevant columns, drop rows with NaN, optionally one-hot-encode canton.
    Returns X, y and (if use_canton) the list of canton dummy column names.
    """
    cols = features + [TARGET] + (["canton"] if use_canton else [])
    data = df[cols].dropna()

    if use_canton:
        data = pd.get_dummies(data, columns=["canton"], drop_first=True)

    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    return X, y


def train_rf(X_train, y_train) -> RandomForestRegressor:
    """
    Train a Random Forest regressor with fixed hyperparameters.
    n_estimators=200 and max_depth=12 balance accuracy and training time
    on ~670k rows; min_samples_leaf=5 reduces overfitting on hourly data.
    """
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1          # use all available CPU cores
    )
    rf.fit(X_train, y_train)
    return rf


def compute_metrics(model, X_test, y_test) -> dict:
    """Return MAE, RMSE and R² for a fitted model on the test set."""
    y_pred = model.predict(X_test)
    return {
        "mae":    mean_absolute_error(y_test, y_pred),
        "rmse":   np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2":     r2_score(y_test, y_pred),
        "y_pred": y_pred,
        "y_test": y_test,
        "index":  X_test.index,
    }


def feature_importance_df(model, feature_names: list, model_name: str) -> pd.DataFrame:
    """
    Build a feature importance DataFrame in the same structure as the
    coefficient table from analyse_regression.py so both can be stacked
    in Power BI without transformation.
    """
    def classify(f: str) -> str:
        if "canton_" in f:   return "region"
        if f in ["hdd", "cdd", "temperature_2m"]: return "weather"
        if f in ["hour", "day_of_week", "month"]: return "time"
        if "lag" in f:       return "lag"
        return "other"

    df_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
        "coefficient": np.nan,         # not applicable for RF; kept for schema compatibility
    })
    df_imp["model"]        = model_name
    df_imp["algorithm"]    = "Random Forest"
    df_imp["feature_type"] = df_imp["feature"].apply(classify)
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp["rank"]         = df_imp.index + 1
    return df_imp

# ── 1. Load data ─────────────────────────────────────────────────────────────

print("Loading feature dataset ...")
df = load_and_prepare(INPUT_FILE)
print(f"  {df.shape[0]:,} rows | {df['canton'].nunique()} cantons "
      f"| {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

# ── 2. Model D – Random Forest: weather + time + canton ──────────────────────
#    Direct counterpart to Model C (Linear Regression, same feature set).
#    Canton is one-hot-encoded so the RF can learn region-specific baselines.

print("\nTraining Model D – RF (weather + time + canton) ...")

X_d, y_d = prepare_xy(df, FEATURES_WITHOUT_LAG, use_canton=True)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

rf_d      = train_rf(X_train_d, y_train_d)
metrics_d = compute_metrics(rf_d, X_test_d, y_test_d)

print(f"  MAE={metrics_d['mae']:.2f}  RMSE={metrics_d['rmse']:.2f}  R²={metrics_d['r2']:.4f}")

# ── 3. Model E – Random Forest: weather + time + lag ─────────────────────────
#    Direct counterpart to Model A (Linear Regression, same feature set).
#    Lag features encode the strong autocorrelation in hourly consumption.

print("\nTraining Model E – RF (weather + time + lag) ...")

X_e, y_e = prepare_xy(df, FEATURES_WITH_LAG, use_canton=False)
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_e, y_e, test_size=0.2, random_state=42
)

rf_e      = train_rf(X_train_e, y_train_e)
metrics_e = compute_metrics(rf_e, X_test_e, y_test_e)

print(f"  MAE={metrics_e['mae']:.2f}  RMSE={metrics_e['rmse']:.2f}  R²={metrics_e['r2']:.4f}")

# ── 4. Update regression_model_metrics.csv ───────────────────────────────────
#
#    Reads the existing file (models A/B/C from analyse_regression.py) and
#    appends models D and E. Also adds:
#      - algorithm:  "Linear Regression" or "Random Forest"
#      - model_label: human-readable name for Power BI axes/legends

print("\nUpdating regression_model_metrics.csv ...")

existing_metrics = pd.read_csv(
    OUTPUT_FOLDER / "regression_model_metrics.csv", sep=";", decimal=","
)

# Add algorithm label to existing linear regression models
algo_map = {
    "A_with_lag":            "Linear Regression",
    "B_weather_time":        "Linear Regression",
    "C_weather_time_canton": "Linear Regression",
}
existing_metrics["algorithm"] = existing_metrics["model"].map(algo_map)

# Append new RF models
new_models = pd.DataFrame([
    {
        "model":     "D_rf_weather_canton",
        "mae":       metrics_d["mae"],
        "rmse":      metrics_d["rmse"],
        "r2":        metrics_d["r2"],
        "algorithm": "Random Forest",
    },
    {
        "model":     "E_rf_with_lag",
        "mae":       metrics_e["mae"],
        "rmse":      metrics_e["rmse"],
        "r2":        metrics_e["r2"],
        "algorithm": "Random Forest",
    },
])

all_metrics = pd.concat([existing_metrics, new_models], ignore_index=True)

# Human-readable label used in Power BI charts
label_map = {
    "A_with_lag":            "A - LinReg + Lag",
    "B_weather_time":        "B - LinReg Weather",
    "C_weather_time_canton": "C - LinReg + Canton",
    "D_rf_weather_canton":   "D - RF + Canton",
    "E_rf_with_lag":         "E - RF + Lag",
}
all_metrics["model_label"] = all_metrics["model"].map(label_map)

# Sort order column so Power BI can display models A → E consistently
sort_map = {k: i for i, k in enumerate(label_map)}
all_metrics["model_sort"] = all_metrics["model"].map(sort_map)

all_metrics.to_csv(
    OUTPUT_FOLDER / "regression_model_metrics.csv", sep=";", decimal=",", index=False
)
print("  Done.")

# ── 5. Update regression_coefficients.csv ────────────────────────────────────
#
#    Reads the existing coefficient table (models A/B/C) and appends RF
#    feature importances for D and E. The shared schema allows Power BI to
#    filter by model or algorithm without separate data sources.

print("Updating regression_coefficients.csv ...")

existing_coef = pd.read_csv(
    OUTPUT_FOLDER / "regression_coefficients.csv", sep=";", decimal=","
)
existing_coef["algorithm"] = "Linear Regression"
existing_coef["importance"] = np.nan   # no RF importance for linear models
existing_coef["rank"]       = np.nan

imp_d = feature_importance_df(rf_d, X_d.columns.tolist(), "D_rf_weather_canton")
imp_e = feature_importance_df(rf_e, X_e.columns.tolist(), "E_rf_with_lag")

all_coef = pd.concat([existing_coef, imp_d, imp_e], ignore_index=True)

all_coef.to_csv(
    OUTPUT_FOLDER / "regression_coefficients.csv", sep=";", decimal=",", index=False
)
print("  Done.")

# ── 6. Create regression_residuals.csv (NEW) ──────────────────────────────────
#
#    Residual analysis of Model E (best model by R²).
#    Each row = one test-set observation with actual vs. predicted consumption,
#    the signed residual and contextual time / canton columns for Power BI slicing.
#
#    Key Power BI visuals this enables:
#      - Scatter: actual vs. predicted  →  how close to the diagonal?
#      - Bar: mean abs. residual by hour  →  which hours are hardest to predict?
#      - Bar: mean abs. residual by canton  →  which region has most uncertainty?
#      - Colour: error_direction  →  systematic over- vs. under-estimation?

print("Creating regression_residuals.csv ...")

residual_df = df.loc[metrics_e["index"], ["timestamp", "canton", "hour", "day_of_week", "month"]].copy()
residual_df["actual_consumption"]    = metrics_e["y_test"].values
residual_df["predicted_consumption"] = metrics_e["y_pred"]
residual_df["residual"]              = residual_df["actual_consumption"] - residual_df["predicted_consumption"]
residual_df["abs_residual"]          = residual_df["residual"].abs()
residual_df["residual_pct"]          = (
    residual_df["residual"] / residual_df["actual_consumption"].replace(0, np.nan) * 100
)

# Direction label: positive residual → model underestimated actual consumption
residual_df["error_direction"] = residual_df["residual"].apply(
    lambda x: "Underestimated" if x > 0 else "Overestimated"
)

day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
residual_df["day_name"] = residual_df["day_of_week"].map(day_map)
residual_df["model"]    = "E_rf_with_lag"

residual_df.to_csv(
    OUTPUT_FOLDER / "regression_residuals.csv", sep=";", decimal=",", index=False
)
print("  Done.")

# ── 7. Console summary ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MODEL COMPARISON (A → E)")
print("=" * 60)

summary = pd.DataFrame([
    {"Model": "A – LinReg + Lag",     "Algorithm": "Linear Regression", "MAE": 45.73,  "RMSE": 73.54,  "R²": 0.9495},
    {"Model": "B – LinReg Weather",   "Algorithm": "Linear Regression", "MAE": 292.37, "RMSE": 325.71, "R²": 0.0081},
    {"Model": "C – LinReg + Canton",  "Algorithm": "Linear Regression", "MAE": 90.25,  "RMSE": 124.71, "R²": 0.8546},
    {"Model": "D – RF + Canton",      "Algorithm": "Random Forest",     "MAE": round(metrics_d["mae"], 2),
                                                                          "RMSE": round(metrics_d["rmse"], 2),
                                                                          "R²": round(metrics_d["r2"], 4)},
    {"Model": "E – RF + Lag",         "Algorithm": "Random Forest",     "MAE": round(metrics_e["mae"], 2),
                                                                          "RMSE": round(metrics_e["rmse"], 2),
                                                                          "R²": round(metrics_e["r2"], 4)},
])
print(summary.to_string(index=False))

print("\n" + "=" * 60)
print("RESIDUAL INSIGHTS – Model E (RF + Lag)")
print("=" * 60)

print("\nMean absolute residual by hour (top 6 worst):")
print(
    residual_df.groupby("hour")["abs_residual"]
    .mean().sort_values(ascending=False).head(6).to_string()
)

print("\nMean absolute residual by canton:")
print(
    residual_df.groupby("canton")["abs_residual"]
    .mean().sort_values(ascending=False).to_string()
)

print("\n" + "=" * 60)
print("EXPORTS COMPLETE")
print("=" * 60)
print("Updated:  data_processed/regression_model_metrics.csv")
print("Updated:  data_processed/regression_coefficients.csv")
print("Created:  data_processed/regression_residuals.csv")
