"""
build_scenario_predictions_RF.py
================================
Builds a scenario prediction table for interactive use in Power BI, based on
a Random Forest model.

This is the Random Forest counterpart of build_scenario_predictions_LR.py.
The two scripts share the same feature set, the same scenario grid and the
same output schema — only the estimator differs. The RF variant captures
non-linear interactions between hour, season and temperature that a linear
model cannot, which typically yields a noticeably higher R².

The chosen hyperparameters mirror Model D from analyse_regression_extended.py
(RF + Canton, no lag features) so the scenario predictions are methodologically
consistent with the wider regression analysis.

Output: ../data_processed/scenario_predictions_rf.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Configuration -----------------------------------------------------------

INPUT_FILE  = Path("../data_processed/feature_dataset.csv")
OUTPUT_FILE = Path("../data_processed/scenario_predictions_rf.csv")

TARGET_CANDIDATES = [
    "consumption_mwh", "electricity_consumption_mwh",
    "stromverbrauch_mwh", "consumption",
]

NUMERIC_COLUMNS = [
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

# Only features the user can choose via Power BI slicers — no lag features.
SCENARIO_FEATURES = [
    "canton", "season", "day_type", "hour", "temperature_bucket",
]


# --- Helpers -----------------------------------------------------------------

def find_target_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate that actually exists as a column."""
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No target column found. Available columns: {list(df.columns)}")


def read_csv_robust(file_path: Path) -> pd.DataFrame:
    """Try common separators until one parses successfully."""
    last_error = None
    for sep in [";", ","]:
        try:
            return pd.read_csv(file_path, sep=sep, encoding="utf-8-sig", engine="python")
        except Exception as e:
            last_error = e
    raise ValueError(f"CSV could not be read:\n{last_error}")


def convert_decimal_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace comma decimals with dots and coerce the given columns to numeric."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col].astype("string")
                       .str.replace(",", ".", regex=False)
                       .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour / month / day_type / season from the timestamp."""
    df = df.copy()
    df["timestamp"]   = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"]        = df["timestamp"].dt.hour.astype("Int64")
    df["month"]       = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Saturday / Sunday stay separate — Sunday consumption is typically even
    # lower than Saturday, collapsing them would hide signal.
    df["day_type"] = df["day_of_week"].map({
        0: "Weekday", 1: "Weekday", 2: "Weekday", 3: "Weekday", 4: "Weekday",
        5: "Saturday", 6: "Sunday",
    })

    df["season"] = df["month"].map({
        12: "Winter",  1: "Winter",  2: "Winter",
         3: "Spring",  4: "Spring",  5: "Spring",
         6: "Summer",  7: "Summer",  8: "Summer",
         9: "Autumn", 10: "Autumn", 11: "Autumn",
    })
    return df


def add_temperature_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Discretise temperature into five buckets used as a slicer in Power BI."""
    df = df.copy()
    if "temperature_2m" not in df.columns:
        raise ValueError("Column 'temperature_2m' is missing.")
    df["temperature_bucket"] = pd.cut(
        df["temperature_2m"],
        bins=[-999, 0, 10, 20, 30, 999],
        labels=["very_cold", "cold", "mild", "warm", "hot"],
    )
    return df


# --- Load and prepare --------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File not found: {INPUT_FILE}")

print("Loading feature dataset ...")
df = read_csv_robust(INPUT_FILE)
print(f"  Shape: {df.shape}")

df = convert_decimal_columns(df, NUMERIC_COLUMNS)
df = add_time_features(df)
df = add_temperature_bucket(df)
target_col = find_target_column(df, TARGET_CANDIDATES)


# --- Build model dataset -----------------------------------------------------

required_cols = SCENARIO_FEATURES + [target_col]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns required for modelling: {missing}")

model_df = df[required_cols].copy().dropna(subset=[target_col])

# `hour` as a string so the pipeline one-hot-encodes it. The RF could in
# principle handle it as an integer, but keeping it categorical guarantees
# identical scenario combinations across LR and RF variants.
model_df["hour"] = model_df["hour"].astype("string")


# --- Train / test split ------------------------------------------------------

X = model_df[SCENARIO_FEATURES]
y = model_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --- Train model -------------------------------------------------------------
# Same preprocessing as the LR variant; only the estimator is different.
# Hyperparameters match Model D in analyse_regression_extended.py:
#   n_estimators=200 / max_depth=12 / min_samples_leaf=5
# These values balance accuracy and training time on ~670k hourly rows and
# control overfitting.

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(drop="first", handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("cat", categorical_transformer, SCENARIO_FEATURES),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,  # use all available CPU cores
    )),
])

print("Training random forest ...")
model.fit(X_train, y_train)


# --- Evaluate ---------------------------------------------------------------

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")


# --- Generate scenario grid and predict -------------------------------------

canton_values      = sorted(df["canton"].dropna().unique().tolist())
season_values      = ["Winter", "Spring", "Summer", "Autumn"]
day_type_values    = ["Weekday", "Saturday", "Sunday"]
hour_values        = [str(h) for h in range(24)]
temperature_values = ["very_cold", "cold", "mild", "warm", "hot"]

scenario_predictions = pd.MultiIndex.from_product(
    [canton_values, season_values, day_type_values, hour_values, temperature_values],
    names=SCENARIO_FEATURES,
).to_frame(index=False)

print(f"Scenario combinations: {len(scenario_predictions):,}")

scenario_predictions["predicted_consumption_mwh"] = model.predict(scenario_predictions)


# --- Export -----------------------------------------------------------------

scenario_predictions.to_csv(OUTPUT_FILE, sep=";", index=False, decimal=",")
print(f"Saved: {OUTPUT_FILE}")
