"""
build_scenario_predictions_LR.py
================================
Builds a scenario prediction table for interactive use in Power BI, based on
a Linear Regression model.

Rationale
---------
Power BI cannot easily drive a model that needs hour-by-hour lag values as
input (the user would have to provide the previous day's consumption via a
slicer). Instead, we train on a reduced, interactively controllable feature
set — canton, season, day_type, hour, temperature_bucket — and pre-compute
the expected consumption for every possible combination of these slicer
values. Power BI then joins the slicer selection against this lookup table.

Counterpart: build_scenario_predictions_RF.py uses the same feature set but
swaps LinearRegression for RandomForestRegressor.

Output: ../data_processed/scenario_predictions.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import OneHotEncoder
from sklearn.impute          import SimpleImputer
from sklearn.linear_model    import LinearRegression
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Configuration -----------------------------------------------------------

INPUT_FILE  = Path("../data_processed/feature_dataset.csv")
OUTPUT_FILE = Path("../data_processed/scenario_predictions.csv")

# Possible names for the target column in different dataset versions — we
# take the first one that actually exists.
TARGET_CANDIDATES = [
    "consumption_mwh", "electricity_consumption_mwh",
    "stromverbrauch_mwh", "consumption",
]

# Numeric columns that may use a comma decimal and must be coerced to float.
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

# Feature set used for the scenario model.
# Lag features are intentionally excluded — they cannot be meaningfully
# chosen via a Power BI slicer.
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
    """
    Derive hour / month / day_type / season from the timestamp. Using labels
    rather than raw integers (e.g. "Winter" instead of month 12) makes the
    resulting scenario table self-explanatory in Power BI.
    """
    df = df.copy()
    df["timestamp"]   = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"]        = df["timestamp"].dt.hour.astype("Int64")
    df["month"]       = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Saturday and Sunday are separated — Sunday consumption is typically even
    # lower than Saturday, so collapsing both into "Weekend" would hide signal.
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

# Treat `hour` as a categorical rather than numeric predictor — consumption
# profiles across the day are non-linear, so a numeric hour would force the
# linear model into a straight line across 24h and lose all daily structure.
model_df["hour"] = model_df["hour"].astype("string")


# --- Train / test split ------------------------------------------------------

X = model_df[SCENARIO_FEATURES]
y = model_df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --- Train model -------------------------------------------------------------
# All five features are categorical, so the pipeline only needs a
# one-hot-encoder (plus imputer for safety). drop="first" avoids the dummy
# trap; handle_unknown="ignore" keeps prediction safe on unseen values.

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(drop="first", handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("cat", categorical_transformer, SCENARIO_FEATURES),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor",    LinearRegression()),
])

print("Training linear regression ...")
model.fit(X_train, y_train)


# --- Evaluate ---------------------------------------------------------------

y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")


# --- Generate scenario grid and predict -------------------------------------
# Build the full Cartesian product of all slicer values. Every combination
# becomes one row in the output table — Power BI then picks the matching row
# when the user adjusts slicers.

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
