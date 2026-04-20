"""
analyse_correlations.py
=======================
Computes Pearson correlations between the target variable (consumption_mwh)
and all numeric features of the feature dataset, sliced along multiple
analytical scopes (overall, canton, season, weekday, hour bucket, ...).

Results are exported in long format — one row per (scope, feature) — which
is the shape Power BI prefers: no joins with fact tables needed, filters
and ranks can be applied directly on this single table.

Outputs
-------
- correlations_target_long.csv       correlations vs. target, enriched with
                                     labels, ranks and categories
- correlations_full_pairs_long.csv   all pairwise correlations (for a full
                                     correlation matrix view in Power BI)
"""

import pandas as pd
from pathlib import Path

# --- Configuration -----------------------------------------------------------

INPUT_FEATURE = Path("../data_processed/feature_dataset.csv")
INPUT_TIME    = Path("../data_processed/dim_time.csv")
OUTPUT_FOLDER = Path("../data_processed")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_TARGET = OUTPUT_FOLDER / "correlations_target_long.csv"
OUTPUT_PAIRS  = OUTPUT_FOLDER / "correlations_full_pairs_long.csv"

TARGET = "consumption_mwh"

# Minimum rows required per scope — below this, sample correlations become
# unstable / misleading, so we skip the scope entirely.
MIN_OBSERVATIONS = 30

# All numeric columns involved in the correlation analysis. Anything not in
# this list (e.g. categorical labels, timestamps) is ignored.
NUMERIC_COLS = [
    TARGET,
    # Weather
    "temperature_2m", "precipitation", "humidity_rel",
    "global_radiation", "sunshine_duration",
    "wind_speed", "wind_direction",
    "pressure_station", "pressure_qnh",
    "vapour_pressure", "dew_point",
    # Derived weather
    "hdd", "cdd",
    "is_extreme_cold", "is_extreme_heat",
    "precipitation_flag", "heavy_precipitation_flag",
    # Lag features
    "consumption_lag_24h", "consumption_lag_168h",
    # Time features (kept numeric so they participate in the correlation matrix)
    "year", "quarter", "month", "week_of_year",
    "day_of_week", "hour",
    "is_weekend", "is_business_hour",
]

# Scopes along which correlations are computed. Each tuple is
# (scope_type_label, grouping column or None). `None` means "whole dataset".
GROUPINGS = [
    ("overall",          None),
    ("canton",           "canton"),
    ("season",           "season"),
    ("is_business_hour", "is_business_hour"),
    ("is_weekend",       "is_weekend"),
    ("hour_bucket",      "hour_bucket"),
    ("day_name",         "day_name"),
    ("hour",             "hour"),
    ("month",            "month"),
    ("quarter",          "quarter"),
]

# Human-readable labels for Power BI axis titles, legends and slicers.
FEATURE_LABEL_MAPPING = {
    "temperature_2m":           "Temperature",
    "precipitation":            "Precipitation",
    "humidity_rel":             "Relative Humidity",
    "global_radiation":         "Global Radiation",
    "sunshine_duration":        "Sunshine Duration",
    "wind_speed":               "Wind Speed",
    "wind_direction":           "Wind Direction",
    "pressure_station":         "Station Pressure",
    "pressure_qnh":             "Pressure QNH",
    "vapour_pressure":          "Vapour Pressure",
    "dew_point":                "Dew Point",
    "hdd":                      "Heating Degree Days",
    "cdd":                      "Cooling Degree Days",
    "is_extreme_cold":          "Extreme Cold",
    "is_extreme_heat":          "Extreme Heat",
    "precipitation_flag":       "Precipitation Flag",
    "heavy_precipitation_flag": "Heavy Precipitation Flag",
    "consumption_lag_24h":      "Consumption Lag (24h)",
    "consumption_lag_168h":     "Consumption Lag (168h)",
    "year":                     "Year",
    "quarter":                  "Quarter",
    "month":                    "Month",
    "week_of_year":             "Week of Year",
    "day_of_week":              "Day of Week",
    "hour":                     "Hour",
    "is_weekend":               "Weekend",
    "is_business_hour":         "Business Hour",
}

SCOPE_TYPE_LABEL_MAPPING = {
    "overall":          "Overall",
    "canton":           "Canton",
    "season":           "Season",
    "is_business_hour": "Business Hour",
    "is_weekend":       "Weekend",
    "hour_bucket":      "Hour Bucket",
    "day_name":         "Day Name",
    "hour":             "Hour",
    "month":            "Month",
    "quarter":          "Quarter",
}


# --- Helpers -----------------------------------------------------------------

def categorize_feature(feature: str) -> str:
    """Group features into thematic categories for dashboard filtering."""
    if feature.startswith("consumption_lag_"):
        return "Lag / Historical"
    if feature in ("temperature_2m", "hdd", "cdd", "dew_point", "vapour_pressure"):
        return "Temperature / Thermal"
    if feature in ("precipitation", "precipitation_flag",
                   "heavy_precipitation_flag", "humidity_rel"):
        return "Precipitation / Humidity"
    if feature in ("global_radiation", "sunshine_duration"):
        return "Solar / Light"
    if feature in ("wind_speed", "wind_direction",
                   "pressure_station", "pressure_qnh"):
        return "Atmospheric"
    if feature in ("is_extreme_cold", "is_extreme_heat"):
        return "Extreme Weather Flags"
    if feature in ("year", "quarter", "month", "week_of_year",
                   "day_of_week", "hour", "is_weekend", "is_business_hour"):
        return "Time Features"
    return "Other"


def compute_target_correlations(
    data: pd.DataFrame, scope_type: str, scope_value: str
) -> pd.DataFrame:
    """Correlation of every numeric feature vs. the target within one scope."""
    corr = data[NUMERIC_COLS].corr()
    if TARGET not in corr.columns:
        return pd.DataFrame()

    target_corr = corr[TARGET].dropna()
    rows = []
    for feature, value in target_corr.items():
        if feature == TARGET:
            continue
        rows.append({
            "scope_type":      scope_type,
            "scope_value":     scope_value,
            "feature":         feature,
            "target":          TARGET,
            "correlation":     value,
            "abs_correlation": abs(value),
            "n_observations":  len(data),
        })
    return pd.DataFrame(rows)


def compute_full_pair_correlations(
    data: pd.DataFrame, scope_type: str, scope_value: str
) -> pd.DataFrame:
    """All pairwise feature correlations within one scope (upper triangle only)."""
    corr = data[NUMERIC_COLS].corr()
    if corr.empty:
        return pd.DataFrame()

    cols = corr.columns.tolist()
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.isna(value):
                continue
            rows.append({
                "scope_type":      scope_type,
                "scope_value":     scope_value,
                "feature_1":       cols[i],
                "feature_2":       cols[j],
                "correlation":     value,
                "abs_correlation": abs(value),
                "n_observations":  len(data),
            })
    return pd.DataFrame(rows)


# --- Build -------------------------------------------------------------------

print("Loading datasets ...")
df       = pd.read_csv(INPUT_FEATURE, sep=";", decimal=",")
dim_time = pd.read_csv(INPUT_TIME,    sep=";", decimal=",")

df["timestamp"]       = pd.to_datetime(df["timestamp"])
dim_time["timestamp"] = pd.to_datetime(dim_time["timestamp"])

# Enrich the feature dataset with calendar attributes required by some scopes
# (season, hour_bucket, is_business_hour, ...).
time_cols = [
    "timestamp", "year", "quarter", "month", "week_of_year",
    "day_of_week", "day_name", "hour",
    "is_weekend", "season", "year_month",
    "is_business_hour", "hour_bucket",
]
df = df.merge(dim_time[time_cols], on="timestamp", how="left")

# Make sure every column that participates in the correlation matrix is numeric.
for col in NUMERIC_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Iterate all scopes and collect per-slice correlations.
target_results = []
pair_results   = []

for scope_type, group_col in GROUPINGS:
    print(f"  scope: {scope_type}")

    if group_col is None:
        # Whole-dataset scope ("Overall").
        target_results.append(compute_target_correlations(df, scope_type, "all"))
        pair_results.append(compute_full_pair_correlations(df, scope_type, "all"))
        continue

    # Grouped scope — one correlation matrix per group value.
    for scope_value, subset in df.groupby(group_col, dropna=False):
        if len(subset) < MIN_OBSERVATIONS:
            continue
        target_results.append(
            compute_target_correlations(subset, scope_type, str(scope_value))
        )
        pair_results.append(
            compute_full_pair_correlations(subset, scope_type, str(scope_value))
        )

correlations_target_long     = pd.concat(target_results, ignore_index=True)
correlations_full_pairs_long = pd.concat(pair_results,   ignore_index=True)


# --- Enrich with Power BI-friendly columns -----------------------------------

# Target table: feature labels, categories, ranks, sign indicator.
correlations_target_long["feature_label"] = (
    correlations_target_long["feature"].map(FEATURE_LABEL_MAPPING)
                                       .fillna(correlations_target_long["feature"])
)
correlations_target_long["feature_category"]   = correlations_target_long["feature"].apply(categorize_feature)
correlations_target_long["scope_type_label"]   = (
    correlations_target_long["scope_type"].map(SCOPE_TYPE_LABEL_MAPPING)
                                           .fillna(correlations_target_long["scope_type"])
)
correlations_target_long["scope"] = (
    correlations_target_long["scope_type_label"].astype(str) + " | "
    + correlations_target_long["scope_value"].astype(str)
)

# Dense rank within each scope — 1 = strongest correlation (by absolute value).
correlations_target_long["rank_abs_corr"] = (
    correlations_target_long
      .groupby(["scope_type", "scope_value"])["abs_correlation"]
      .rank(method="dense", ascending=False)
)
correlations_target_long["rank_corr_desc"] = (
    correlations_target_long
      .groupby(["scope_type", "scope_value"])["correlation"]
      .rank(method="dense", ascending=False)
)
correlations_target_long["correlation_direction"] = (
    correlations_target_long["correlation"]
      .apply(lambda x: "Positive" if x >= 0 else "Negative")
)

# Pair table: same scope enrichment + readable feature labels.
correlations_full_pairs_long["scope_type_label"] = (
    correlations_full_pairs_long["scope_type"].map(SCOPE_TYPE_LABEL_MAPPING)
                                               .fillna(correlations_full_pairs_long["scope_type"])
)
correlations_full_pairs_long["scope"] = (
    correlations_full_pairs_long["scope_type_label"].astype(str) + " | "
    + correlations_full_pairs_long["scope_value"].astype(str)
)
correlations_full_pairs_long["feature_1_label"] = (
    correlations_full_pairs_long["feature_1"].map(FEATURE_LABEL_MAPPING)
                                              .fillna(correlations_full_pairs_long["feature_1"])
)
correlations_full_pairs_long["feature_2_label"] = (
    correlations_full_pairs_long["feature_2"].map(FEATURE_LABEL_MAPPING)
                                              .fillna(correlations_full_pairs_long["feature_2"])
)


# --- Sort and export ---------------------------------------------------------

correlations_target_long = correlations_target_long.sort_values(
    by=["scope_type", "scope_value", "rank_abs_corr", "feature"],
    ascending=[True, True, True, True],
)
correlations_full_pairs_long = correlations_full_pairs_long.sort_values(
    by=["scope_type", "scope_value", "abs_correlation"],
    ascending=[True, True, False],
)

correlations_target_long.to_csv(OUTPUT_TARGET,     sep=";", decimal=",", index=False)
correlations_full_pairs_long.to_csv(OUTPUT_PAIRS,  sep=";", decimal=",", index=False)

print(f"Saved: {OUTPUT_TARGET} ({len(correlations_target_long):,} rows)")
print(f"Saved: {OUTPUT_PAIRS} ({len(correlations_full_pairs_long):,} rows)")
