import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1) Pfade
# ------------------------------------------------------------

input_feature = Path("../data_processed/feature_dataset.csv")
input_time = Path("../data_processed/dim_time.csv")
output_folder = Path("../data_processed")
output_folder.mkdir(parents=True, exist_ok=True)

output_target = output_folder / "correlations_target_long.csv"
output_pairs = output_folder / "correlations_full_pairs_long.csv"

# ------------------------------------------------------------
# 2) Daten laden
# ------------------------------------------------------------

df = pd.read_csv(input_feature, sep=";", decimal=",")
dim_time = pd.read_csv(input_time, sep=";", decimal=",")

# ------------------------------------------------------------
# 3) Zeitstempel konvertieren
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])
dim_time["timestamp"] = pd.to_datetime(dim_time["timestamp"])

# ------------------------------------------------------------
# 4) Join mit Zeitdimension
# ------------------------------------------------------------

time_cols = [
    "timestamp",
    "year",
    "quarter",
    "month",
    "week_of_year",
    "day_of_week",
    "day_name",
    "hour",
    "is_weekend",
    "season",
    "year_month",
    "is_business_hour",
    "hour_bucket"
]

df = df.merge(dim_time[time_cols], on="timestamp", how="left")

# ------------------------------------------------------------
# 5) Numerische Spalten definieren
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
    "consumption_lag_168h",
    "year",
    "quarter",
    "month",
    "week_of_year",
    "day_of_week",
    "hour",
    "is_weekend",
    "is_business_hour"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

target = "consumption_mwh"

# ------------------------------------------------------------
# 6) Analyse-Sichten definieren
# ------------------------------------------------------------

groupings = [
    ("overall", None),
    ("canton", "canton"),
    ("season", "season"),
    ("is_business_hour", "is_business_hour"),
    ("is_weekend", "is_weekend"),
    ("hour_bucket", "hour_bucket"),
    ("day_name", "day_name"),
    ("hour", "hour"),
    ("month", "month"),
    ("quarter", "quarter")
]

# ------------------------------------------------------------
# 7) Labels / Kategorien für Power BI
# ------------------------------------------------------------

feature_label_mapping = {
    "temperature_2m": "Temperature",
    "precipitation": "Precipitation",
    "humidity_rel": "Relative Humidity",
    "global_radiation": "Global Radiation",
    "sunshine_duration": "Sunshine Duration",
    "wind_speed": "Wind Speed",
    "wind_direction": "Wind Direction",
    "pressure_station": "Station Pressure",
    "pressure_qnh": "Pressure QNH",
    "vapour_pressure": "Vapour Pressure",
    "dew_point": "Dew Point",
    "hdd": "Heating Degree Days",
    "cdd": "Cooling Degree Days",
    "is_extreme_cold": "Extreme Cold",
    "is_extreme_heat": "Extreme Heat",
    "precipitation_flag": "Precipitation Flag",
    "heavy_precipitation_flag": "Heavy Precipitation Flag",
    "consumption_lag_24h": "Consumption Lag (24h)",
    "consumption_lag_168h": "Consumption Lag (168h)",
    "year": "Year",
    "quarter": "Quarter",
    "month": "Month",
    "week_of_year": "Week of Year",
    "day_of_week": "Day of Week",
    "hour": "Hour",
    "is_weekend": "Weekend",
    "is_business_hour": "Business Hour"
}

scope_type_label_mapping = {
    "overall": "Overall",
    "canton": "Canton",
    "season": "Season",
    "is_business_hour": "Business Hour",
    "is_weekend": "Weekend",
    "hour_bucket": "Hour Bucket",
    "day_name": "Day Name",
    "hour": "Hour",
    "month": "Month",
    "quarter": "Quarter"
}

def categorize_feature(feature: str) -> str:
    if feature.startswith("consumption_lag_"):
        return "Lag / Historical"
    elif feature in ["temperature_2m", "hdd", "cdd", "dew_point", "vapour_pressure"]:
        return "Temperature / Thermal"
    elif feature in ["precipitation", "precipitation_flag", "heavy_precipitation_flag", "humidity_rel"]:
        return "Precipitation / Humidity"
    elif feature in ["global_radiation", "sunshine_duration"]:
        return "Solar / Light"
    elif feature in ["wind_speed", "wind_direction", "pressure_station", "pressure_qnh"]:
        return "Atmospheric"
    elif feature in ["is_extreme_cold", "is_extreme_heat"]:
        return "Extreme Weather Flags"
    elif feature in ["year", "quarter", "month", "week_of_year", "day_of_week", "hour", "is_weekend", "is_business_hour"]:
        return "Time Features"
    else:
        return "Other"

# ------------------------------------------------------------
# 8) Hilfsfunktionen: Korrelationen berechnen
# ------------------------------------------------------------

def compute_target_correlations(data: pd.DataFrame, scope_type: str, scope_value: str) -> pd.DataFrame:
    rows = []
    n_rows = len(data)

    corr = data[numeric_cols].corr()

    if target not in corr.columns:
        return pd.DataFrame()

    target_corr = corr[target].dropna()

    for feature, corr_value in target_corr.items():
        if feature == target:
            continue

        rows.append({
            "scope_type": scope_type,
            "scope_value": scope_value,
            "feature": feature,
            "target": target,
            "correlation": corr_value,
            "abs_correlation": abs(corr_value),
            "n_observations": n_rows
        })

    return pd.DataFrame(rows)


def compute_full_pair_correlations(data: pd.DataFrame, scope_type: str, scope_value: str) -> pd.DataFrame:
    rows = []

    corr = data[numeric_cols].corr()

    if corr.empty:
        return pd.DataFrame()

    cols = corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            feature_1 = cols[i]
            feature_2 = cols[j]
            corr_value = corr.loc[feature_1, feature_2]

            if pd.isna(corr_value):
                continue

            rows.append({
                "scope_type": scope_type,
                "scope_value": scope_value,
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": corr_value,
                "abs_correlation": abs(corr_value),
                "n_observations": len(data)
            })

    return pd.DataFrame(rows)

# ------------------------------------------------------------
# 9) Korrelationen für alle Sichten berechnen
# ------------------------------------------------------------

target_results = []
pair_results = []

for scope_type, group_col in groupings:
    print(f"Running scope: {scope_type}")

    if group_col is None:
        subset = df.copy()

        target_corr_df = compute_target_correlations(
            data=subset,
            scope_type=scope_type,
            scope_value="all"
        )
        pair_corr_df = compute_full_pair_correlations(
            data=subset,
            scope_type=scope_type,
            scope_value="all"
        )

        target_results.append(target_corr_df)
        pair_results.append(pair_corr_df)

    else:
        for scope_value, subset in df.groupby(group_col, dropna=False):
            if len(subset) < 30:
                continue

            target_corr_df = compute_target_correlations(
                data=subset,
                scope_type=scope_type,
                scope_value=str(scope_value)
            )
            pair_corr_df = compute_full_pair_correlations(
                data=subset,
                scope_type=scope_type,
                scope_value=str(scope_value)
            )

            target_results.append(target_corr_df)
            pair_results.append(pair_corr_df)

# ------------------------------------------------------------
# 10) Zusammenführen
# ------------------------------------------------------------

correlations_target_long = pd.concat(target_results, ignore_index=True)
correlations_full_pairs_long = pd.concat(pair_results, ignore_index=True)

# ------------------------------------------------------------
# 11) Power-BI-freundliche Zusatzspalten
# ------------------------------------------------------------

correlations_target_long["feature_label"] = correlations_target_long["feature"].map(feature_label_mapping)
correlations_target_long["feature_label"] = correlations_target_long["feature_label"].fillna(correlations_target_long["feature"])

correlations_target_long["feature_category"] = correlations_target_long["feature"].apply(categorize_feature)

correlations_target_long["scope_type_label"] = correlations_target_long["scope_type"].map(scope_type_label_mapping)
correlations_target_long["scope_type_label"] = correlations_target_long["scope_type_label"].fillna(correlations_target_long["scope_type"])

correlations_target_long["scope"] = (
    correlations_target_long["scope_type_label"].astype(str)
    + " | "
    + correlations_target_long["scope_value"].astype(str)
)

correlations_target_long["rank_abs_corr"] = correlations_target_long.groupby(
    ["scope_type", "scope_value"]
)["abs_correlation"].rank(method="dense", ascending=False)

correlations_target_long["rank_corr_desc"] = correlations_target_long.groupby(
    ["scope_type", "scope_value"]
)["correlation"].rank(method="dense", ascending=False)

correlations_target_long["correlation_direction"] = correlations_target_long["correlation"].apply(
    lambda x: "Positive" if x >= 0 else "Negative"
)

# Optional auch für Pair-Tabelle etwas lesbarer machen
correlations_full_pairs_long["scope_type_label"] = correlations_full_pairs_long["scope_type"].map(scope_type_label_mapping)
correlations_full_pairs_long["scope_type_label"] = correlations_full_pairs_long["scope_type_label"].fillna(correlations_full_pairs_long["scope_type"])

correlations_full_pairs_long["scope"] = (
    correlations_full_pairs_long["scope_type_label"].astype(str)
    + " | "
    + correlations_full_pairs_long["scope_value"].astype(str)
)

correlations_full_pairs_long["feature_1_label"] = correlations_full_pairs_long["feature_1"].map(feature_label_mapping)
correlations_full_pairs_long["feature_1_label"] = correlations_full_pairs_long["feature_1_label"].fillna(correlations_full_pairs_long["feature_1"])

correlations_full_pairs_long["feature_2_label"] = correlations_full_pairs_long["feature_2"].map(feature_label_mapping)
correlations_full_pairs_long["feature_2_label"] = correlations_full_pairs_long["feature_2_label"].fillna(correlations_full_pairs_long["feature_2"])

# ------------------------------------------------------------
# 12) Sortierung
# ------------------------------------------------------------

correlations_target_long = correlations_target_long.sort_values(
    by=["scope_type", "scope_value", "rank_abs_corr", "feature"],
    ascending=[True, True, True, True]
)

correlations_full_pairs_long = correlations_full_pairs_long.sort_values(
    by=["scope_type", "scope_value", "abs_correlation"],
    ascending=[True, True, False]
)

# ------------------------------------------------------------
# 13) Export
# ------------------------------------------------------------

correlations_target_long.to_csv(output_target, sep=";", decimal=",", index=False)
correlations_full_pairs_long.to_csv(output_pairs, sep=";", decimal=",", index=False)

# ------------------------------------------------------------
# 14) Konsolenoutput
# ------------------------------------------------------------

print("\nSaved files:")
print(output_target)
print(output_pairs)

print("\nPreview: correlations_target_long")
print(correlations_target_long.head(20))

print("\nPreview: correlations_full_pairs_long")
print(correlations_full_pairs_long.head(20))