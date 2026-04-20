# Energy-Weather Project — Data Processing

This document describes the individual scripts of the data pipeline, what they produce, and in which order they depend on each other. For an overview of the project, the research question, and the Power BI deliverable, see [`../readme.md`](../readme.md).

## Pipeline orchestration

### `src/run_pipeline.py`
Orchestrator that runs every step of the pipeline in the correct dependency order. Each script is executed as an isolated subprocess, and the pipeline stops immediately on the first non-zero exit code so a broken upstream step can never silently corrupt downstream outputs.

Key features:
- Streams each script's output live to the console
- Prints a per-step timing summary at the end
- `--list` / `--dry-run` to inspect the plan without executing anything
- `--start N` / `--stop N` to run a sub-range (resume after a failure, or rerun only the analysis steps)
- On failure, prints the exact command to resume after fixing the problem

## Data preparation

### `src/build_electricity_dataset.py`
Reads the raw Swissgrid Excel files, keeps only the consumption columns for the seven canton groups, aggregates the 15-minute values to hourly sums, converts kWh to MWh, and writes a single long-format CSV.

Output:
- `electricity_hourly_2015_2026.csv` — `timestamp`, `canton_code`, `consumption_mwh`

### `src/build_weather_dataset.py`
Reads the raw MeteoSwiss station CSVs, keeps the selected meteorological variables, maps each station to its canton group, restricts the data to the common analysis window (2015–2025), and runs quality checks (no duplicates, complete hour × canton grid).

Output:
- `weather_dataset.csv` — hourly weather measurements per canton group

### `src/build_dim_location.py`
Builds the location dimension for the star schema. Enriches the seven station metadata rows with human-readable canton names and keeps the columns relevant for mapping and reporting (coordinates, elevation, station owner).

Output:
- `dim_location.csv` — spatial dimension used by Power BI

### `src/build_analytical_base.py`
Merges the electricity and weather datasets on `(timestamp, canton)`. Both sides are first restricted to their common time window and their join keys are standardised. The merge is an inner join with `validate="one_to_one"` so any duplicate key on either side is caught immediately rather than silently multiplying rows.

Output:
- `analytical_base.csv` — consolidated fact table at hour × canton granularity

### `src/build_dim_time.py`
Derives a time dimension from the unique timestamps of the analytical base. Attributes include year, quarter, month, ISO week, day of week, hour, season, weekend flag, business-hour flag, and four-part hour buckets (Night / Morning / Afternoon / Evening).

Output:
- `dim_time.csv` — time dimension used by Power BI

### `src/build_feature_dataset.py`
Extends the analytical base with analytical and modelling features:

- Temperature-based: `hdd` (Heating Degree Days, base 18 °C), `cdd` (Cooling Degree Days, base 18 °C), `is_extreme_cold`, `is_extreme_heat`
- Precipitation: `precipitation_flag` (any), `heavy_precipitation_flag` (≥ 5 mm/h)
- Temporal lags per canton: `consumption_lag_24h`, `consumption_lag_168h`

Output:
- `feature_dataset.csv` — central fact table for correlation, regression, and scenario analysis

## Analysis

### `src/analyse_correlations.py`
Computes Pearson correlations between `consumption_mwh` and every numeric feature across multiple analytical scopes (overall, per canton, per season, per hour bucket, weekday/weekend, etc.). Results are emitted in long format so Power BI can slice the single table directly without joins.

Outputs:
- `correlations_target_long.csv` — correlation of each feature with the target, enriched with feature labels, thematic categories, ranking and direction (main input for Power BI)
- `correlations_full_pairs_long.csv` — pairwise correlations between all numeric features (for multicollinearity diagnostics)

### `src/analyse_regression.py`
Fits three linear regression models to quantify the effect of weather, time, and regional structure on consumption:

| Model | Features |
|-------|----------|
| A | Weather + time + lag features |
| B | Weather + time only |
| C | Weather + time + canton (one-hot encoded) |

All three use the same 80/20 train-test split (`random_state=42`) so their metrics are directly comparable.

Outputs:
- `regression_model_metrics.csv` — R², MAE, RMSE per model
- `regression_coefficients.csv` — coefficients with feature-type classification
- `regression_predictions_sample.csv` — 5 000 actual vs. predicted pairs (based on Model C) for visual checks

### `src/analyse_regression_extended.py`
Extends `analyse_regression.py` with two Random Forest models and must be executed **after** it (reads and appends to the CSVs produced in step 8).

| Model | Algorithm | Features | Linear counterpart |
|-------|-----------|----------|-------------------|
| D | Random Forest | Weather + time + canton | Model C |
| E | Random Forest | Weather + time + lag    | Model A |

Outputs:
- `regression_model_metrics.csv` — updated (D + E appended, `algorithm` and `model_label` columns added)
- `regression_coefficients.csv` — updated (RF feature importances appended)
- `regression_residuals.csv` — new; residual analysis of Model E (best R²) for Power BI

## Scenario predictions (what-if)

### `src/build_scenario_predictions_LR.py`
Builds an interactive, explainable scenario model using five interpretable features: canton, season, day type, hour, and temperature bucket. Fits a linear regression, generates the full Cartesian product of slicer values (~10 000 combinations), and writes one predicted consumption value per combination.

Output:
- `scenario_predictions.csv` — **wired into the Power BI what-if page**. The user selects a combination of filters and immediately sees the model-based estimate of the daily consumption profile.

The linear model is the deliberate choice here because it produces realistic and interpretable hourly profiles. The Random Forest variant (below) averages too aggressively under fixed scenario conditions and therefore produces flat, uninformative daily curves.

### `src/build_scenario_predictions_RF.py`
Random Forest variant of `build_scenario_predictions_LR.py` (analogous to Model D in `analyse_regression_extended.py`: RF with canton, no lag features). Globally achieves a higher R² (~0.93 vs. ~0.85), but less suitable for visualising typical daily profiles under fixed scenarios because the hourly variation is strongly smoothed.

Kept for model comparison and archival purposes.

Output:
- `scenario_predictions_rf.csv` — same structure as `scenario_predictions.csv`, **not wired into Power BI** (see reasoning above)

## Execution order

For full-pipeline runs, use the orchestrator:

```bash
cd src
python run_pipeline.py
```

Individual scripts (manual execution) must be run in this order:

```
build_electricity_dataset.py
build_weather_dataset.py
build_dim_location.py
build_analytical_base.py
build_dim_time.py
build_feature_dataset.py
analyse_correlations.py
analyse_regression.py
analyse_regression_extended.py   # must run after analyse_regression.py
build_scenario_predictions_LR.py
build_scenario_predictions_RF.py # optional, for model comparison only
```

## Folder structure

### `data_raw/`
Original raw inputs from Swissgrid (electricity) and MeteoSwiss (weather).

### `data_processed/`
All cleaned, joined, and analytical CSV outputs consumed by Power BI.

### `archive/`
Earlier versions, scratch scripts, and files no longer part of the active pipeline.

## Goal

Establish a clean, reproducible data basis for feature engineering, analyses, predictions, and visualisation in Power BI.
