# Energy-Weather Project

Data-driven analysis of how weather influences hourly electricity consumption in Switzerland (2015–2025). The project builds a clean, reproducible pipeline from raw open data to an interactive Power BI dashboard — covering data preparation, feature engineering, correlation and regression analysis, and scenario-based predictions.

## Research question

> *How strongly — and in which patterns — does weather/climate influence electricity consumption across Swiss cantons?*

To answer this, the project combines two public data sources at an hourly resolution per canton group, enriches them with engineered features (HDD/CDD, lags, flags, time attributes), and quantifies the relationships using correlation analysis, multiple linear regression models, and random-forest models. A what-if scenario model feeds an interactive Power BI report.

## Data sources

| Source | Provider | Content | Granularity |
|---|---|---|---|
| Electricity consumption | [Swissgrid — Energy Statistics](https://www.swissgrid.ch/en/home/operation/grid-data/generation.html) | Quarter-Hourly electricity consumption per canton group | Quarter-Hourly, 7 canton groups |
| Weather | [MeteoSwiss — IDAweb / Open Data](https://www.meteoswiss.admin.ch/services-and-publications/service/weather-and-climate-products/data-portal-for-teaching-and-research.html) | Temperature, precipitation, humidity, radiation, wind, pressure, etc. | Hourly, one representative station per canton group |

Seven canton groups are used (matching Swissgrid's reporting grid): **BE_JU, GE_VD, GR, SG, TI, VS, ZH_SH**.

## Folder structure

```
energy-weather-project/
├── src/                         # Python pipeline scripts
│   ├── run_pipeline.py          #   orchestrator — runs the full pipeline
│   └── build_*.py / analyse_*.py
├── data_raw/                    # Raw inputs
│   ├── electricity/             #   Swissgrid Excel files
│   └── weather/                 #   MeteoSwiss CSV files per canton
├── data_processed/              # Cleaned, joined, and analytical outputs (CSV)
├── documentation/                        
│   └── dashboard.md             # Screenshots of the Dashboard
│   └── data_processing.md       # Technical documentation of the pipeline
├── powerbi/                     # Power BI dashboard (.pbix) and assets
├── archive/                     # Legacy files, scratch scripts, older README
├── requirements.txt             # Required Python packages
└── readme.md                    # This file
```

## Setup

**Prerequisites:** Python 3.10+ and a working internet connection (only needed for the initial package installation).

Create and activate a virtual environment, then install dependencies:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate

# Required packages
pip install pandas numpy scikit-learn openpyxl xlrd
```

`openpyxl` reads the newer `.xlsx` Swissgrid files (2020+); `xlrd` reads the older `.xls` files (2015–2019). Both are required for the electricity step.

All scripts are written to be run from the `src/` folder (they use relative paths like `../data_processed/...`), so `cd src` before executing anything.

## Running the pipeline

The recommended way to build the full dataset is via the orchestrator **`src/run_pipeline.py`**. It runs all 11 steps in the correct dependency order, streams each script's output live, stops immediately on the first failure, and prints a timing summary at the end.

```bash
cd src
python run_pipeline.py              # full pipeline
python run_pipeline.py --list       # show the planned steps and exit
python run_pipeline.py --dry-run    # verify each script exists, run nothing
python run_pipeline.py --start 7    # resume at step 7 (skips 1–6)
python run_pipeline.py --start 7 --stop 9   # run only steps 7–9
```

If a step fails, the orchestrator aborts the pipeline and prints the exact command to resume after you've fixed the problem:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
STEP FAILED:  Build analytical base
Script:       build_analytical_base.py
Exit code:    1

Pipeline stopped. After fixing the error, resume with:
    python run_pipeline.py --start 4
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Downstream steps never run after a failure, so a broken upstream script cannot silently corrupt the Power BI inputs.

### Manual execution (alternative)

For development or to rerun a single step in isolation, the scripts can also be executed one at a time in the same order as the orchestrator:

```bash
cd src
python build_electricity_dataset.py
python build_weather_dataset.py
python build_dim_location.py
python build_analytical_base.py
python build_dim_time.py
python build_feature_dataset.py
python analyse_correlations.py
python analyse_regression.py
python analyse_regression_extended.py
python build_scenario_predictions_LR.py
python build_scenario_predictions_RF.py   # optional
```

## Pipeline steps and outputs

| # | Script | Output(s) in `data_processed/` |
|---|---|---|
| 1 | `build_electricity_dataset.py`      | `electricity_hourly_2015_2026.csv` |
| 2 | `build_weather_dataset.py`          | `weather_dataset.csv` |
| 3 | `build_dim_location.py`             | `dim_location.csv` |
| 4 | `build_analytical_base.py`          | `analytical_base.csv` |
| 5 | `build_dim_time.py`                 | `dim_time.csv` |
| 6 | `build_feature_dataset.py`          | `feature_dataset.csv` |
| 7 | `analyse_correlations.py`           | `correlations_target_long.csv`, `correlations_full_pairs_long.csv` |
| 8 | `analyse_regression.py`             | `regression_model_metrics.csv`, `regression_coefficients.csv`, `regression_predictions_sample.csv` |
| 9 | `analyse_regression_extended.py`    | updates the three files above, adds `regression_residuals.csv` (must run **after** step 8) |
| 10 | `build_scenario_predictions_LR.py` | `scenario_predictions.csv` |
| 11 | `build_scenario_predictions_RF.py` | `scenario_predictions_rf.csv` (optional — for model comparison only) |

See [`docs/data_processing.md`](documentation/data_processing.md) for per-script detail.

## Models

Five models are trained and compared:

| Model | Algorithm | Features | Purpose |
|---|---|---|---|
| A | Linear Regression | Weather + time + lag (24h, 168h) | Benchmark with autoregressive information |
| B | Linear Regression | Weather + time | Isolate the pure weather/time effect |
| C | Linear Regression | Weather + time + canton | Quantify regional differences |
| D | Random Forest | Weather + time + canton (no lag) | Non-linear counterpart to Model C |
| E | Random Forest | Weather + time + lag | Non-linear counterpart to Model A |

Results are exported to `data_processed/regression_model_metrics.csv` (R², MAE, RMSE per model) together with coefficients / feature importances and a residual table for Model E.

## Scenario predictions (what-if)

Two small, explainable models predict hourly consumption from five interactive inputs — **canton, season, day type, hour, temperature bucket**:

- `scenario_predictions.csv` — Linear Regression, used in the Power BI what-if page (produces realistic, interpretable hourly profiles).
- `scenario_predictions_rf.csv` — Random Forest variant, slightly higher global accuracy but over-smoothed hourly profiles. Kept for comparison only; not wired into Power BI.

## Power BI dashboard

The end product is an interactive dashboard in [`powerbi/Energy_Weather_Dashboard.pbix`](powerbi/Energy_Weather_Dashboard.pbix). It reads the long-format CSVs in `data_processed/` and covers:

- Overview of consumption and weather patterns across cantons.
- Correlation explorer (per canton, season, time-of-day, etc.).
- Regression results and feature-importance views.
- An interactive what-if page where the user selects a scenario and sees the predicted daily consumption profile.

To refresh the dashboard after re-running the pipeline, open the `.pbix` file and click *Home → Refresh*.

## Reproducibility notes

- All CSV outputs use the Swiss/European convention: `;` as column separator and `,` as decimal separator — ready for Excel and Power BI in a Swiss locale.
- Random seeds are fixed (`random_state=42`) wherever train/test splits or random-forest models are used, so numerical results are stable across runs.
- Before a full production run, it is recommended to back up `data_processed/` (e.g. `cp -r data_processed data_processed_backup_YYYY-MM-DD`) so the previous Power BI inputs can be restored if needed.
- The `archive/` folder holds earlier scratch code and the previous German README; it is not part of the active pipeline.

## License & attribution

Data used in this project is published under the respective terms of Swissgrid and MeteoSwiss. Please cite the original providers when re-using derived datasets or visuals.
