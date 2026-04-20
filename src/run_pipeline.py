"""
run_pipeline.py
===============
End-to-end pipeline orchestrator.

Runs every build/analyse script in its documented dependency order. Each
script is executed as an isolated subprocess so failures are contained, and
the pipeline stops immediately on the first non-zero exit code — no broken
upstream step can silently corrupt downstream outputs.

Usage
-----
    python run_pipeline.py                 Run the full pipeline (11 steps)
    python run_pipeline.py --list          Show the planned steps and exit
    python run_pipeline.py --dry-run       Show what WOULD run (and check that
                                           each script file actually exists),
                                           without executing anything.
    python run_pipeline.py --start 7       Resume from step 7 (skips 1-6).
                                           Useful after a failure: fix the
                                           broken script and restart from
                                           exactly where it died.
    python run_pipeline.py --start 7 --stop 9
                                           Run only steps 7 through 9.

Notes
-----
- Script output (stdout/stderr) is streamed live to the console.
- On failure, the orchestrator prints a clear banner with the failing step,
  its exit code, and the command needed to resume.
- The working directory for each subprocess is the directory of this file —
  this matches the relative paths (../data_processed/...) used inside the
  individual scripts.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Directory that holds all pipeline scripts (including this orchestrator).
SCRIPTS_DIR = Path(__file__).resolve().parent


# --- Pipeline definition -----------------------------------------------------
# Ordered list of (display name, script filename). Comments document the data
# dependency so anyone reading this file immediately sees why the order matters.

PIPELINE: list[tuple[str, str]] = [
    # Raw sources -> processed hourly CSVs
    ("Build electricity dataset",        "build_electricity_dataset.py"),
    ("Build weather dataset",            "build_weather_dataset.py"),
    ("Build location dimension",         "build_dim_location.py"),

    # Consolidation and feature engineering
    ("Build analytical base",            "build_analytical_base.py"),           # needs: electricity + weather
    ("Build time dimension",             "build_dim_time.py"),                  # needs: analytical_base
    ("Build feature dataset",            "build_feature_dataset.py"),           # needs: analytical_base

    # Analyses
    ("Correlation analysis",             "analyse_correlations.py"),            # needs: feature_dataset + dim_time
    ("Regression (Linear A/B/C)",        "analyse_regression.py"),              # needs: feature_dataset
    ("Regression (Random Forest D/E)",   "analyse_regression_extended.py"),     # needs: outputs of previous step

    # Scenario tables for Power BI
    ("Scenario predictions (Linear)",    "build_scenario_predictions_LR.py"),   # needs: feature_dataset
    ("Scenario predictions (Random F.)", "build_scenario_predictions_RF.py"),   # needs: feature_dataset
]


# --- Helpers -----------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as '42.3s' or '3m 12.7s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.1f}s"


def run_step(step_num: int, total: int, name: str, script_filename: str) -> float:
    """
    Execute one pipeline step as a subprocess.

    Returns the duration in seconds on success. On non-zero exit code, prints
    a clear failure banner and terminates the orchestrator with the same exit
    code — so the caller (shell, CI, scheduler) sees the pipeline failed.
    """
    script_path = SCRIPTS_DIR / script_filename
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 80)
    print(f"[{step_num}/{total}] {name}   ({script_filename})")
    print("=" * 80)

    start = time.time()

    # stdout/stderr inherit the parent's streams -> output appears live.
    # This matters for long-running steps (RF training) where silent waits
    # would be confusing.
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SCRIPTS_DIR,
    )
    duration = time.time() - start

    if result.returncode != 0:
        print("\n" + "!" * 80)
        print(f"STEP FAILED:  {name}")
        print(f"Script:       {script_filename}")
        print(f"Exit code:    {result.returncode}")
        print(f"Duration:     {fmt_duration(duration)}")
        print()
        print(f"Pipeline stopped. After fixing the error, resume with:")
        print(f"    python {Path(__file__).name} --start {step_num}")
        print("!" * 80)
        sys.exit(result.returncode)

    print(f"\n--> OK in {fmt_duration(duration)}")
    return duration


# --- Main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full electricity/weather data pipeline."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all pipeline steps and exit without running anything.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned commands and verify each script file exists, "
             "without executing anything. Safe to run at any time.",
    )
    parser.add_argument(
        "--start", type=int, default=1, metavar="N",
        help="Start at step N (1-based). Useful for resuming after a failure.",
    )
    parser.add_argument(
        "--stop", type=int, default=None, metavar="N",
        help="Stop after step N (inclusive). Defaults to the last step.",
    )
    args = parser.parse_args()

    total = len(PIPELINE)

    # --list: just print the plan and exit, do nothing else.
    if args.list:
        print("Pipeline steps:")
        for i, (name, script) in enumerate(PIPELINE, start=1):
            print(f"  {i:2d}. {name:36s}  {script}")
        return

    # Validate --start / --stop bounds.
    if not (1 <= args.start <= total):
        parser.error(f"--start must be between 1 and {total}")
    stop = args.stop if args.stop is not None else total
    if not (args.start <= stop <= total):
        parser.error(f"--stop must be between --start ({args.start}) and {total}")

    # --dry-run: show what would run and verify that each script file exists,
    # but do NOT start any subprocess. Completely safe for data.
    if args.dry_run:
        print("DRY RUN — no scripts will be executed.")
        print(f"Scripts dir:  {SCRIPTS_DIR}")
        print(f"Python:       {sys.executable}")
        print(f"Steps:        {args.start} -> {stop} of {total}\n")

        missing: list[str] = []
        for i, (name, script) in enumerate(PIPELINE, start=1):
            if i < args.start or i > stop:
                continue
            script_path = SCRIPTS_DIR / script
            status = "OK" if script_path.exists() else "!!"
            if not script_path.exists():
                missing.append(script)
            print(f"  [{status}] step {i:2d}/{total}: {name}  ({script})")

        if missing:
            print(f"\nMissing scripts: {len(missing)}")
            for m in missing:
                print(f"  - {m}")
            sys.exit(1)

        print("\nAll scripts present. Pipeline can be run.")
        return

    # --- Real run ---
    print(f"Pipeline run started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Scripts dir:          {SCRIPTS_DIR}")
    print(f"Python interpreter:   {sys.executable}")
    print(f"Steps to run:         {args.start} -> {stop} of {total}")

    overall_start = time.time()
    timings: list[tuple[int, str, float]] = []

    try:
        for i, (name, script) in enumerate(PIPELINE, start=1):
            if i < args.start or i > stop:
                continue
            d = run_step(i, total, name, script)
            timings.append((i, name, d))
    except KeyboardInterrupt:
        # Ctrl-C is propagated to the subprocess automatically; we just need
        # a clean message here. Exit code 130 is the shell convention for
        # "terminated by SIGINT".
        print("\n\nInterrupted by user. Pipeline stopped.")
        sys.exit(130)

    # --- Summary ---
    total_time = time.time() - overall_start
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total duration: {fmt_duration(total_time)}")
    print("\nPer-step timing:")
    for i, name, d in timings:
        print(f"  {i:2d}. {name:36s}  {fmt_duration(d)}")


if __name__ == "__main__":
    main()
