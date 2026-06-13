"""
predictor.py
------------
Thin pipeline that wires data → features → model → viz together.
Every race script calls run() with a race_config dict.

Example
-------
from predictor import run

result, model, mae = run({
    "round_2026":      1,
    "round_train":     1,          # equivalent 2025 round for training
    "race_name":       "Australian GP",
    "base_lap_time":   88.0,       # circuit reference lap in seconds
    "RainProbability": 0.10,
    "Temperature":     22.0,
    # Optional — provide after real 2026 qualifying:
    # "qualifying_2026": pd.DataFrame({"DriverCode": [...], "QualifyingTime": [...]})
})
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import core.data as d
import core.features as f
import core.model as m
import core.viz as v
from core.config import FEATURE_COLS


def run(
    race_config: dict,
    training_year: int = 2025,
    season_perf_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
    save_chart: Optional[str] = None,
) -> tuple[pd.DataFrame, GradientBoostingRegressor, float]:
    """
    Full prediction pipeline for one 2026 race.

    race_config required keys
    -------------------------
    round_2026       int       2026 calendar round number
    round_train      int|str   2025 round used for training (number or name)
    race_name        str       e.g. "Australian GP"
    base_lap_time    float     circuit reference lap in seconds
    RainProbability  float     0.0 – 1.0
    Temperature      float     degrees Celsius

    race_config optional keys
    -------------------------
    qualifying_2026  pd.DataFrame   real 2026 qualifying (DriverCode, QualifyingTime)
                                    once available; synthetic used otherwise

    Parameters
    ----------
    training_year  : year of historical data to train on (default 2025)
    season_perf_df : DriverCode + AverageSeasonPerformance — mid-season use
    verbose        : print step-by-step progress and final table
    save_chart     : file path to save PNG chart; None = display inline

    Returns
    -------
    (results_df, trained_model, mae)
    results_df is sorted by predicted finishing order (P1 first)
    """
    race_name   = race_config["race_name"]
    round_train = race_config["round_train"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  🏁  2026 {race_name} — ML Prediction")
        print(f"{'='*60}")
        print(f"  Training : {training_year} round {round_train}")
        print(f"  Rain     : {race_config['RainProbability']:.0%}  |  "
              f"Temp: {race_config['Temperature']}°C")

    # 1. Fetch 2025 training data 
    if verbose:
        print("\n[1/4] Fetching training data...")

    raw = d.get_all_training_data(training_year, round_train)

    # Remap 2025 driver codes → 2026
    for key in ("qualifying", "sectors", "race"):
        raw[key] = f.remap_2025_codes(raw[key])

    train_df = f.build_features(
        raw["qualifying"], raw["sectors"], raw["weather"], season_perf_df
    )
    train_df = (
        train_df
        .merge(raw["race"][["DriverCode", "RaceTime"]], on="DriverCode", how="inner")
        .dropna(subset=FEATURE_COLS + ["RaceTime"])
    )

    # 2. Train model 
    if verbose:
        print("[2/4] Training Gradient Boosting Regressor...")

    trained_model, mae = m.train(
        train_df[FEATURE_COLS], train_df["RaceTime"], verbose=verbose
    )

    # 3. Build 2026 prediction features 
    if verbose:
        print("[3/4] Building 2026 feature set...")

    qual_2026 = race_config.get("qualifying_2026") or f.synthetic_quali(race_config)
    sect_2026 = f.synthetic_sectors(qual_2026)

    pred_df = f.build_features(qual_2026, sect_2026, race_config, season_perf_df)
    pred_df = pred_df.dropna(subset=FEATURE_COLS)

    # 4. Predict, rank, display
    if verbose:
        print("[4/4] Predicting...")

    results = m.predict(trained_model, pred_df)

    if verbose:
        v.print_results(results, race_name, mae)

    v.plot_results(results, race_name, mae, save_path=save_chart)

    return results, trained_model, mae