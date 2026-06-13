"""
predictor.py
------------
Pipeline: fetch 2026 completed rounds → train GBR → predict next race.

Target variable: Race Position (1–22).
Training data:   All completed 2026 rounds specified in training_rounds.
Prediction:      Real qualifying (or FP2/FP1 pace / synthetic) for the target race.

Auto-saves prediction CSV to results/<race_name>.csv on every run.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import data as d
from . import features as f
from . import model as m
from . import viz as v
from .config import FEATURE_COLS

# Path to results/ folder (two levels up from core/)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run(
    race_config: dict,
    season_perf_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
    save_chart: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[GradientBoostingRegressor], float]:
    """
    Full prediction pipeline for one 2026 race.

    race_config required keys
    -------------------------
    round_2026        int         target race round number
    training_rounds   list[int]   completed 2026 rounds to train on
    race_name         str
    base_lap_time     float       circuit reference lap in seconds
    RainProbability   float
    Temperature       float

    race_config optional speed data (uses best available)
    -------------------------
    qualifying_2026   pd.DataFrame   DriverCode, QualifyingTime  ← after qualifying
    fp3_pace          pd.DataFrame   DriverCode, QualifyingTime  ← after FP3
    fp2_pace          pd.DataFrame   DriverCode, QualifyingTime  ← after FP2
    fp1_pace          pd.DataFrame   DriverCode, QualifyingTime  ← after FP1
    """
    race_name       = race_config["race_name"]
    training_rounds = race_config.get("training_rounds", [])

    if verbose:
        print(f"\n{'='*60}")
        print(f"  🏁  2026 {race_name} — ML Prediction")
        print(f"{'='*60}")
        print(f"  Training rounds : {training_rounds if training_rounds else 'none (team score ranking)'}")
        print(f"  Rain     : {race_config['RainProbability']:.0%}  |  "
              f"Temp: {race_config['Temperature']}°C")

    # ── 1. Determine best available speed data ────────────────────────────────
    speed_source, qual_2026 = _best_speed_data(race_config)
    if verbose:
        print(f"  Speed source    : {speed_source}")

    # ── 2. Build prediction features ─────────────────────────────────────────
    if verbose:
        print("\n[1/3] Building prediction features...")
    sect_2026 = f.synthetic_sectors(qual_2026)
    pred_df   = f.build_features(qual_2026, sect_2026, race_config, season_perf_df)
    pred_df   = pred_df.dropna(subset=FEATURE_COLS)

    # ── 3. Train on completed 2026 rounds (if any) ────────────────────────────
    trained_model = None
    mae           = None

    if training_rounds:
        if verbose:
            print(f"[2/3] Loading 2026 training data (rounds {training_rounds})...")

        train_df = _build_training_df(training_rounds, race_config, season_perf_df, verbose)

        if train_df is not None and len(train_df) >= 10:
            if verbose:
                print(f"       {len(train_df)} training rows across {len(training_rounds)} rounds")
                print("[3/3] Training Gradient Boosting Regressor...")
            trained_model, mae = m.train(
                train_df[FEATURE_COLS], train_df["Position"], verbose=verbose
            )
            pred_df["PredictedPosition"] = trained_model.predict(pred_df[FEATURE_COLS])
        else:
            if verbose:
                print("       Not enough training data — falling back to team score ranking")
            pred_df = _rank_by_team_score(pred_df)
    else:
        if verbose:
            print("[2/3] No training rounds — ranking by qualifying + team score")
        pred_df = _rank_by_team_score(pred_df)

    # ── 4. Final ranking ──────────────────────────────────────────────────────
    pred_df = pred_df.sort_values("PredictedPosition").reset_index(drop=True)
    pred_df["Position"] = range(1, len(pred_df) + 1)

    base = race_config.get("base_lap_time", 90.0)
    pred_df["PredictedRaceTime (s)"] = base + (pred_df["Position"] - 1) * 0.4

    mae_str = f"{mae:.2f}s" if mae is not None else "N/A (no training data)"

    if verbose:
        v.print_results(pred_df, race_name, mae_str)

    v.plot_results(pred_df, race_name, mae_str, save_path=save_chart)

    # ── 5. Auto-save prediction CSV ───────────────────────────────────────────
    _save_prediction(pred_df, race_config, speed_source, mae)

    return pred_df, trained_model, mae if mae is not None else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_speed_data(race_config: dict) -> tuple[str, pd.DataFrame]:
    """Return (source_label, DataFrame) using best available speed data."""
    for key, label in [
        ("qualifying_2026", "qualifying"),
        ("fp3_pace",        "FP3 pace"),
        ("fp2_pace",        "FP2 pace"),
        ("fp1_pace",        "FP1 pace"),
    ]:
        val = race_config.get(key)
        if val is not None:
            return label, val
    return "synthetic", f.synthetic_quali(race_config)


def _build_training_df(
    rounds: list,
    race_config: dict,
    season_perf_df,
    verbose: bool,
) -> pd.DataFrame | None:
    """Fetch 2026 qualifying + positions for each training round, return combined DataFrame."""
    frames = []
    for rnd in rounds:
        raw = d.get_round_data(2026, rnd)
        if raw is None:
            continue
        feat_df = f.build_features(
            raw["qualifying"],
            f.synthetic_sectors(raw["qualifying"]),
            raw["weather"],
            season_perf_df,
        )
        merged = feat_df.merge(
            raw["positions"][["DriverCode", "Position"]], on="DriverCode", how="inner"
        ).dropna(subset=FEATURE_COLS + ["Position"])
        if len(merged) > 0:
            frames.append(merged)
            if verbose:
                print(f"       Round {rnd}: {len(merged)} drivers loaded")
    return pd.concat(frames, ignore_index=True) if frames else None


def _rank_by_team_score(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback ranking: qualifying gap (lower = better) offset by team score."""
    df = pred_df.copy()
    df["PredictedPosition"] = (
        df["QualifyingTime"] * 2.0
        - df["TeamPerformanceScore"] * 0.5
    )
    return df


def _save_prediction(
    pred_df: pd.DataFrame,
    race_config: dict,
    speed_source: str,
    mae: float | None,
) -> None:
    """Save prediction to results/<race_slug>_prediction.csv."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    race_name = race_config["race_name"]
    slug      = race_name.lower().replace(" ", "_").replace("/", "_")
    path      = os.path.join(RESULTS_DIR, f"{slug}_prediction.csv")

    out = pred_df[["Position", "DriverCode", "FullName", "Team",
                   "PredictedRaceTime (s)"]].copy()
    out.insert(0, "Race",        race_name)
    out.insert(1, "Round",       race_config.get("round_2026", "?"))
    out.insert(2, "SpeedSource", speed_source)
    out.insert(3, "MAE",         f"{mae:.2f}s" if mae is not None else "N/A")

    out.to_csv(path, index=False)
    print(f"  Prediction saved → results/{slug}_prediction.csv")