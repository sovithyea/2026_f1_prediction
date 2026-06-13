"""
predictor.py
------------
Pipeline: fetch 2026 completed rounds → train GBR → predict next race.

Target variable : Race Position (1–22)
Training data   : All completed 2026 rounds in training_rounds
Features        : QualifyingTime, FP1Time, FP2Time, FP3Time,
                  TeamPerformanceScore, RainProbability, Temperature,
                  AverageSeasonPerformance

Speed data priority for prediction (uses best available):
  qualifying_2026 > fp3_pace > fp2_pace > fp1_pace > synthetic
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
    round_2026        int
    training_rounds   list[int]   completed 2026 rounds to train on
    race_name         str
    base_lap_time     float
    RainProbability   float
    Temperature       float

    race_config speed data (uses best available, all optional)
    -------------------------
    qualifying_2026   pd.DataFrame  DriverCode, QualifyingTime
    fp3_pace          pd.DataFrame  DriverCode, QualifyingTime
    fp2_pace          pd.DataFrame  DriverCode, QualifyingTime
    fp1_pace          pd.DataFrame  DriverCode, QualifyingTime

    FP data for feature columns (optional — imputed from quali if absent)
    -------------------------
    fp1_laps          pd.DataFrame  DriverCode, FPTime
    fp2_laps          pd.DataFrame  DriverCode, FPTime
    fp3_laps          pd.DataFrame  DriverCode, FPTime
    """
    race_name       = race_config["race_name"]
    training_rounds = race_config.get("training_rounds", [])

    if verbose:
        print(f"\n{'='*60}")
        print(f"  🏁  2026 {race_name} — ML Prediction")
        print(f"{'='*60}")
        print(f"  Training rounds : {training_rounds or 'none (team score ranking)'}")
        print(f"  Rain : {race_config['RainProbability']:.0%}  |  "
              f"Temp : {race_config['Temperature']}°C")

    # ── 1. Best qualifying proxy ──────────────────────────────────────────────
    qual_source, qual_2026 = _best_speed_data(race_config)
    if verbose:
        print(f"  Speed source    : {qual_source}")

    # ── 2. FP session data — manual override or auto-fetch from FastF1 ────────
    fp1 = race_config.get("fp1_laps")
    fp2 = race_config.get("fp2_laps")
    fp3 = race_config.get("fp3_laps")

    if any(x is None for x in [fp1, fp2, fp3]) and d.FASTF1_AVAILABLE:
        rnd = race_config.get("round_2026")
        if rnd:
            if verbose:
                print(f"  Auto-fetching FP data for round {rnd}...")
            def _safe_fp(session):
                try:
                    return d.get_fp_laps(2026, rnd, session)
                except Exception:
                    return None
            fp1 = fp1 or _safe_fp("FP1")
            fp2 = fp2 or _safe_fp("FP2")
            fp3 = fp3 or _safe_fp("FP3")
            fetched = [s for s, v in [("FP1", fp1), ("FP2", fp2), ("FP3", fp3)] if v is not None]
            if verbose and fetched:
                print(f"  FP sessions fetched : {', '.join(fetched)}")

    # ── 3. Build prediction features ─────────────────────────────────────────
    if verbose:
        print("\n[1/3] Building prediction features...")

    pred_df = f.build_features(
        qualifying_df  = qual_2026,
        race_config    = race_config,
        fp1_df         = fp1,
        fp2_df         = fp2,
        fp3_df         = fp3,
        season_perf_df = season_perf_df,
    )
    pred_df = pred_df.dropna(subset=FEATURE_COLS)

    # ── 3. Train on completed 2026 rounds ─────────────────────────────────────
    trained_model, mae = None, None

    if training_rounds:
        if verbose:
            print(f"[2/3] Loading 2026 training data (rounds {training_rounds})...")
        train_df = _build_training_df(training_rounds, race_config, season_perf_df, verbose)

        if train_df is not None and len(train_df) >= 10:
            if verbose:
                print(f"       {len(train_df)} rows  |  features: {FEATURE_COLS}")
                print("[3/3] Training Gradient Boosting Regressor...")
            trained_model, mae = m.train(
                train_df[FEATURE_COLS], train_df["Position"], verbose=verbose
            )
            pred_df["PredictedPosition"] = trained_model.predict(pred_df[FEATURE_COLS])
        else:
            if verbose:
                print("       Not enough data — falling back to team score ranking")
            pred_df = _rank_by_team_score(pred_df)
    else:
        if verbose:
            print("[2/3] No training rounds — ranking by qualifying + team score")
        pred_df = _rank_by_team_score(pred_df)

    # ── 4. Final ranking & display ────────────────────────────────────────────
    pred_df  = pred_df.sort_values("PredictedPosition").reset_index(drop=True)
    pred_df["Position"] = range(1, len(pred_df) + 1)

    base = race_config.get("base_lap_time", 90.0)
    pred_df["PredictedRaceTime (s)"] = base + (pred_df["Position"] - 1) * 0.4

    mae_str = f"{mae:.2f}s" if mae is not None else "N/A (no training data)"
    if verbose:
        v.print_results(pred_df, race_name, mae_str)

    v.plot_results(pred_df, race_name, mae_str, save_path=save_chart)
    _save_prediction(pred_df, race_config, qual_source, mae)

    return pred_df, trained_model, mae if mae is not None else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _best_speed_data(race_config: dict) -> tuple[str, pd.DataFrame]:
    """Return (label, DataFrame) for the best available qualifying proxy."""
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
    """
    Fetch 2026 qualifying + FP1/FP2/FP3 + race positions for each training round.
    Builds normalized features and returns one combined DataFrame.
    """
    frames = []
    for rnd in rounds:
        raw = d.get_round_data(2026, rnd)
        if raw is None:
            continue

        feat_df = f.build_features(
            qualifying_df  = raw["qualifying"],
            race_config    = race_config,
            fp1_df         = raw.get("fp1"),
            fp2_df         = raw.get("fp2"),
            fp3_df         = raw.get("fp3"),
            season_perf_df = season_perf_df,
        )
        merged = feat_df.merge(
            raw["positions"][["DriverCode", "Position"]], on="DriverCode", how="inner"
        ).dropna(subset=FEATURE_COLS + ["Position"])

        if len(merged) > 0:
            frames.append(merged)
            if verbose:
                fp_available = [s for s in ["fp1", "fp2", "fp3"]
                                if raw.get(s) is not None and len(raw[s]) > 0]
                print(f"       Round {rnd}: {len(merged)} drivers  "
                      f"(sessions: Q + {', '.join(fp_available) or 'none'})")

    return pd.concat(frames, ignore_index=True) if frames else None


def _rank_by_team_score(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: rank by qualifying gap adjusted by team score."""
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
    """Auto-save full prediction to results/<race_slug>_prediction.csv."""
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