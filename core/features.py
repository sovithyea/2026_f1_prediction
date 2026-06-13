"""
features.py
-----------
Feature engineering for the 2026 F1 prediction project.

All time-based features are normalized to GAP FROM FASTEST in that session
(fastest driver = 0.0, everyone else = positive seconds behind).
This makes sessions comparable across circuits and regulation eras.

FP1 / FP2 / FP3 are separate features — each captures a different dimension:
  FP1  →  first-run baseline, useful under new regs for early pecking order
  FP2  →  race pace on used tyres / long-run simulation
  FP3  →  setup confirmation, closest pre-qualifying signal
  Qualifying  →  single-lap raw speed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .config import (
    DRIVERS_DF,
    TEAM_PERFORMANCE_2026,
    DRIVER_MODIFIER,
    CODE_MAP_2025_TO_2026,
    FEATURE_COLS,
)

# Approximate offsets for imputing missing FP times from qualifying
# (how much slower each session typically is vs qualifying)
_FP_OFFSETS = {"FP1Time": 0.012, "FP2Time": 0.018, "FP3Time": 0.006}


def build_features(
    qualifying_df: pd.DataFrame,
    race_config: dict,
    fp1_df: Optional[pd.DataFrame] = None,   # DriverCode, FPTime
    fp2_df: Optional[pd.DataFrame] = None,
    fp3_df: Optional[pd.DataFrame] = None,
    season_perf_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge all session data into one model-ready DataFrame.

    All time columns are normalized to gap-from-fastest (fastest = 0.0).
    Missing FP sessions are imputed from qualifying time × (1 + offset).

    Parameters
    ----------
    qualifying_df  : DriverCode, QualifyingTime (raw seconds)
    race_config    : dict — RainProbability, Temperature
    fp1_df         : DriverCode, FPTime (raw seconds) — optional
    fp2_df         : DriverCode, FPTime (raw seconds) — optional
    fp3_df         : DriverCode, FPTime (raw seconds) — optional
    season_perf_df : DriverCode, AverageSeasonPerformance — optional
    """
    df = DRIVERS_DF[["DriverCode", "FullName", "Team"]].copy()

    # ── Qualifying ────────────────────────────────────────────────────────────
    df = df.merge(qualifying_df[["DriverCode", "QualifyingTime"]],
                  on="DriverCode", how="left")

    # ── FP sessions ───────────────────────────────────────────────────────────
    for fp_df, col in [(fp1_df, "FP1Time"), (fp2_df, "FP2Time"), (fp3_df, "FP3Time")]:
        if fp_df is not None and len(fp_df) > 0:
            fp_renamed = (
                fp_df[["DriverCode", "FPTime"]]
                .rename(columns={"FPTime": col})
            )
            df = df.merge(fp_renamed, on="DriverCode", how="left")
        else:
            df[col] = np.nan

    # ── Weather & team scores ─────────────────────────────────────────────────
    df["RainProbability"] = float(race_config.get("RainProbability", 0.10))
    df["Temperature"]     = float(race_config.get("Temperature", 25.0))

    df["TeamPerformanceScore"] = (
        df["Team"].map(TEAM_PERFORMANCE_2026).fillna(5.0)
        + df["DriverCode"].map(DRIVER_MODIFIER).fillna(0.0)
    )

    if season_perf_df is not None:
        df = df.merge(season_perf_df[["DriverCode", "AverageSeasonPerformance"]],
                      on="DriverCode", how="left")
        df["AverageSeasonPerformance"] = df["AverageSeasonPerformance"].fillna(
            df["TeamPerformanceScore"]
        )
    else:
        df["AverageSeasonPerformance"] = df["TeamPerformanceScore"]

    # ── Impute missing times ──────────────────────────────────────────────────
    # Qualifying: field mean + 2% for truly missing drivers
    q_mean = df["QualifyingTime"].mean()
    df["QualifyingTime"] = df["QualifyingTime"].fillna(q_mean * 1.02)

    # FP sessions: impute from qualifying × (1 + session offset)
    for col, offset in _FP_OFFSETS.items():
        df[col] = df[col].fillna(df["QualifyingTime"] * (1 + offset))

    # ── Normalize all time columns to gap from fastest ────────────────────────
    for col in ["QualifyingTime", "FP1Time", "FP2Time", "FP3Time"]:
        df[col] = df[col] - df[col].min()

    return df


def remap_2025_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Map 2025 FastF1 driver abbreviations to 2026 equivalents.
    Rows for drivers not continuing (mapped to None) are dropped."""
    df = df.copy()
    df["DriverCode"] = df["DriverCode"].map(
        lambda c: CODE_MAP_2025_TO_2026.get(c, c)
    )
    return df.dropna(subset=["DriverCode"]).reset_index(drop=True)


def synthetic_quali(race_config: dict) -> pd.DataFrame:
    """Generate synthetic qualifying times from team scores + noise.
    Used pre-FP1 when no real session data is available."""
    rng       = np.random.default_rng(seed=race_config.get("round_2026", 1) * 7)
    base_time = race_config.get("base_lap_time", 90.0)
    rows = []
    for _, row in DRIVERS_DF.iterrows():
        code      = row["DriverCode"]
        score     = TEAM_PERFORMANCE_2026.get(row["Team"], 7.0) + DRIVER_MODIFIER.get(code, 0.0)
        q_time    = base_time + (10.0 - score) * 0.35 + rng.normal(0, 0.12)
        rows.append({"DriverCode": code, "QualifyingTime": q_time})
    return pd.DataFrame(rows)