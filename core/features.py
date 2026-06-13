"""
features.py
-----------
Feature engineering for the 2026 F1 prediction project.

build_features()   — merges all data sources into one model-ready DataFrame
synthetic_quali()  — generates qualifying times from team scores when real
                     2026 data isn't available yet (future races)
remap_2025_codes() — maps 2025 FastF1 driver abbreviations to 2026 equivalents
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from config import (
    DRIVERS_DF,
    TEAM_PERFORMANCE_2026,
    DRIVER_MODIFIER,
    CODE_MAP_2025_TO_2026,
    FEATURE_COLS,
)


def build_features(
    qualifying_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    race_config: dict,
    season_perf_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge all feature sources into one model-ready DataFrame.

    Parameters
    ----------
    qualifying_df  : DriverCode, QualifyingTime (s)
    sector_df      : DriverCode, TotalSectorTime (s)
    race_config    : dict with at least RainProbability and Temperature
    season_perf_df : DriverCode, AverageSeasonPerformance
                     Pass once races have been run to improve mid-season accuracy.
                     If None, team score is used as the season performance proxy.

    Returns
    -------
    DataFrame with all FEATURE_COLS plus DriverCode, FullName, Team
    """
    df = DRIVERS_DF[["DriverCode", "FullName", "Team"]].copy()

    # Lap & sector times
    df = df.merge(qualifying_df[["DriverCode", "QualifyingTime"]],
                  on="DriverCode", how="left")
    df = df.merge(sector_df[["DriverCode", "TotalSectorTime (s)"]],
                  on="DriverCode", how="left")

    # Weather (constant across all drivers for this race)
    df["RainProbability"] = float(race_config.get("RainProbability", 0.10))
    df["Temperature"]     = float(race_config.get("Temperature", 25.0))

    # Team score + driver modifier
    df["TeamPerformanceScore"] = (
        df["Team"].map(TEAM_PERFORMANCE_2026).fillna(5.0)
        + df["DriverCode"].map(DRIVER_MODIFIER).fillna(0.0)
    )

    # Rolling season performance
    if season_perf_df is not None:
        df = df.merge(season_perf_df[["DriverCode", "AverageSeasonPerformance"]],
                      on="DriverCode", how="left")
        df["AverageSeasonPerformance"] = df["AverageSeasonPerformance"].fillna(
            df["TeamPerformanceScore"]
        )
    else:
        df["AverageSeasonPerformance"] = df["TeamPerformanceScore"]

    # Impute missing lap/sector times: field mean + 2% slowdown penalty
    for col in ["QualifyingTime", "TotalSectorTime (s)"]:
        col_mean = df[col].mean()
        df[col] = df[col].fillna(col_mean * 1.02)

    return df


def remap_2025_codes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 2025 FastF1 driver abbreviations to their 2026 equivalents.
    Rows for drivers not continuing in 2026 are dropped (mapped to None).

    Parameters
    ----------
    df : any DataFrame with a 'DriverCode' column

    Returns
    -------
    DataFrame with updated DriverCode column, None-mapped rows removed
    """
    df = df.copy()
    df["DriverCode"] = df["DriverCode"].map(
        lambda c: CODE_MAP_2025_TO_2026.get(c, c)
    )
    return df.dropna(subset=["DriverCode"]).reset_index(drop=True)


def synthetic_quali(race_config: dict) -> pd.DataFrame:
    """
    Generate synthetic qualifying times from team scores + Gaussian noise.
    Used for future races where 2026 qualifying hasn't happened yet.

    Parameters
    ----------
    race_config : dict — must contain round_2026 and base_lap_time

    Returns
    -------
    DataFrame — columns: DriverCode, QualifyingTime (s)
    """
    rng       = np.random.default_rng(seed=race_config.get("round_2026", 1) * 7)
    base_time = race_config.get("base_lap_time", 90.0)

    rows = []
    for _, row in DRIVERS_DF.iterrows():
        code       = row["DriverCode"]
        team_score = TEAM_PERFORMANCE_2026.get(row["Team"], 7.0)
        modifier   = DRIVER_MODIFIER.get(code, 0.0)
        # Higher combined score -> smaller offset from base time -> faster lap
        pace_offset = (10.0 - (team_score + modifier)) * 0.35
        q_time = base_time + pace_offset + rng.normal(0, 0.12)
        rows.append({"DriverCode": code, "QualifyingTime": q_time})

    return pd.DataFrame(rows)


def synthetic_sectors(qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate sector total as 97% of the qualifying lap.
    Used whenever real sector data is unavailable.
    """
    df = qualifying_df[["DriverCode", "QualifyingTime"]].copy()
    df["TotalSectorTime (s)"] = df["QualifyingTime"] * 0.97
    return df[["DriverCode", "TotalSectorTime (s)"]]