"""
data.py
-------
FastF1 data fetchers — 2026 season only.
Fetches qualifying, race positions, sector times and weather.

Install: pip install fastf1
Docs:    https://docs.fastf1.dev
"""

import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# ── Cache setup ───────────────────────────────────────────────────────────────
try:
    import fastf1
    _CACHE = os.path.join(os.path.dirname(__file__), "..", "fastf1_cache")
    os.makedirs(_CACHE, exist_ok=True)
    fastf1.Cache.enable_cache(_CACHE)
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    print("[data.py] FastF1 not found — run: pip install fastf1")


def _check():
    if not FASTF1_AVAILABLE:
        raise RuntimeError("FastF1 required. Run: pip install fastf1")


# ── Single-session fetchers ───────────────────────────────────────────────────

def get_qualifying(year: int, gp: int | str) -> pd.DataFrame:
    """Best qualifying lap per driver.
    Returns: DriverCode (str), QualifyingTime (float, seconds)"""
    _check()
    session = fastf1.get_session(year, gp, "Q")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    best = (
        session.laps.pick_quicklaps()
        .groupby("Driver")["LapTime"]
        .min()
        .dt.total_seconds()
        .reset_index()
        .rename(columns={"Driver": "DriverCode", "LapTime": "QualifyingTime"})
    )
    return best


def get_race_positions(year: int, gp: int | str) -> pd.DataFrame:
    """Official race finishing positions from session.results.
    Returns: DriverCode (str), Position (float)"""
    _check()
    session = fastf1.get_session(year, gp, "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    results = session.results[["Abbreviation", "Position"]].copy()
    results = results.rename(columns={"Abbreviation": "DriverCode"})
    results["Position"] = pd.to_numeric(results["Position"], errors="coerce")
    return results.dropna(subset=["Position"]).reset_index(drop=True)


def get_fp_laps(year: int, gp: int | str, session_type: str = "FP2") -> pd.DataFrame:
    """Best lap per driver from a practice session (FP1 / FP2 / FP3).
    Returns: DriverCode (str), FPTime (float, seconds)"""
    _check()
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    best = (
        session.laps.pick_quicklaps()
        .groupby("Driver")["LapTime"]
        .min()
        .dt.total_seconds()
        .reset_index()
        .rename(columns={"Driver": "DriverCode", "LapTime": "FPTime"})
    )
    return best


def get_sector_times(year: int, gp: int | str) -> pd.DataFrame:
    """Best combined sector time per driver from qualifying.
    Returns: DriverCode (str), TotalSectorTime (s) (float)"""
    _check()
    session = fastf1.get_session(year, gp, "Q")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = (
        session.laps.pick_quicklaps()
        [["Driver", "Sector1Time", "Sector2Time", "Sector3Time"]]
        .copy()
    )
    for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[col] = laps[col].dt.total_seconds()
    laps["TotalSectorTime (s)"] = (
        laps["Sector1Time"] + laps["Sector2Time"] + laps["Sector3Time"]
    )
    return (
        laps.groupby("Driver")["TotalSectorTime (s)"]
        .min()
        .reset_index()
        .rename(columns={"Driver": "DriverCode"})
    )


def get_weather(year: int, gp: int | str, session_type: str = "R") -> dict:
    """Average race-day weather.
    Returns: dict with Temperature (°C) and RainProbability (0–1)"""
    _check()
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=False, telemetry=False, weather=True, messages=False)
    w    = session.weather_data
    temp = float(w["AirTemp"].mean()) if "AirTemp"  in w.columns else 25.0
    rain = bool(w["Rainfall"].any())  if "Rainfall" in w.columns else False
    return {"Temperature": round(temp, 1), "RainProbability": 0.75 if rain else 0.10}


# ── Multi-round training builder ──────────────────────────────────────────────

def get_round_data(year: int, gp: int | str) -> dict | None:
    """
    Fetch all data needed to build training features for one completed round.
    Returns dict with keys: qualifying, fp1, fp2, fp3, positions, weather
    Returns None if the round can't be loaded.
    """
    try:
        quali     = get_qualifying(year, gp)
        positions = get_race_positions(year, gp)
        weather   = get_weather(year, gp)

        # FP sessions — fail gracefully (wet/cancelled sessions return empty df)
        def safe_fp(session_type: str) -> pd.DataFrame:
            try:
                return get_fp_laps(year, gp, session_type)
            except Exception:
                return pd.DataFrame(columns=["DriverCode", "FPTime"])

        return {
            "qualifying": quali,
            "fp1":        safe_fp("FP1"),
            "fp2":        safe_fp("FP2"),
            "fp3":        safe_fp("FP3"),
            "positions":  positions,
            "weather":    weather,
        }
    except Exception as e:
        print(f"  [data] Warning: could not load {year} round {gp}: {e}")
        return None