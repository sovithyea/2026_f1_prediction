"""
data.py
-------
FastF1 data fetchers for the 2026 F1 prediction project.
All functions return plain pandas DataFrames with consistent column names.

Install FastF1 with:  pip install fastf1
Docs:  https://docs.fastf1.dev
"""

import os
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# Cache setup — enable once on import

try:
    import fastf1
    _CACHE_DIR = os.path.join(os.path.dirname(__file__), "fastf1_cache")
    os.makedirs(_CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(_CACHE_DIR)
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    print("[data.py] FastF1 not found — run: pip install fastf1")


def _check_fastf1():
    if not FASTF1_AVAILABLE:
        raise RuntimeError("FastF1 required. Run: pip install fastf1")


# Public fetchers

def get_qualifying(year: int, gp: int | str) -> pd.DataFrame:
    """
    Best qualifying lap per driver.

    Parameters
    ----------
    year : int        e.g. 2025
    gp   : int|str    round number or circuit name, e.g. 1 or 'Australia'

    Returns
    -------
    DataFrame — columns: DriverCode (str), QualifyingTime (float, seconds)
    """
    _check_fastf1()
    session = fastf1.get_session(year, gp, "Q")
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    best = (
        session.laps
        .pick_quicklaps()
        .groupby("Driver")["LapTime"]
        .min()
        .dt.total_seconds()
        .reset_index()
        .rename(columns={"Driver": "DriverCode", "LapTime": "QualifyingTime"})
    )
    return best


def get_race_laps(year: int, gp: int | str) -> pd.DataFrame:
    """
    Mean quick-lap race time per driver (used as GBR training target).

    Returns
    -------
    DataFrame — columns: DriverCode (str), RaceTime (float, seconds)
    """
    _check_fastf1()
    session = fastf1.get_session(year, gp, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    avg = (
        session.laps
        .pick_quicklaps()
        .groupby("Driver")["LapTime"]
        .mean()
        .dt.total_seconds()
        .reset_index()
        .rename(columns={"Driver": "DriverCode", "LapTime": "RaceTime"})
    )
    return avg


def get_sector_times(year: int, gp: int | str) -> pd.DataFrame:
    """
    Best combined sector time per driver from qualifying.

    Returns
    -------
    DataFrame — columns: DriverCode (str), TotalSectorTime (s) (float)
    """
    _check_fastf1()
    session = fastf1.get_session(year, gp, "Q")
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = (
        session.laps
        .pick_quicklaps()[["Driver", "Sector1Time", "Sector2Time", "Sector3Time"]]
        .copy()
    )
    for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[col] = laps[col].dt.total_seconds()

    laps["TotalSectorTime (s)"] = (
        laps["Sector1Time"] + laps["Sector2Time"] + laps["Sector3Time"]
    )

    best = (
        laps.groupby("Driver")["TotalSectorTime (s)"]
        .min()
        .reset_index()
        .rename(columns={"Driver": "DriverCode"})
    )
    return best


def get_weather(year: int, gp: int | str, session_type: str = "Q") -> dict:
    """
    Average air temperature and rain flag for a session.

    Returns
    -------
    dict — keys: Temperature (float, °C), RainProbability (float, 0–1)
    """
    _check_fastf1()
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=False, telemetry=False, weather=True, messages=False)

    w = session.weather_data
    temp = float(w["AirTemp"].mean()) if "AirTemp" in w.columns else 25.0
    rain = bool(w["Rainfall"].any())  if "Rainfall" in w.columns else False

    return {
        "Temperature":     round(temp, 1),
        "RainProbability": 0.75 if rain else 0.10,
    }


def get_all_training_data(year: int, gp: int | str) -> dict:
    """
    Convenience wrapper — fetches qualifying, sector times, race laps,
    and weather in one call.

    Returns
    -------
    dict with keys: qualifying, sectors, race, weather
    """
    return {
        "qualifying": get_qualifying(year, gp),
        "sectors":    get_sector_times(year, gp),
        "race":       get_race_laps(year, gp),
        "weather":    get_weather(year, gp),
    }