"""
core/__init__.py
Public surface of the core package.
Race scripts only need: from core.predictor import run
"""
from .predictor import run
from .config import CALENDAR_DF, DRIVERS_DF, TEAM_COLORS

__all__ = ["run", "CALENDAR_DF", "DRIVERS_DF", "TEAM_COLORS"]