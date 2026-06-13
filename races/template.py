"""
COUNTRY.py
----------
Round X — 2026 RACE NAME
Circuit  |  Date  |  XX laps

Run:
    python races/COUNTRY.py
"""

import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.predictor import run

# ---------------------------------------------------------------------------
# STAGE 1 — After FP1
# Best lap per driver from FP1 (used as FPTime feature)
# ---------------------------------------------------------------------------
FP1_LAPS = None
# FP1_LAPS = pd.DataFrame([
#     {"DriverCode": "ANT", "FPTime": 89.5},
#     {"DriverCode": "RUS", "FPTime": 89.7},
#     ...
# ])

# ---------------------------------------------------------------------------
# STAGE 2 — After FP2
# Long-run average OR best lap — FP2 is the race pace session
# ---------------------------------------------------------------------------
FP2_LAPS = None
# FP2_LAPS = pd.DataFrame([
#     {"DriverCode": "ANT", "FPTime": 91.2},   # long-run avg (s)
#     {"DriverCode": "RUS", "FPTime": 91.6},
#     ...
# ])

# ---------------------------------------------------------------------------
# STAGE 3 — After FP3
# Best lap per driver from FP3 (setup confirmation)
# ---------------------------------------------------------------------------
FP3_LAPS = None
# FP3_LAPS = pd.DataFrame([
#     {"DriverCode": "ANT", "FPTime": 88.9},
#     {"DriverCode": "RUS", "FPTime": 89.1},
#     ...
# ])

# ---------------------------------------------------------------------------
# STAGE 4 — After qualifying
# Real qualifying times — most important input
# ---------------------------------------------------------------------------
QUALIFYING_2026 = None
# QUALIFYING_2026 = pd.DataFrame([
#     {"DriverCode": "ANT", "QualifyingTime": 87.5},   # P1 — pole
#     {"DriverCode": "RUS", "QualifyingTime": 87.7},   # P2
#     ...
# ])

# ---------------------------------------------------------------------------
# Race config — update Temperature/Rain from weekend forecast
# ---------------------------------------------------------------------------
RACE_CONFIG = {
    "round_2026":       0,            # UPDATE
    "training_rounds":  [],           # UPDATE: e.g. [1,2,3,4,5,6]
    "race_name":        "RACE NAME",  # UPDATE
    "base_lap_time":    90.0,         # UPDATE: circuit reference lap (s)
    "RainProbability":  0.10,         # UPDATE from forecast
    "Temperature":      25.0,         # UPDATE from forecast
    # Speed source for quali proxy (best available used automatically)
    "qualifying_2026":  QUALIFYING_2026,
    "fp3_pace":         FP3_LAPS,
    "fp2_pace":         FP2_LAPS,
    "fp1_pace":         FP1_LAPS,
    # FP features (separate from quali proxy — all four are model features)
    "fp1_laps":         FP1_LAPS,
    "fp2_laps":         FP2_LAPS,
    "fp3_laps":         FP3_LAPS,
}

# ---------------------------------------------------------------------------
# STAGE 5 — After race: fill in for accuracy tracking
# ---------------------------------------------------------------------------
ACTUAL_RESULT = {
    1: "???", 2: "???", 3: "???", 4: "???", 5: "???",
    "DNF": [], "DNS": [],
}

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results, model, mae = run(
        race_config=RACE_CONFIG,
        verbose=True,
        save_chart=os.path.join(os.path.dirname(__file__), "..", "charts", "COUNTRY.png"),
    )

    print("\n" + "=" * 60)
    print("  Prediction vs Actual (Top 5)")
    print("=" * 60)
    print(f"  {'Pos':<4} {'Predicted':<20} {'Actual':<20} {'Match'}")
    print("  " + "─" * 56)

    predicted_order = results["DriverCode"].tolist()
    for pos in range(1, 6):
        pred_code   = predicted_order[pos - 1] if pos - 1 < len(predicted_order) else "—"
        actual_code = ACTUAL_RESULT.get(pos, "?")
        match       = "Correct" if pred_code == actual_code else "Wrong"
        pred_name   = results.loc[results["DriverCode"] == pred_code, "FullName"].values
        pred_name   = pred_name[0] if len(pred_name) else pred_code
        print(f"  P{pos:<3} {pred_name:<20} {actual_code:<20} {match}")

    top1 = predicted_order[0] == ACTUAL_RESULT.get(1)
    print(f"\n  Winner prediction: {'Correct' if top1 else 'Wrong'}")