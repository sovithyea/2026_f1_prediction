"""
COUNTRY.py
----------
Round X — 2026 RACE NAME
Circuit, City  |  Date 2026  |  XX laps

Run:
    python races/COUNTRY.py
"""

import os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.predictor import run

# STAGE 1 — After FP2: fill in long-run race pace per driver
#           Use sector/stint averages from FP2 telemetry (slowest meaningful stint)
#           Leave as None if not yet available
FP2_PACE = None
# FP2_PACE = pd.DataFrame([
#     {"DriverCode": "ANT", "QualifyingTime": 93.2},  # FP2 long-run avg (s)
#     {"DriverCode": "RUS", "QualifyingTime": 93.5},
#     ...
# ])


# STAGE 2 — After qualifying: swap in real times, set FP2_PACE to None
QUALIFYING_2026 = None
# QUALIFYING_2026 = pd.DataFrame([
#     {"DriverCode": "ANT", "QualifyingTime": 90.1},   # P1 — pole
#     {"DriverCode": "RUS", "QualifyingTime": 90.3},   # P2
#     ...
# ])

# Race configuration — update Temperature/Rain per weekend forecast
RACE_CONFIG = {
    "round_2026":      0,            # UPDATE: calendar round number
    "round_train":     "CIRCUIT",    # UPDATE: 2025 equivalent circuit name
    "race_name":       "RACE NAME",  # UPDATE
    "base_lap_time":   90.0,         # UPDATE: circuit reference lap (seconds)
    "RainProbability": 0.10,         # UPDATE after weather forecast
    "Temperature":     25.0,         # UPDATE after weather forecast
    "qualifying_2026": QUALIFYING_2026 or FP2_PACE,
}

# STAGE 3 — After race: fill in actual result for accuracy tracking
ACTUAL_RESULT = {
    1: "???",
    2: "???",
    3: "???",
    4: "???",
    5: "???",
    "DNF": [],
    "DNS": [],
}

# Run
if __name__ == "__main__":
    results, model, mae = run(
        race_config=RACE_CONFIG,
        training_year=2025,
        verbose=True,
        save_chart=os.path.join(os.path.dirname(__file__), "..", "charts", "COUNTRY.png"),
    )

    print("\n" + "=" * 60)
    print("   Prediction vs Actual (Top 5)")
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

    top1_correct = predicted_order[0] == ACTUAL_RESULT.get(1)
    print(f"\n  Winner prediction: {'Correct' if top1_correct else 'Wrong'}")
    print(f"  Model MAE: {mae:.2f}s")