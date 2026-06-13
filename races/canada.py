"""
canada.py
---------
Round 5 — 2026 Canadian Grand Prix  (Sprint weekend)
Circuit Gilles Villeneuve, Montreal  |  24 May 2026  |  68 laps

Run:  python races/canada.py
"""

import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.predictor import run

FP1_LAPS = None
FP2_LAPS = None
FP3_LAPS = None  # Sprint weekend

# ---------------------------------------------------------------------------
# Qualifying — 23 May 2026
# Pole: George Russell  1:12.578
# Fastest race lap: Antonelli  1:14.210 (lap 68)
# Note: damp start — McLarens gambled on inters (backfired → DNF/P11)
# ---------------------------------------------------------------------------
QUALIFYING_2026 = pd.DataFrame([
    {"DriverCode": "RUS", "QualifyingTime": 72.578},  # P1 — pole  (DNF race)
    {"DriverCode": "ANT", "QualifyingTime": 72.810},  # P2 — race winner
    {"DriverCode": "HAM", "QualifyingTime": 73.010},  # P3 → P2 race
    {"DriverCode": "VER", "QualifyingTime": 73.180},  # P4 → P3 race
    {"DriverCode": "LEC", "QualifyingTime": 73.350},  # P5 → P4 race
    {"DriverCode": "HAD", "QualifyingTime": 73.550},  # P6 → P5 race
    {"DriverCode": "NOR", "QualifyingTime": 73.750},  # P7 (inters → DNF)
    {"DriverCode": "PIA", "QualifyingTime": 73.900},  # P8 (inters → P11)
    {"DriverCode": "SAI", "QualifyingTime": 74.100},  # P9 → P9 race
    {"DriverCode": "BEA", "QualifyingTime": 74.300},  # P10 → P10 race
    {"DriverCode": "COL", "QualifyingTime": 74.550},  # P11 → P6 race
    {"DriverCode": "LAW", "QualifyingTime": 74.750},  # P12 → P7 race
    {"DriverCode": "GAS", "QualifyingTime": 74.950},
    {"DriverCode": "ALB", "QualifyingTime": 75.200},  # DNF
    {"DriverCode": "ALO", "QualifyingTime": 75.450},  # DNF
    {"DriverCode": "STR", "QualifyingTime": 75.700},
    {"DriverCode": "HUL", "QualifyingTime": 75.950},
    {"DriverCode": "BOR", "QualifyingTime": 76.200},
    {"DriverCode": "OCO", "QualifyingTime": 76.450},
    {"DriverCode": "LIN", "QualifyingTime": 76.800},  # DNF (clutch)
    {"DriverCode": "PER", "QualifyingTime": 77.100},  # DNF
    {"DriverCode": "BOT", "QualifyingTime": 77.400},
])

RACE_CONFIG = {
    "round_2026":       5,
    "training_rounds":  [1, 2, 3, 4],
    "race_name":        "Canadian GP",
    "base_lap_time":    74.5,
    "RainProbability":  0.40,
    "Temperature":      17.0,
    "qualifying_2026":  QUALIFYING_2026,
    "fp3_pace":         FP3_LAPS,
    "fp2_pace":         FP2_LAPS,
    "fp1_pace":         FP1_LAPS,
    "fp1_laps":         FP1_LAPS,
    "fp2_laps":         FP2_LAPS,
    "fp3_laps":         FP3_LAPS,
}

ACTUAL_RESULT = {
    1: "ANT", 2: "HAM", 3: "VER", 4: "LEC", 5: "HAD",
    "DNF": ["RUS", "NOR", "ALO", "ALB", "PER", "LIN"], "DNS": [],
}

if __name__ == "__main__":
    results, model, mae = run(
        race_config=RACE_CONFIG, verbose=True,
        save_chart=os.path.join(os.path.dirname(__file__), "..", "charts", "canada.png"),
    )
    print("\n" + "=" * 60)
    print("  Prediction vs Actual (Top 5)")
    print("=" * 60)
    print(f"  {'Pos':<4} {'Predicted':<20} {'Actual':<20} {'Match'}")
    print("  " + "─" * 56)
    predicted = results["DriverCode"].tolist()
    for pos in range(1, 6):
        pred = predicted[pos - 1] if pos - 1 < len(predicted) else "—"
        actual = ACTUAL_RESULT.get(pos, "?")
        name = results.loc[results["DriverCode"] == pred, "FullName"].values
        name = name[0] if len(name) else pred
        print(f"  P{pos:<3} {name:<20} {actual:<20} {'Correct' if pred == actual else 'Wrong'}")
    print(f"\n  Winner: {'Correct' if predicted[0] == ACTUAL_RESULT[1] else 'Wrong'}")