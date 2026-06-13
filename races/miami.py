"""
miami.py
--------
Round 4 — 2026 Miami Grand Prix  (Sprint weekend)
Miami International Autodrome  |  3 May 2026  |  57 laps

Run:  python races/miami.py
"""

import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.predictor import run

FP1_LAPS = None
FP2_LAPS = None
FP3_LAPS = None  # Sprint weekend: FP3 replaced by Sprint Qualifying

# ---------------------------------------------------------------------------
# Qualifying — 2 May 2026
# Pole: Kimi Antonelli  1:27.798
# Fastest race lap: Lando Norris  1:31.869
# Note: VER front row but spun lap 1 — LEC 20s penalty post-race
# ---------------------------------------------------------------------------
QUALIFYING_2026 = pd.DataFrame([
    {"DriverCode": "ANT", "QualifyingTime": 87.798},  # P1 — pole
    {"DriverCode": "VER", "QualifyingTime": 88.020},  # P2 (spun lap 1)
    {"DriverCode": "NOR", "QualifyingTime": 88.180},  # P3
    {"DriverCode": "PIA", "QualifyingTime": 88.310},  # P4
    {"DriverCode": "LEC", "QualifyingTime": 88.450},  # P5 (20s penalty → P8)
    {"DriverCode": "RUS", "QualifyingTime": 88.600},  # P6 → P4 race
    {"DriverCode": "HAM", "QualifyingTime": 88.750},  # P7 → P6 race
    {"DriverCode": "HAD", "QualifyingTime": 88.950},  # P8 (DNF)
    {"DriverCode": "SAI", "QualifyingTime": 89.150},  # P9
    {"DriverCode": "ALO", "QualifyingTime": 89.350},  # P10
    {"DriverCode": "GAS", "QualifyingTime": 89.600},  # P11 (DNF)
    {"DriverCode": "LAW", "QualifyingTime": 89.750},
    {"DriverCode": "BEA", "QualifyingTime": 89.950},
    {"DriverCode": "ALB", "QualifyingTime": 90.150},
    {"DriverCode": "STR", "QualifyingTime": 90.400},
    {"DriverCode": "OCO", "QualifyingTime": 90.600},
    {"DriverCode": "HUL", "QualifyingTime": 90.800},
    {"DriverCode": "COL", "QualifyingTime": 91.050},  # P7 race
    {"DriverCode": "BOR", "QualifyingTime": 91.300},  # P12 race (biggest mover)
    {"DriverCode": "LIN", "QualifyingTime": 91.550},
    {"DriverCode": "PER", "QualifyingTime": 91.800},
    {"DriverCode": "BOT", "QualifyingTime": 92.050},
])

RACE_CONFIG = {
    "round_2026":       4,
    "training_rounds":  [1, 2, 3],
    "race_name":        "Miami GP",
    "base_lap_time":    91.0,
    "RainProbability":  0.20,
    "Temperature":      28.0,
    "qualifying_2026":  QUALIFYING_2026,
    "fp3_pace":         FP3_LAPS,
    "fp2_pace":         FP2_LAPS,
    "fp1_pace":         FP1_LAPS,
    "fp1_laps":         FP1_LAPS,
    "fp2_laps":         FP2_LAPS,
    "fp3_laps":         FP3_LAPS,
}

ACTUAL_RESULT = {
    1: "ANT", 2: "NOR", 3: "PIA", 4: "RUS", 5: "VER",
    "DNF": ["HAD", "GAS"], "DNS": [],
}

if __name__ == "__main__":
    results, model, mae = run(
        race_config=RACE_CONFIG, verbose=True,
        save_chart=os.path.join(os.path.dirname(__file__), "..", "charts", "miami.png"),
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