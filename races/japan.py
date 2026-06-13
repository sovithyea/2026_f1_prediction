"""
japan.py
--------
Round 3 — 2026 Japanese Grand Prix
Suzuka International Racing Course  |  29 March 2026  |  53 laps

Run:  python races/japan.py
"""

import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.predictor import run

# ---------------------------------------------------------------------------
# FP1 — 27 March 2026  (None = imputed from qualifying)
# ---------------------------------------------------------------------------
FP1_LAPS = None

# ---------------------------------------------------------------------------
# FP2 — 27 March 2026  (None = imputed from qualifying)
# ---------------------------------------------------------------------------
FP2_LAPS = None

# ---------------------------------------------------------------------------
# FP3 — 28 March 2026  (None = imputed from qualifying)
# ---------------------------------------------------------------------------
FP3_LAPS = None

# ---------------------------------------------------------------------------
# Qualifying — 28 March 2026
# Pole: Kimi Antonelli  1:28.778
# Fastest race lap: Antonelli  1:32.432
# ---------------------------------------------------------------------------
QUALIFYING_2026 = pd.DataFrame([
    {"DriverCode": "ANT", "QualifyingTime": 88.778},  # P1 — pole
    {"DriverCode": "RUS", "QualifyingTime": 89.020},  # P2 (est.)
    {"DriverCode": "PIA", "QualifyingTime": 89.180},  # P3 (est.)
    {"DriverCode": "LEC", "QualifyingTime": 89.310},  # P4 (est.)
    {"DriverCode": "NOR", "QualifyingTime": 89.450},  # P5 (est.)
    {"DriverCode": "HAM", "QualifyingTime": 89.580},  # P6 (est.)
    {"DriverCode": "VER", "QualifyingTime": 89.720},  # P7 (est.)
    {"DriverCode": "HAD", "QualifyingTime": 89.900},  # P8 (est.)
    {"DriverCode": "SAI", "QualifyingTime": 90.100},  # P9 (est.)
    {"DriverCode": "ALO", "QualifyingTime": 90.280},  # P10 (est.)
    {"DriverCode": "ALB", "QualifyingTime": 90.500},
    {"DriverCode": "GAS", "QualifyingTime": 90.650},
    {"DriverCode": "LAW", "QualifyingTime": 90.800},
    {"DriverCode": "BEA", "QualifyingTime": 90.980},
    {"DriverCode": "STR", "QualifyingTime": 91.150},
    {"DriverCode": "OCO", "QualifyingTime": 91.350},
    {"DriverCode": "HUL", "QualifyingTime": 91.550},
    {"DriverCode": "BOR", "QualifyingTime": 91.750},
    {"DriverCode": "COL", "QualifyingTime": 91.950},
    {"DriverCode": "LIN", "QualifyingTime": 92.200},
    {"DriverCode": "PER", "QualifyingTime": 92.450},
    {"DriverCode": "BOT", "QualifyingTime": 92.700},
])

RACE_CONFIG = {
    "round_2026":       3,
    "training_rounds":  [1, 2],
    "race_name":        "Japanese GP",
    "base_lap_time":    92.0,
    "RainProbability":  0.15,
    "Temperature":      16.0,
    "qualifying_2026":  QUALIFYING_2026,
    "fp3_pace":         FP3_LAPS,
    "fp2_pace":         FP2_LAPS,
    "fp1_pace":         FP1_LAPS,
    "fp1_laps":         FP1_LAPS,
    "fp2_laps":         FP2_LAPS,
    "fp3_laps":         FP3_LAPS,
}

ACTUAL_RESULT = {
    1: "ANT", 2: "PIA", 3: "LEC", 4: "RUS", 5: "HAM",
    "DNF": [], "DNS": [],
}

if __name__ == "__main__":
    results, model, mae = run(
        race_config=RACE_CONFIG, verbose=True,
        save_chart=os.path.join(os.path.dirname(__file__), "..", "charts", "japan.png"),
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