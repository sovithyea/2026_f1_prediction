"""
monaco.py
---------
Round 6 — 2026 Monaco Grand Prix
Circuit de Monaco  |  7 June 2026  |  78 laps

Run:  python races/monaco.py
"""

import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.predictor import run

FP1_LAPS = None
FP2_LAPS = None
FP3_LAPS = None

# ---------------------------------------------------------------------------
# Qualifying — 6 June 2026
# Pole: Kimi Antonelli  1:12.051  (+0.043 over VER)
# LEC hit wall on final Q3 run → P4
# Full Q3: ANT VER HAM LEC HAD RUS PIA NOR GAS LAW
# ---------------------------------------------------------------------------
QUALIFYING_2026 = pd.DataFrame([
    {"DriverCode": "ANT", "QualifyingTime": 72.051},  # P1 — pole
    {"DriverCode": "VER", "QualifyingTime": 72.094},  # P2 (+0.043) — DNF race
    {"DriverCode": "HAM", "QualifyingTime": 72.290},  # P3 → P2 race
    {"DriverCode": "LEC", "QualifyingTime": 72.420},  # P4 — hit wall in Q3 → DNF race
    {"DriverCode": "HAD", "QualifyingTime": 72.600},  # P5 → P3 race (maiden podium)
    {"DriverCode": "RUS", "QualifyingTime": 72.780},  # P6 — penalised → P12
    {"DriverCode": "PIA", "QualifyingTime": 72.960},  # P7 → P4 race
    {"DriverCode": "NOR", "QualifyingTime": 73.100},  # P8 — DNF race
    {"DriverCode": "GAS", "QualifyingTime": 73.260},  # P9 — penalties then rescinded → P3
    {"DriverCode": "LAW", "QualifyingTime": 73.450},  # P10 → P5 race
    {"DriverCode": "ALB", "QualifyingTime": 73.680},  # P11 → P8 race
    {"DriverCode": "SAI", "QualifyingTime": 73.850},
    {"DriverCode": "HUL", "QualifyingTime": 74.050},
    {"DriverCode": "COL", "QualifyingTime": 74.250},
    {"DriverCode": "LIN", "QualifyingTime": 74.500},  # P6 race
    {"DriverCode": "BOR", "QualifyingTime": 74.750},
    {"DriverCode": "OCO", "QualifyingTime": 75.050},  # P9 race
    {"DriverCode": "PER", "QualifyingTime": 75.350},  # penalised → P15
    {"DriverCode": "BEA", "QualifyingTime": 75.700},  # DNF
    {"DriverCode": "BOT", "QualifyingTime": 76.050},  # DNF
    {"DriverCode": "ALO", "QualifyingTime": 76.500},  # P21 start → P10 race
    {"DriverCode": "STR", "QualifyingTime": 77.000},  # P22 — DNF
])

RACE_CONFIG = {
    "round_2026":       6,
    "training_rounds":  [1, 2, 3, 4, 5],
    "race_name":        "Monaco GP",
    "base_lap_time":    74.5,
    "RainProbability":  0.10,
    "Temperature":      21.0,
    "qualifying_2026":  QUALIFYING_2026,
    "fp3_pace":         FP3_LAPS,
    "fp2_pace":         FP2_LAPS,
    "fp1_pace":         FP1_LAPS,
    "fp1_laps":         FP1_LAPS,
    "fp2_laps":         FP2_LAPS,
    "fp3_laps":         FP3_LAPS,
}

# FIA confirmed result (pre-appeal: HAD P3, post-appeal: GAS P3)
ACTUAL_RESULT = {
    1: "ANT", 2: "HAM", 3: "HAD", 4: "PIA", 5: "LAW",
    "DNF": ["VER", "NOR", "BOT", "BEA", "STR", "LEC"], "DNS": [],
}

if __name__ == "__main__":
    results, model, mae = run(
        race_config=RACE_CONFIG, verbose=True,
        save_chart=os.path.join(os.path.dirname(__file__), "..", "charts", "monaco.png"),
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