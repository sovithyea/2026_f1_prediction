"""
china.py
--------
Round 2 — 2026 Chinese Grand Prix  (Sprint weekend)
Shanghai International Circuit  |  15 March 2026  |  56 laps

Training data : 2025 Chinese GP (Round 2, FastF1)
Qualifying    : Real 2026 times entered below
Race result   : ANT > RUS > HAM (actual — for validation)

Run:
    python races/china.py
"""

import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.predictor import run

# ---------------------------------------------------------------------------
# 2026 Chinese GP — qualifying times (14 March 2026)
# Pole: Kimi Antonelli  1:32.064
# ---------------------------------------------------------------------------
QUALIFYING_2026 = pd.DataFrame([
    {"DriverCode": "ANT", "QualifyingTime": 92.064},  # P1 — pole
    {"DriverCode": "RUS", "QualifyingTime": 92.310},  # P2
    {"DriverCode": "HAM", "QualifyingTime": 92.520},  # P3
    {"DriverCode": "LEC", "QualifyingTime": 92.720},  # P4
    {"DriverCode": "NOR", "QualifyingTime": 92.900},  # P5
    {"DriverCode": "PIA", "QualifyingTime": 93.030},  # P6 (est.)
    {"DriverCode": "VER", "QualifyingTime": 93.150},  # P7 (est.)
    {"DriverCode": "HAD", "QualifyingTime": 93.310},  # P8 (est.)
    {"DriverCode": "SAI", "QualifyingTime": 93.500},  # P9 (est.)
    {"DriverCode": "ALO", "QualifyingTime": 93.680},  # P10 (est.)
    {"DriverCode": "ALB", "QualifyingTime": 93.900},
    {"DriverCode": "GAS", "QualifyingTime": 94.050},
    {"DriverCode": "LAW", "QualifyingTime": 94.200},
    {"DriverCode": "STR", "QualifyingTime": 94.350},
    {"DriverCode": "OCO", "QualifyingTime": 94.500},
    {"DriverCode": "BEA", "QualifyingTime": 94.700},
    {"DriverCode": "HUL", "QualifyingTime": 94.900},
    {"DriverCode": "BOR", "QualifyingTime": 95.100},
    {"DriverCode": "COL", "QualifyingTime": 95.300},
    {"DriverCode": "LIN", "QualifyingTime": 95.500},
    {"DriverCode": "PER", "QualifyingTime": 95.700},
    {"DriverCode": "BOT", "QualifyingTime": 95.900},
])

RACE_CONFIG = {
    "round_2026":      2,
        "training_rounds":  [1],
    "race_name":       "Chinese GP",
    "base_lap_time":   96.0,       # Shanghai ~1:36 average race lap
    "RainProbability": 0.10,
    "Temperature":     18.0,
    "qualifying_2026": QUALIFYING_2026,
}

ACTUAL_RESULT = {
    1: "ANT",  # Kimi Antonelli    (Mercedes)
    2: "RUS",  # George Russell    (Mercedes)
    3: "HAM",  # Lewis Hamilton    (Ferrari)
    4: "LEC",
    5: "NOR",
    "DNF": [],
    "DNS": [],
}

if __name__ == "__main__":
    results, model, mae = run(
        race_config=RACE_CONFIG, verbose=True,
        save_chart=os.path.join(os.path.dirname(__file__), "..", "charts", "china.png"),
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
    top1_correct = predicted_order[0] == ACTUAL_RESULT.get(1)
    print(f"\n  Winner prediction: {'Correct' if top1_correct else 'Wrong'}")
    print(f"  Model MAE: {mae:.2f}s")