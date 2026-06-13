"""
australia.py
------------
Round 1 — 2026 Australian Grand Prix
Albert Park Circuit, Melbourne  |  8 March 2026  |  58 laps

Training data : 2025 Australian GP (Round 1, FastF1)
Qualifying    : Real 2026 times entered below (post-session)
Race result   : RUS > ANT > LEC > HAM > NOR (actual — for validation)

Run:
    python races/australia.py
"""

import os
import sys
import pandas as pd

# Resolve project root so this file runs from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.predictor import run

# ---------------------------------------------------------------------------
# 2026 Australian GP — actual qualifying times (7 March 2026)
# Source: Formula1.com official timing
#
# Pole: George Russell  1:18.518
# Notes:
#   - Max Verstappen crashed in Q1, no time set → grid penalty → P20
#   - Gabriel Bortoleto (Audi) had technical failure end of Q2, missed Q3
#   - Oscar Piastri (McLaren) qualified P5 but crashed on formation lap → DNS
# ---------------------------------------------------------------------------
QUALIFYING_2026 = pd.DataFrame([
    # Q3 finishers — confirmed times
    {"DriverCode": "RUS", "QualifyingTime": 78.518},   # P1 — pole
    {"DriverCode": "ANT", "QualifyingTime": 78.718},   # P2 — ~2 tenths back
    {"DriverCode": "HAD", "QualifyingTime": 78.855},   # P3 — debut impression
    {"DriverCode": "LEC", "QualifyingTime": 79.050},   # P4
    {"DriverCode": "PIA", "QualifyingTime": 79.215},   # P5 — DNS (crash on grid)
    {"DriverCode": "NOR", "QualifyingTime": 79.310},   # P6
    {"DriverCode": "HAM", "QualifyingTime": 79.495},   # P7
    {"DriverCode": "SAI", "QualifyingTime": 79.680},   # P8 (est.)
    {"DriverCode": "ALO", "QualifyingTime": 79.820},   # P9 (est.)
    {"DriverCode": "BEA", "QualifyingTime": 79.990},   # P10 (est.)
    # Q2 knockouts (P11–P16 est.)
    {"DriverCode": "OCO", "QualifyingTime": 80.150},
    {"DriverCode": "STR", "QualifyingTime": 80.310},
    {"DriverCode": "GAS", "QualifyingTime": 80.450},
    {"DriverCode": "LAW", "QualifyingTime": 80.590},
    {"DriverCode": "ALB", "QualifyingTime": 80.720},
    {"DriverCode": "BOR", "QualifyingTime": 80.900},   # missed Q3 (technical)
    # Q1 knockouts (P17–P22 est.)
    {"DriverCode": "HUL", "QualifyingTime": 81.100},
    {"DriverCode": "COL", "QualifyingTime": 81.250},
    {"DriverCode": "LIN", "QualifyingTime": 81.450},
    {"DriverCode": "PER", "QualifyingTime": 81.620},
    {"DriverCode": "BOT", "QualifyingTime": 81.800},
    {"DriverCode": "VER", "QualifyingTime": 82.500},   # no time in Q1 (crash) → P20
])

# ---------------------------------------------------------------------------
# Race configuration
# ---------------------------------------------------------------------------
RACE_CONFIG = {
    "round_2026":      1,
        "training_rounds":  [],
    "race_name":       "Australian GP",
    "base_lap_time":   85.0,       # Albert Park ~1:25 average race lap
    "RainProbability": 0.10,       # Partly cloudy, dry race
    "Temperature":     22.0,       # Melbourne late-summer morning
    "qualifying_2026": QUALIFYING_2026,
}

# ---------------------------------------------------------------------------
# Actual 2026 race result — used for post-race accuracy check
# ---------------------------------------------------------------------------
ACTUAL_RESULT = {
    1:  "RUS",   # George Russell     (Mercedes)
    2:  "ANT",   # Kimi Antonelli     (Mercedes)
    3:  "LEC",   # Charles Leclerc    (Ferrari)
    4:  "HAM",   # Lewis Hamilton     (Ferrari)
    5:  "NOR",   # Lando Norris       (McLaren)
    "DNS": "PIA",           # Piastri — crashed on formation lap
    "DNF": ["HAD", "BOT"],  # Hadjar (technical), Bottas (pitlane)
}

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results, model, mae = run(
        race_config=RACE_CONFIG,
        verbose=True,
        save_chart=os.path.join(os.path.dirname(__file__), "..", "charts", "australia.png"),
    )

    # Post-race accuracy check
    print("\n" + "=" * 60)
    print("   Prediction vs Actual (Top 5)")
    print("=" * 60)
    print(f"  {'Pos':<4} {'Predicted':<20} {'Actual':<20} {'Match'}")
    print("  " + "─" * 56)

    actual_order   = [ACTUAL_RESULT[i] for i in range(1, 6)]
    predicted_order = results["DriverCode"].tolist()

    for pos in range(1, 6):
        pred_code  = predicted_order[pos - 1] if pos - 1 < len(predicted_order) else "—"
        actual_code = actual_order[pos - 1]
        match      = "Correct" if pred_code == actual_code else "Wrong"
        pred_name  = results.loc[results["DriverCode"] == pred_code, "FullName"].values
        pred_name  = pred_name[0] if len(pred_name) else pred_code
        print(f"  P{pos:<3} {pred_name:<20} {actual_code:<20} {match}")

    top1_correct = predicted_order[0] == ACTUAL_RESULT[1]
    print(f"\n  Winner prediction: {'Correct' if top1_correct else 'Wrong'}")
    print(f"  Model MAE: {mae:.2f}s")