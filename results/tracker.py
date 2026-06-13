"""
tracker.py
----------
2026 F1 Season — Prediction Accuracy Tracker

Reads prediction CSVs saved by each race file and compares
against actual results in actual.csv.

Run from project root:
    python results/tracker.py
"""

import os
import sys
import pandas as pd

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, "results")


def load_actual() -> pd.DataFrame:
    path = os.path.join(RES_DIR, "actual.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def load_prediction(race_name: str) -> pd.DataFrame | None:
    slug = race_name.lower().replace(" ", "_").replace("/", "_")
    path = os.path.join(RES_DIR, f"{slug}_prediction.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def score(pred_df: pd.DataFrame, actual_row: pd.Series) -> dict:
    pred5   = pred_df.head(5)["DriverCode"].tolist()
    actual5 = [actual_row[f"p{i}"] for i in range(1, 6)
               if pd.notna(actual_row.get(f"p{i}"))]

    p1_hit     = bool(pred5 and pred5[0] == actual5[0])
    top3_hits  = len([p for p in pred5[:3] if p in actual5[:3]])
    top5_hits  = len([p for p in pred5      if p in actual5])

    pos_errors = []
    for i, code in enumerate(actual5, 1):
        pos_errors.append(abs(pred5.index(code) + 1 - i) if code in pred5 else 5)
    avg_err = round(sum(pos_errors) / len(pos_errors), 2) if pos_errors else 5.0

    return {"p1": p1_hit, "top3": top3_hits, "top5": top5_hits, "err": avg_err,
            "pred5": pred5, "actual5": actual5}


def main():
    actual_df = load_actual()
    if actual_df.empty:
        print("  actual.csv not found in results/"); return

    print("\n" + "=" * 62)
    print("  2026 F1 SEASON — PREDICTION ACCURACY TRACKER")
    print("=" * 62)

    rows, missing = [], []

    for _, row in actual_df.iterrows():
        name    = row["race"]
        pred_df = load_prediction(name)

        if pred_df is None:
            missing.append(name); continue

        s = score(pred_df, row)

        # Per-race breakdown
        print(f"\n  Round {int(row['round'])} — {name}")
        print(f"  {'Pos':<4} {'Predicted':<10} {'Actual':<10} {'Match'}")
        print("  " + "─" * 34)
        for i, (pred, actual) in enumerate(zip(s["pred5"], s["actual5"]), 1):
            mark = "✅" if pred == actual else "❌"
            print(f"  P{i:<3} {pred:<10} {actual:<10} {mark}")
        print(f"  Top-5 hits: {s['top5']}/5  |  Avg position error: {s['err']} places")

        rows.append({"round": int(row["round"]), "race": name,
                     "p1": s["p1"], "top3": s["top3"],
                     "top5": s["top5"], "err": s["err"]})

    if missing:
        print(f"\n  Not yet predicted: {', '.join(missing)}")

    if not rows:
        print("\n  No predictions saved yet — run a race file first."); return

    # Season summary
    df = pd.DataFrame(rows)
    n  = len(df)
    p1_pct   = df["p1"].sum() / n * 100
    top5_avg = df["top5"].mean()
    err_avg  = df["err"].mean()

    print(f"\n{'='*62}")
    print(f"   SEASON SUMMARY — {n} races predicted")
    print(f"{'='*62}")
    print(f"  {'Race':<22} {'P1':<5} {'Top5':<8} {'Avg Pos Err'}")
    print(f"  {'─'*50}")
    for _, r in df.iterrows():
        p1 = "✅" if r["p1"] else "❌"
        print(f"  {r['race']:<22} {p1:<5} {int(r['top5'])}/5     {r['err']:.2f}")
    print(f"  {'─'*50}")
    print(f"  {'TOTALS':<22} {df['p1'].sum()}/{n}  {top5_avg:.1f}/5    {err_avg:.2f}")
    print(f"\n  Winner accuracy : {p1_pct:.0f}%")
    print(f"  Top-5 accuracy  : {top5_avg/5*100:.0f}%")
    print(f"  Avg pos error   : {err_avg:.2f} places\n")


if __name__ == "__main__":
    main()