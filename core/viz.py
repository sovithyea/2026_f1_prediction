"""
viz.py
------
Visualisation helpers for the 2026 F1 prediction project.
"""

from __future__ import annotations
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from .config import TEAM_COLORS   # relative import — must stay as-is


def print_results(df: pd.DataFrame, race_name: str, mae) -> None:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    print(f"\n  Predicted 2026 {race_name} Results")
    print(f"  {'Pos':<4} {'Driver':<22} {'Team':<16} Predicted Lap (s)")
    print("  " + "─" * 62)
    for _, row in df.iterrows():
        pos = int(row["Position"])
        m   = medals.get(pos, f"   ")
        print(
            f"  {m} {pos:<3} "
            f"{row['FullName']:<22} "
            f"{row['Team']:<16} "
            f"{row['PredictedRaceTime (s)']:.2f}s"
        )
    mae_str = mae if isinstance(mae, str) else f"{mae:.2f}s"
    print(f"\n  Model MAE : {mae_str}")


def plot_results(
    df: pd.DataFrame,
    race_name: str,
    mae,
    save_path: Optional[str] = None,
) -> None:
    import os
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mae_str = mae if isinstance(mae, str) else f"{mae:.2f}s"
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    colors = [TEAM_COLORS.get(t, "#888888") for t in df["Team"]]
    ax.barh(df["DriverCode"], df["PredictedRaceTime (s)"],
            color=colors, edgecolor="#333333", height=0.7)
    ax.invert_yaxis()

    ax.set_xlabel("Predicted Mean Race Lap Time (s)", color="#C0C0C0", fontsize=11)
    ax.set_title(
        f"2026 {race_name} — Predicted Finishing Order\nMAE: {mae_str}",
        color="white", fontsize=14, pad=16,
    )
    ax.tick_params(colors="#C0C0C0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    patches = [mpatches.Patch(color=c, label=t) for t, c in TEAM_COLORS.items()]
    ax.legend(handles=patches, loc="lower right",
              framealpha=0.2, labelcolor="white", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved -> {save_path}")
    else:
        plt.show()
    plt.close()