"""
viz.py
------
Visualisation helpers for the 2026 F1 prediction project.

plot_results()   — horizontal bar chart coloured by team
print_results()  — formatted console output with medals
"""

from __future__ import annotations

from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from config import TEAM_COLORS


def print_results(df: pd.DataFrame, race_name: str, mae: float) -> None:
    """
    Print a formatted finishing-order table to the console.

    Parameters
    ----------
    df        : results DataFrame with Position, FullName, Team,
                PredictedRaceTime (s)
    race_name : e.g. "Australian GP"
    mae       : model MAE in seconds
    """
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    print(f"\n  🏆  Predicted 2026 {race_name} Results")
    print(f"  {'Pos':<4} {'Driver':<22} {'Team':<16} Predicted Lap (s)")
    print("  " + "─" * 62)
    for _, row in df.iterrows():
        pos = int(row["Position"])
        m   = medals.get(pos, "  ")
        print(
            f"  {m} {pos:<3} "
            f"{row['FullName']:<22} "
            f"{row['Team']:<16} "
            f"{row['PredictedRaceTime (s)']:.3f}s"
        )
    print(f"\n  Model MAE: {mae:.2f}s")


def plot_results(
    df: pd.DataFrame,
    race_name: str,
    mae: float,
    save_path: Optional[str] = None,
) -> None:
    """
    Dark-themed horizontal bar chart of predicted race lap times.

    Parameters
    ----------
    df        : results DataFrame (must have Team, DriverCode,
                PredictedRaceTime (s))
    race_name : e.g. "Australian GP"
    mae       : model MAE in seconds
    save_path : if provided, saves PNG to this path instead of showing
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0D0D0D")
    ax.set_facecolor("#1A1A1A")

    colors = [TEAM_COLORS.get(t, "#888888") for t in df["Team"]]
    ax.barh(
        df["DriverCode"],
        df["PredictedRaceTime (s)"],
        color=colors,
        edgecolor="#333333",
        height=0.7,
    )
    ax.invert_yaxis() # P1 at the top

    ax.set_xlabel("Predicted Mean Race Lap Time (s)", color="#C0C0C0", fontsize=11)
    ax.set_title(
        f"  2026 {race_name} — Predicted Finishing Order\n"
        f"Model MAE: {mae:.2f}s",
        color="white", fontsize=14, pad=16,
    )
    ax.tick_params(colors="#C0C0C0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    legend_patches = [
        mpatches.Patch(color=c, label=t)
        for t, c in TEAM_COLORS.items()
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        framealpha=0.2,
        labelcolor="white",
        fontsize=8,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved -> {save_path}")
    else:
        plt.show()
    plt.close()