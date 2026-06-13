"""
model.py
--------
Gradient Boosting Regressor training and prediction.
Hyperparameters mirror the mar-antaya reference repo.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from .config import GBR_PARAMS, FEATURE_COLS


def train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    verbose: bool = True,
) -> tuple[GradientBoostingRegressor, float]:
    """
    Train the GBR model on historical race data.

    Parameters
    ----------
    X         : feature DataFrame (must contain FEATURE_COLS columns)
    y         : target Series — mean race lap time in seconds
    test_size : fraction of data held out for MAE evaluation
    verbose   : print MAE if True

    Returns
    -------
    (trained_model, mae)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X[FEATURE_COLS], y, test_size=test_size, random_state=42
    )
    model = GradientBoostingRegressor(**GBR_PARAMS)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))
    if verbose:
        print(f"  MAE: {mae:.2f}s")

    return model, mae


def predict(
    model: GradientBoostingRegressor,
    pred_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the trained model on a 2026 prediction feature DataFrame.

    Parameters
    ----------
    model   : trained GBR from train()
    pred_df : DataFrame with FEATURE_COLS — one row per driver

    Returns
    -------
    pred_df with two new columns:
        PredictedRaceTime (s)  — model output
        Position               — finishing order (1 = fastest)
    """
    df = pred_df.copy()
    df["PredictedRaceTime (s)"] = model.predict(df[FEATURE_COLS])
    df = df.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
    df["Position"] = range(1, len(df) + 1)
    return df