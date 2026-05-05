"""Classification + strategy metrics used by the results notebook.

Pure functions — no I/O — so the notebook (or any caller) can hold the data in
memory and just plot the outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true: pd.Series, p: pd.Series, threshold: float = 0.6) -> dict:
    """One-shot summary of probabilistic classifier quality."""

    y, p_arr = y_true.to_numpy().astype(int), p.to_numpy().astype(float)
    fired = p_arr > threshold

    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p_arr)),
        "ap": float(average_precision_score(y, p_arr)),
        "brier": float(brier_score_loss(y, p_arr)),
        "log_loss": float(log_loss(y, np.clip(p_arr, 1e-6, 1 - 1e-6))),
        "fire_rate": float(fired.mean()),
        "precision_at_thr": float(y[fired].mean()) if fired.any() else float("nan"),
        "recall_at_thr": float(y[fired].sum() / max(1, y.sum())),
    }


def reliability_curve(y_true: pd.Series, p: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Mean-predicted vs observed-frequency, with per-bin counts."""

    bins = np.linspace(0, 1, n_bins + 1)
    df = pd.DataFrame({"y": y_true.to_numpy(), "p": p.to_numpy()})
    df["bin"] = pd.cut(df["p"], bins, include_lowest=True, labels=False)
    grouped = df.groupby("bin").agg(p_mean=("p", "mean"), y_mean=("y", "mean"), n=("y", "size"))
    return grouped


def annualized_return(equity: pd.Series) -> float:
    """Return CAGR from an equity curve."""

    n_days = (equity.index[-1] - equity.index[0]).days
    if n_days < 1:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (365 / n_days) - 1)


def sharpe(daily_ret: pd.Series, risk_free: float = 0.0) -> float:
    """Daily-Sharpe annualized to crypto convention (sqrt(365))."""

    excess = daily_ret - risk_free / 365
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(365))


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough drawdown of an equity curve (negative number)."""

    cummax = equity.cummax()
    return float((equity / cummax - 1).min())


def strategy_summary(equity: pd.Series, label: str = "") -> dict:
    daily_ret = equity.pct_change().dropna()
    return {
        "label": label,
        "apy": annualized_return(equity),
        "sharpe": sharpe(daily_ret),
        "max_dd": max_drawdown(equity),
        "final_equity": float(equity.iloc[-1]),
    }
