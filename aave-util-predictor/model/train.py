"""Walk-forward training of the calibrated logistic regression.

Loop structure (rolling-window variant):

    for t in range(train_window, n - retrain_every, retrain_every):
        train = X[t - train_window : t - embargo]      # purge last `embargo` rows
        test  = X[t : t + retrain_every]               # next chunk = OOS preds
        fit a calibrated logreg on train
        predict_proba on test, store p_increase[index]

Inside each fit, hyperparameters are chosen via `LogisticRegressionCV` with a
nested `TimeSeriesSplit(gap=embargo)` over the train slice. The resulting
classifier is then wrapped in `CalibratedClassifierCV(method="isotonic")`
using the same time-aware splits so the probabilities you threshold at 0.6
are well-calibrated rather than logistic-curve raw scores.

Output: `model_predictions.parquet` with columns:
    `[p_increase, signal, y_true]`
indexed by date over the OOS period.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
from features.builders import build_dataset

# Convergence warnings on tiny CV folds are noisy and harmless here.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _make_estimator(embargo: int) -> CalibratedClassifierCV:
    """One fresh, fully-specified estimator per walk-forward window."""

    inner_cv = TimeSeriesSplit(n_splits=5, gap=embargo)
    base = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=10,
                    cv=inner_cv,
                    penalty="l2",
                    solver="lbfgs",
                    scoring="neg_log_loss",
                    class_weight="balanced",
                    max_iter=5000,
                ),
            ),
        ]
    )
    # Calibrate on the *same* time-aware splits to avoid leakage from the
    # default stratified KFold inside CalibratedClassifierCV.
    return CalibratedClassifierCV(base, method="isotonic", cv=inner_cv)


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int = config.TRAIN_WINDOW_DAYS,
    retrain: int = config.RETRAIN_EVERY_DAYS,
    embargo: int = config.EMBARGO_DAYS,
) -> pd.DataFrame:
    """Run rolling-window walk-forward CV. Returns a probabilities frame."""

    n = len(X)
    if n < train_window + retrain + embargo:
        raise ValueError(
            f"need at least {train_window + retrain + embargo} rows; got {n}. "
            f"Either fetch more history or shrink TRAIN_WINDOW_DAYS in config.py."
        )

    probs: list[float] = []
    idxs: list[pd.Timestamp] = []
    t = train_window
    pbar = tqdm(total=(n - t) // retrain, desc="walk-forward")
    while t + retrain <= n:
        tr_end = t - embargo
        tr_start = max(0, tr_end - train_window)
        X_tr, y_tr = X.iloc[tr_start:tr_end], y.iloc[tr_start:tr_end]
        X_te = X.iloc[t : t + retrain]

        # Skip degenerate folds where one class is missing
        if y_tr.nunique() < 2:
            t += retrain
            pbar.update(1)
            continue

        est = _make_estimator(embargo)
        est.fit(X_tr.to_numpy(), y_tr.to_numpy())
        p = est.predict_proba(X_te.to_numpy())[:, 1]
        probs.extend(p.tolist())
        idxs.extend(X_te.index.tolist())

        t += retrain
        pbar.update(1)
    pbar.close()

    out = pd.DataFrame(
        {
            "p_increase": probs,
            "signal": [int(p > config.P_ENTER) for p in probs],
            "y_true": y.reindex(idxs).astype(int).tolist(),
        },
        index=pd.DatetimeIndex(idxs, name="ts"),
    )
    return out


def main() -> None:
    print("[train] building dataset…")
    X, y, meta = build_dataset()
    print(f"  X={X.shape}  y class balance={y.mean():.3f}  range={X.index[0].date()}→{X.index[-1].date()}")

    print("[train] walk-forward fitting…")
    preds = walk_forward(X, y)

    pred_path: Path = config.CACHE_DIR / "model_predictions.parquet"
    meta_path: Path = config.CACHE_DIR / "backtest_meta.parquet"
    preds.to_parquet(pred_path)
    meta.loc[preds.index].to_parquet(meta_path)
    print(f"  wrote {len(preds):,} OOS preds → {pred_path.name}")
    print(f"  wrote backtest meta → {meta_path.name}")
    print(
        f"  fire rate (p>{config.P_ENTER}) = "
        f"{preds['signal'].mean():.1%}  hit rate when fired = "
        f"{preds.loc[preds['signal'] == 1, 'y_true'].mean():.1%}"
    )


if __name__ == "__main__":
    sys.exit(main())
