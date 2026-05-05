# AAVE V3 WETH utilization predictor

A backtested logistic-regression strategy that predicts the 7-day directional
change in AAVE V3 WETH utilization on Ethereum and supplies WETH whenever the
calibrated probability of an increase exceeds 0.6.

The repo is intentionally small and dependency-light. Every step (data pull,
feature build, model fit, backtest) is a standalone Python module that caches
to parquet so iteration is fast.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in GRAPH_API_KEY (free at thegraph.com)

# 1. Pull all data (cached to data/cache/*.parquet)
python -m data.fetch_subgraph
python -m data.fetch_ohlcv
python -m data.fetch_funding
python -m data.fetch_options
python -m data.fetch_gas

# 2. Validate the AAVE rate model against the subgraph (must pass)
pytest tests/

# 3. Train + backtest
python -m model.train
python -m backtest.simulator

# 4. Inspect results
jupyter lab notebooks/02_results.ipynb
```

## What is being modelled

- Target: `y_t = 1 if U_{t+7d} > U_t else 0`, where `U` is WETH reserve utilization.
- Features: realized-vol z-scores (Yang-Zhang), volume z-score, vol/volume rolling
  correlation, vol autocorrelation, Lo-MacKinlay variance ratio, plus a utilization
  autocorrelation block (lag-1 ACF, half-life of mean reversion, gap to the kink),
  protocol features (borrow rate, liquidity flow, USDC cross-asset utilization),
  Deribit options features (DVOL, IV term-structure slope, 25-delta risk reversal,
  variance risk premium), perp funding, and day-of-week dummies.
- Model: `StandardScaler` + `LogisticRegressionCV` (L2, balanced, `TimeSeriesSplit`
  with `gap=7`), wrapped in `CalibratedClassifierCV(method="isotonic")`.
- Validation: rolling 365-day train, 30-day retrain cadence, 7-day embargo to
  prevent label leakage.

## What the backtester simulates

- AAVE V3 piecewise-linear interest-rate model with time-varying reserve params
  (`baseVariableBorrowRate`, `slope1`, `slope2`, `optimalUsageRatio`,
  `reserveFactor`) sourced from the subgraph.
- Self-impact: when the strategy supplies, U is recomputed as
  `debt / (liquidity + S + debt)` and the resulting supply rate is used for PnL.
- Continuous accrual: `equity *= exp(supply_rate * dt / 365)`.
- Gas debits: `gas_units * gwei * 1e-9 * eth_price` per supply or withdraw.
- Hysteresis (enter at 0.60, exit at 0.40) and 3-day minimum hold to prevent churn.
- Baselines: always-supplied, random-signal-matched-frequency, naive `dU_lag1` rule.

## Repo layout

```
config.py               # constants: reserve id, windows, strategy knobs
data/                   # cached parquets + fetchers
backtest/rate_model.py  # AAVE V3 rate function + self-impact
backtest/simulator.py   # walk-forward PnL with hysteresis
features/builders.py    # all feature transforms
model/train.py          # walk-forward fit + calibrated logreg
model/evaluate.py       # AUC, PR, Brier, reliability, PnL metrics
tests/                  # rate-model validation against subgraph
notebooks/              # EDA + results
```

## Caveats

- Past performance, etc. The model is calibrated only on ~18 months of data; AAVE's
  rate parameters change via governance and structural breaks happen.
- Self-impact at $1M notional on the WETH reserve is small (~0.1-0.3% on U), but
  scale notional with caution.
- Gas drag uses historical median; spike scenarios are not modelled.
