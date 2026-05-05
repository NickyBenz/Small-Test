"""Strategy simulator with self-impact, gas, hysteresis, and baselines.

Inputs (from `model.train`):
- `model_predictions.parquet` — daily `p_increase`
- `backtest_meta.parquet`     — daily `U`, `debt`, `available_liquidity`,
                                reserve params, `gas_gwei`, `price_usd`

What it produces (`backtest_results.parquet`):
- For each strategy: a daily equity curve plus an exit/entry trace.

Strategies compared:
1. `model`   — supply when `p_increase > P_ENTER`, exit when `< P_EXIT`,
               minimum hold of `MIN_HOLD_DAYS`, gas debited per tx.
2. `always`  — always supplied. The honest hurdle.
3. `random`  — random fire schedule with the same fire rate as `model`.
               Controls for "trades fewer days = avoids low-rate days."
4. `naive`   — supply if yesterday's ΔU > 0. Beats? Then the model is just
               recovering AR(1).

Self-impact: while a strategy is supplied, the day's effective U is recomputed
as `debt / (debt + liquidity + S)` and the supply rate is calculated off that.
At $1M notional on WETH this is small (~0.1-0.3% of U) but free to include.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from backtest.rate_model import accrue, derive_supply_rate, U_with_supply


# ---------------------------------------------------------------------------
# Single-strategy simulator
# ---------------------------------------------------------------------------


@dataclass
class StratResult:
    label: str
    equity: pd.Series
    position: pd.Series
    n_supplies: int
    n_withdraws: int
    total_gas_usd: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                f"{self.label}_equity": self.equity,
                f"{self.label}_position": self.position,
            }
        )


def _gas_usd_for_tx(gas_gwei: float, eth_price: float) -> float:
    return config.GAS_UNITS_PER_TX * gas_gwei * 1e-9 * eth_price


def _supply_rate_for_day(row: pd.Series, deployed: bool, notional_usd: float) -> float:
    """Effective supply rate today, accounting for self-impact if deployed."""

    # Convert our USD notional to WETH at today's price for the self-impact calc
    notional_weth = notional_usd / row["price_usd"] if deployed else 0.0
    U_eff = U_with_supply(
        row["U"],
        row["debt"],
        row["available_liquidity"],
        notional_weth,
    )
    return float(
        derive_supply_rate(
            U_eff,
            row["opt_U"],
            row["r0"],
            row["slope1"],
            row["slope2"],
            row["reserve_factor"],
        )
    )


def simulate(
    daily_signal: pd.Series,
    meta: pd.DataFrame,
    *,
    label: str,
    notional_usd: float = config.NOTIONAL_USD,
    p_enter: float = config.P_ENTER,
    p_exit: float = config.P_EXIT,
    min_hold: int = config.MIN_HOLD_DAYS,
    use_hysteresis: bool = True,
) -> StratResult:
    """Walk one strategy day by day, returning equity and trade stats.

    `daily_signal` semantics depend on `use_hysteresis`:
      - if True: the signal is a probability in [0, 1] and we apply
        enter/exit thresholds + min-hold,
      - if False: the signal is already a 0/1 position decision and we
        respect it as-is (used for `always` and `random` baselines).
    """

    df = meta.join(daily_signal.rename("sig"), how="inner").sort_index()
    if df.empty:
        raise ValueError(f"[{label}] meta and signal don't overlap")

    equity = np.empty(len(df))
    position = np.zeros(len(df), dtype=np.int8)
    cash = notional_usd
    pos = 0
    hold = 0
    n_sup = n_wd = 0
    gas_total = 0.0

    for i, (_, row) in enumerate(df.iterrows()):
        # 1) accrue PnL on yesterday's position over today's day
        if pos == 1:
            sr = _supply_rate_for_day(row, deployed=True, notional_usd=cash)
            cash = float(accrue(cash, sr, 1.0))
        # cash sits idle when not deployed (no risk-free leg in v1)

        # 2) decide today's position from today's signal
        if use_hysteresis:
            sig = row["sig"]
            want_in = bool(sig > p_enter) if pos == 0 else bool(sig >= p_exit)
            # Min-hold: once entered, refuse to exit before MIN_HOLD_DAYS
            if pos == 1 and hold < min_hold:
                want_in = True
        else:
            want_in = bool(row["sig"] >= 0.5)

        if want_in and pos == 0:
            gas = _gas_usd_for_tx(row["gas_gwei"], row["price_usd"])
            cash -= gas
            gas_total += gas
            pos, hold, n_sup = 1, 0, n_sup + 1
        elif (not want_in) and pos == 1:
            gas = _gas_usd_for_tx(row["gas_gwei"], row["price_usd"])
            cash -= gas
            gas_total += gas
            pos, hold, n_wd = 0, 0, n_wd + 1
        else:
            hold += 1

        equity[i] = cash
        position[i] = pos

    return StratResult(
        label=label,
        equity=pd.Series(equity, index=df.index, name=f"{label}_equity"),
        position=pd.Series(position, index=df.index, name=f"{label}_position"),
        n_supplies=n_sup,
        n_withdraws=n_wd,
        total_gas_usd=gas_total,
    )


# ---------------------------------------------------------------------------
# Baseline signal generators
# ---------------------------------------------------------------------------


def baseline_always(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(1.0, index=idx, name="always")


def baseline_random(idx: pd.DatetimeIndex, fire_rate: float, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series((rng.random(len(idx)) < fire_rate).astype(float), index=idx, name="random")


def baseline_naive_du(meta: pd.DataFrame) -> pd.Series:
    """Supply if yesterday's ΔU > 0."""

    return (meta["U"].diff().shift(1) > 0).astype(float).rename("naive")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all() -> pd.DataFrame:
    preds = pd.read_parquet(config.CACHE_DIR / "model_predictions.parquet")
    meta = pd.read_parquet(config.CACHE_DIR / "backtest_meta.parquet")

    # Align both on the same index (preds drives, meta is a superset)
    common = preds.index.intersection(meta.index)
    preds = preds.loc[common]
    meta = meta.loc[common]

    fire_rate = float((preds["p_increase"] > config.P_ENTER).mean())
    print(f"[backtest] OOS days={len(preds):,}  model fire-rate={fire_rate:.1%}")

    results: list[StratResult] = []

    print("  - model (calibrated logreg, hysteresis 0.60/0.40, min-hold 3d)")
    results.append(
        simulate(preds["p_increase"], meta, label="model", use_hysteresis=True)
    )

    print("  - always supplied")
    results.append(
        simulate(baseline_always(preds.index), meta, label="always", use_hysteresis=False)
    )

    print(f"  - random matched fire rate ({fire_rate:.1%})")
    results.append(
        simulate(
            baseline_random(preds.index, fire_rate),
            meta,
            label="random",
            use_hysteresis=False,
        )
    )

    print("  - naive dU_lag1 > 0")
    results.append(
        simulate(baseline_naive_du(meta), meta, label="naive", use_hysteresis=False)
    )

    df = pd.concat([r.to_frame() for r in results], axis=1)
    df.to_parquet(config.CACHE_DIR / "backtest_results.parquet")

    print()
    print("strategy   |  CAGR     n_sup  n_wd   gas_usd")
    print("-----------+--------------------------------")
    for r in results:
        n = (r.equity.index[-1] - r.equity.index[0]).days
        cagr = (r.equity.iloc[-1] / r.equity.iloc[0]) ** (365 / max(n, 1)) - 1
        print(
            f"{r.label:10s} | {cagr:6.2%}  {r.n_supplies:5d}  {r.n_withdraws:4d}  ${r.total_gas_usd:,.0f}"
        )

    return df


def main() -> None:
    run_all()


if __name__ == "__main__":
    sys.exit(main())
