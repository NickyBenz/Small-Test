"""AAVE V3 piecewise-linear interest-rate model + self-impact helper.

Mirrors `DefaultReserveInterestRateStrategy.sol`:

    if U <= optimalU:
        borrow_rate = r0 + (U / optimalU) * slope1
    else:
        excess      = (U - optimalU) / (1 - optimalU)
        borrow_rate = r0 + slope1 + excess * slope2

    supply_rate = U * borrow_rate * (1 - reserve_factor)

Everything below is vectorised across pandas Series so it can be used inside
the backtester with the time-varying reserve parameters columns we attached
in `data/fetch_subgraph.py`.
"""

from __future__ import annotations

from typing import overload

import numpy as np
import pandas as pd

ArrayLike = float | np.ndarray | pd.Series


def _as_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def borrow_rate(
    U: ArrayLike,
    opt_U: ArrayLike,
    r0: ArrayLike,
    slope1: ArrayLike,
    slope2: ArrayLike,
) -> np.ndarray:
    """AAVE V3 piecewise borrow-rate, vectorised."""

    U_arr = _as_array(U)
    opt = _as_array(opt_U)
    r0_arr = _as_array(r0)
    s1 = _as_array(slope1)
    s2 = _as_array(slope2)

    below = r0_arr + (U_arr / opt) * s1
    # Clip the (1 - opt) denominator to avoid divide-by-zero on degenerate
    # parameters; the result is irrelevant in that regime.
    excess = (U_arr - opt) / np.clip(1.0 - opt, 1e-9, None)
    above = r0_arr + s1 + excess * s2
    return np.where(U_arr <= opt, below, above)


def supply_rate(
    U: ArrayLike,
    br: ArrayLike,
    reserve_factor: ArrayLike,
) -> np.ndarray:
    """U * borrow_rate * (1 - reserve_factor)."""

    return _as_array(U) * _as_array(br) * (1.0 - _as_array(reserve_factor))


def derive_supply_rate(
    U: ArrayLike,
    opt_U: ArrayLike,
    r0: ArrayLike,
    slope1: ArrayLike,
    slope2: ArrayLike,
    reserve_factor: ArrayLike,
) -> np.ndarray:
    """Convenience: borrow + supply in one call."""

    br = borrow_rate(U, opt_U, r0, slope1, slope2)
    return supply_rate(U, br, reserve_factor)


def U_with_supply(
    U_market: ArrayLike,
    debt: ArrayLike,
    liquidity: ArrayLike,
    supply_size: ArrayLike,
) -> np.ndarray:
    """Recompute utilization after we add `supply_size` to the reserve.

    `liquidity` here is *available* liquidity (not totalLiquidity) so that
    `liquidity + debt = totalLiquidity` is preserved. The market's `U_market`
    is supplied for context but isn't used directly — we recompute from
    balances to keep the formula self-consistent.
    """

    debt_arr = _as_array(debt)
    new_total = _as_array(liquidity) + _as_array(supply_size) + debt_arr
    return np.where(new_total > 0, debt_arr / new_total, _as_array(U_market))


@overload
def accrue(equity: float, rate: float, dt_days: float) -> float: ...


@overload
def accrue(equity: ArrayLike, rate: ArrayLike, dt_days: ArrayLike) -> np.ndarray: ...


def accrue(equity, rate, dt_days):
    """Continuous-compound interest accrual: equity * exp(rate * dt / 365)."""

    return _as_array(equity) * np.exp(_as_array(rate) * _as_array(dt_days) / 365.0)
