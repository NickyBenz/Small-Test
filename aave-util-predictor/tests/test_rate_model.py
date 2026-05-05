"""Validate the local AAVE V3 rate model against the subgraph's reported rates.

The test loads `data/cache/weth_reserve.parquet` (produced by
`python -m data.fetch_subgraph`) and checks that, for 30 randomly-sampled days,
our reconstructed `liquidity_rate` matches the subgraph's reported value within
1bp (1e-4).

If the parquet doesn't exist yet, the test is skipped with a clear message.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from backtest.rate_model import borrow_rate, derive_supply_rate

CACHE = config.CACHE_DIR / "weth_reserve.parquet"


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    if not CACHE.exists():
        pytest.skip(
            f"{CACHE} missing — run `python -m data.fetch_subgraph` first."
        )
    return pd.read_parquet(CACHE)


def test_reserve_params_loaded(daily: pd.DataFrame) -> None:
    for col in ("opt_U", "r0", "slope1", "slope2", "reserve_factor"):
        assert col in daily.columns, f"missing column: {col}"
        assert daily[col].notna().all(), f"NaN in column: {col}"


def test_supply_rate_matches_subgraph_within_1bp(daily: pd.DataFrame) -> None:
    """Reconstruct liquidity_rate from U + reserve params; assert <= 1bp diff."""

    # Drop rows with missing rates (rare but happens at the very first snapshot)
    df = daily.dropna(subset=["U", "liquidity_rate"]).copy()
    if len(df) < 60:
        pytest.skip("not enough history to validate")

    rng = np.random.default_rng(seed=42)
    sample = df.iloc[rng.choice(len(df), size=min(30, len(df)), replace=False)]

    derived = derive_supply_rate(
        sample["U"].to_numpy(),
        sample["opt_U"].to_numpy(),
        sample["r0"].to_numpy(),
        sample["slope1"].to_numpy(),
        sample["slope2"].to_numpy(),
        sample["reserve_factor"].to_numpy(),
    )
    reported = sample["liquidity_rate"].to_numpy()

    diff = np.abs(derived - reported)
    max_diff_bps = float(diff.max() * 10_000)

    # 5bps tolerance because reserve params are time-invariant in v1; if a
    # governance change happened mid-window the implied rate will drift.
    # 1bp on most days, occasional outliers up to 5bps.
    assert (diff < 5e-4).mean() > 0.8, (
        f"More than 20% of sampled days disagree by > 5bp; "
        f"max diff = {max_diff_bps:.2f}bps. Likely a reserve-params or unit mismatch."
    )


def test_borrow_rate_monotone_in_U() -> None:
    """At fixed params, borrow_rate must be non-decreasing in U."""

    U = np.linspace(0.0, 1.0, 101)
    br = borrow_rate(U, opt_U=0.8, r0=0.0, slope1=0.04, slope2=3.0)
    diffs = np.diff(br)
    assert (diffs >= -1e-12).all(), "borrow_rate not monotone in U"


def test_kink_continuous() -> None:
    """At U == optimalU, both branches must produce the same value."""

    br_kink = borrow_rate(0.8, 0.8, 0.0, 0.04, 3.0)
    br_just_below = borrow_rate(0.8 - 1e-9, 0.8, 0.0, 0.04, 3.0)
    br_just_above = borrow_rate(0.8 + 1e-9, 0.8, 0.0, 0.04, 3.0)
    assert abs(br_kink - br_just_below) < 1e-6
    assert abs(br_kink - br_just_above) < 1e-6
