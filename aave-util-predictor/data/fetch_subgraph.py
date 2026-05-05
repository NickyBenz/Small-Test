"""Pull AAVE V3 reserve history from The Graph and cache as parquet.

Two artifacts are produced per reserve:

1. `<symbol>_reserve.parquet` — daily-resampled state with columns:
   `U, available_liquidity, debt, liquidity_rate, borrow_rate, price_usd,
    opt_U, r0, slope1, slope2, reserve_factor`.
2. `<symbol>_reserve_raw.parquet` — every raw `ReserveParamsHistoryItem` snapshot
   (event-driven, irregular cadence). Useful for sanity-checking resampling.

Why we resample: AAVE emits a `ReserveParamsHistoryItem` on every
borrow / repay / supply / withdraw, which is not a fixed-time series. Downstream
features assume a daily grid, so we forward-fill the *last* snapshot of each
UTC day.

Pagination: The Graph caps `skip` at 5000. We page with `timestamp_gt: lastTs`
instead, which has no skip ceiling.

Units: AAVE stores rates in rays (1e27) and balances in token decimals (1e18
for WETH). We convert at this boundary so downstream code only ever sees
floats in [0, 1] for utilization/rates and human-readable token quantities.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# GraphQL primitives
# ---------------------------------------------------------------------------

_HISTORY_QUERY = """
query History($reserve: String!, $tsGt: Int!) {
  reserveParamsHistoryItems(
    first: 1000
    where: { reserve: $reserve, timestamp_gt: $tsGt }
    orderBy: timestamp
    orderDirection: asc
  ) {
    timestamp
    utilizationRate
    availableLiquidity
    totalLiquidity
    liquidityRate
    variableBorrowRate
    stableBorrowRate
    priceInUsd
  }
}
"""

# Reserve params (rate-model knobs) live on the Reserve entity itself. They
# change occasionally via governance — for v1 we snapshot the *current* value
# and apply it across history. If a governance change is detected (slope change
# materially) we'd want to fetch from `reserveConfigurationHistoryItems`.
_RESERVE_QUERY = """
query Reserve($reserve: String!) {
  reserve(id: $reserve) {
    id
    symbol
    decimals
    optimalUsageRatio
    baseVariableBorrowRate
    variableRateSlope1
    variableRateSlope2
    reserveFactor
  }
}
"""


def _post(query: str, variables: dict[str, Any], retries: int = 5) -> dict:
    """POST to the Graph gateway with simple exponential backoff."""

    for attempt in range(retries):
        try:
            r = requests.post(
                config.SUBGRAPH_URL,
                json={"query": query, "variables": variables},
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
            if "errors" in payload:
                raise RuntimeError(f"subgraph error: {payload['errors']}")
            return payload["data"]
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            sleep_s = 2**attempt
            print(f"  retry {attempt + 1}/{retries} after {sleep_s}s: {exc}")
            time.sleep(sleep_s)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


def to_ray(x: str | int | float | None) -> float:
    """Convert a ray-scaled (1e27) string to a decimal rate."""

    return float(x) / 1e27 if x is not None else float("nan")


def to_token(x: str | int | float | None, decimals: int) -> float:
    """Convert a wei-scaled string to a token quantity."""

    return float(x) / (10**decimals) if x is not None else float("nan")


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


@dataclass
class ReserveParams:
    """Time-invariant rate-model parameters for one reserve."""

    symbol: str
    decimals: int
    opt_U: float
    r0: float
    slope1: float
    slope2: float
    reserve_factor: float


def fetch_reserve_params(reserve_id: str) -> ReserveParams:
    """Fetch the current rate-model parameters for one reserve."""

    data = _post(_RESERVE_QUERY, {"reserve": reserve_id})
    r = data["reserve"]
    if r is None:
        raise RuntimeError(
            f"Reserve {reserve_id} not found. Check SUBGRAPH_ID and reserve id format."
        )
    return ReserveParams(
        symbol=r["symbol"],
        decimals=int(r["decimals"]),
        opt_U=to_ray(r["optimalUsageRatio"]),
        r0=to_ray(r["baseVariableBorrowRate"]),
        slope1=to_ray(r["variableRateSlope1"]),
        slope2=to_ray(r["variableRateSlope2"]),
        # reserveFactor is stored as bps * 1e2 (so 1000 == 10%) in V3
        reserve_factor=float(r["reserveFactor"]) / 10_000.0,
    )


def fetch_reserve_history(reserve_id: str, decimals: int, history_days: int) -> pd.DataFrame:
    """Page through every `ReserveParamsHistoryItem` newer than `history_days` ago."""

    cutoff = int(time.time()) - history_days * 86_400
    last_ts = cutoff
    rows: list[dict] = []

    pbar = tqdm(desc=f"  reserve {reserve_id[:10]}…", unit="snap")
    while True:
        data = _post(
            _HISTORY_QUERY,
            {"reserve": reserve_id, "tsGt": last_ts},
        )
        batch = data.get("reserveParamsHistoryItems", [])
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1]["timestamp"])
        pbar.update(len(batch))
        if len(batch) < 1000:
            break
    pbar.close()

    if not rows:
        raise RuntimeError("subgraph returned 0 history items — check reserve id and date range")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    # --- unit conversion ------------------------------------------------
    df["utilizationRate"] = df["utilizationRate"].map(to_ray)
    df["liquidityRate"] = df["liquidityRate"].map(to_ray)
    df["variableBorrowRate"] = df["variableBorrowRate"].map(to_ray)
    df["stableBorrowRate"] = df["stableBorrowRate"].map(to_ray)
    for col in ("availableLiquidity", "totalLiquidity"):
        df[col] = df[col].map(lambda x: to_token(x, decimals))
    # Derive debt from the accounting identity: totalLiquidity = available + debt.
    # This sidesteps the per-debt-type field naming variance across subgraphs.
    df["debt"] = (df["totalLiquidity"] - df["availableLiquidity"]).clip(lower=0)
    # priceInUsd is stored *1e8 in AAVE V3 (Chainlink convention).
    df["priceInUsd"] = pd.to_numeric(df["priceInUsd"], errors="coerce") / 1e8

    return df


def resample_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Down-sample event-driven snapshots to a clean daily UTC grid."""

    out = pd.DataFrame(
        {
            "U": raw["utilizationRate"],
            "available_liquidity": raw["availableLiquidity"],
            "debt": raw["debt"],
            "liquidity_rate": raw["liquidityRate"],
            "borrow_rate": raw["variableBorrowRate"],
            "price_usd": raw["priceInUsd"],
        }
    )
    daily = out.resample("1D").last().ffill()
    return daily


def attach_params(daily: pd.DataFrame, params: ReserveParams) -> pd.DataFrame:
    """Attach the time-invariant rate-model knobs as broadcast columns."""

    daily = daily.copy()
    daily["opt_U"] = params.opt_U
    daily["r0"] = params.r0
    daily["slope1"] = params.slope1
    daily["slope2"] = params.slope2
    daily["reserve_factor"] = params.reserve_factor
    return daily


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def fetch_one(reserve_id: str, label: str, history_days: int = config.HISTORY_DAYS) -> pd.DataFrame:
    """Pull, resample, attach params, and write parquet for one reserve."""

    print(f"[{label}] params…")
    params = fetch_reserve_params(reserve_id)
    print(
        f"  symbol={params.symbol} decimals={params.decimals} "
        f"opt_U={params.opt_U:.3f} r0={params.r0:.4f} "
        f"slope1={params.slope1:.4f} slope2={params.slope2:.4f} rf={params.reserve_factor:.3f}"
    )

    print(f"[{label}] history (last {history_days}d)…")
    raw = fetch_reserve_history(reserve_id, params.decimals, history_days)
    raw_path = config.CACHE_DIR / f"{label}_reserve_raw.parquet"
    raw.to_parquet(raw_path)
    print(f"  wrote {len(raw):,} raw snapshots → {raw_path.name}")

    daily = attach_params(resample_daily(raw), params)
    out_path = config.CACHE_DIR / f"{label}_reserve.parquet"
    daily.to_parquet(out_path)
    print(f"  wrote {len(daily):,} daily rows → {out_path.name}")
    return daily


def main() -> None:
    fetch_one(config.WETH_RESERVE_ID, "weth")
    # USDC is used as a cross-asset feature (stablecoin leverage demand proxy)
    fetch_one(config.USDC_RESERVE_ID, "usdc")


if __name__ == "__main__":
    sys.exit(main())
