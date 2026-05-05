"""Historical Ethereum gas price for cost-of-trade modelling.

Two paths:

1. If `ETHERSCAN_API_KEY` is set, hit `gastracker dailyavggasprice` (the public
   endpoint) and cache the resulting daily mean gas price in gwei.
2. Otherwise, fall back to a constant of 25 gwei applied across the window.
   This is intentionally a poor man's fallback — for a real backtest, plug in
   a proper gas oracle. Gas drag at $1M notional with ~150k gas per tx is
   small (a few dollars) so the choice rarely flips PnL conclusions.

Output: `eth_gas.parquet` with a single `gas_gwei` column on a daily UTC grid.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config


def fetch_gas_etherscan(history_days: int) -> pd.DataFrame:
    """Hit Etherscan's gas tracker endpoint. Requires API key."""

    end = datetime.now(tz=timezone.utc).date()
    start = end - timedelta(days=history_days)

    r = requests.get(
        "https://api.etherscan.io/api",
        params={
            "module": "gastracker",
            "action": "dailyavggasprice",
            "startdate": start.isoformat(),
            "enddate": end.isoformat(),
            "sort": "asc",
            "apikey": config.ETHERSCAN_API_KEY,
        },
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "1":
        raise RuntimeError(f"etherscan: {payload.get('message')} | {payload.get('result')}")

    df = pd.DataFrame(payload["result"])
    df["ts"] = pd.to_datetime(df["UTCDate"], utc=True)
    # `avgGasPrice_Wei` -> gwei
    df["gas_gwei"] = pd.to_numeric(df["avgGasPrice_Wei"]) / 1e9
    return df.set_index("ts")[["gas_gwei"]].sort_index()


def fetch_gas_constant_fallback(history_days: int, gwei: float = 25.0) -> pd.DataFrame:
    end = datetime.now(tz=timezone.utc).normalize()
    idx = pd.date_range(end - timedelta(days=history_days), end, freq="1D", tz="UTC")
    return pd.DataFrame({"gas_gwei": gwei}, index=idx)


def main() -> None:
    if config.ETHERSCAN_API_KEY:
        print("[gas] etherscan dailyavggasprice…")
        try:
            df = fetch_gas_etherscan(config.HISTORY_DAYS)
        except Exception as exc:  # noqa: BLE001
            print(f"  etherscan failed ({exc}); falling back to constant 25 gwei")
            df = fetch_gas_constant_fallback(config.HISTORY_DAYS)
    else:
        print("[gas] no ETHERSCAN_API_KEY; using constant 25 gwei fallback")
        df = fetch_gas_constant_fallback(config.HISTORY_DAYS)

    out = config.CACHE_DIR / "eth_gas.parquet"
    df.to_parquet(out)
    print(f"  wrote {len(df):,} rows → {out.name}")


if __name__ == "__main__":
    sys.exit(main())
