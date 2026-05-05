"""Historical perpetual funding rates for ETH/USDT from Binance via ccxt.

Funding is paid every 8h on Binance. We resample to daily by *summing* all
fundings within the UTC day so the feature represents the day's total cost of
leverage. (Mean would be biased low since some days have 2 prints, some 3.)
"""

from __future__ import annotations

import sys
import time

import ccxt
import pandas as pd

import config


def fetch_funding(
    symbol: str = config.BINANCE_SYMBOL_PERP,
    history_days: int = config.HISTORY_DAYS,
) -> pd.DataFrame:
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    ex.load_markets()

    since = ex.milliseconds() - history_days * 86_400_000
    rows: list[dict] = []
    cursor = since
    while True:
        batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["timestamp"] + 1
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000.0)

    df = pd.DataFrame(
        [
            {"ts": r["timestamp"], "funding_rate": float(r["fundingRate"])}
            for r in rows
            if r.get("fundingRate") is not None
        ]
    )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts").set_index("ts").sort_index()

    daily = pd.DataFrame(
        {
            "funding_sum_24h": df["funding_rate"].resample("1D").sum(),
            "funding_count_24h": df["funding_rate"].resample("1D").count(),
        }
    )
    return daily


def main() -> None:
    print(f"[funding] {config.BINANCE_SYMBOL_PERP} 8h → daily…")
    df = fetch_funding()
    out = config.CACHE_DIR / "eth_funding.parquet"
    df.to_parquet(out)
    print(f"  wrote {len(df):,} rows → {out.name}")


if __name__ == "__main__":
    sys.exit(main())
