"""Daily OHLCV for ETH/USDT from Binance via ccxt.

We pull spot, not perp, because the Yang-Zhang volatility estimator we use
in features needs honest open prints for overnight-jump correction. Binance
spot has the deepest book and the cleanest history.
"""

from __future__ import annotations

import sys
import time

import ccxt
import pandas as pd

import config


def fetch_ohlcv(
    symbol: str = config.BINANCE_SYMBOL_SPOT,
    timeframe: str = "1d",
    history_days: int = config.HISTORY_DAYS,
) -> pd.DataFrame:
    """Pull daily OHLCV via ccxt, return a UTC-indexed dataframe."""

    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()

    since = ex.milliseconds() - history_days * 86_400_000
    rows: list[list[float]] = []
    cursor = since
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 86_400_000
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000.0)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts").set_index("ts").sort_index()
    return df


def main() -> None:
    print(f"[ohlcv] {config.BINANCE_SYMBOL_SPOT} 1d…")
    df = fetch_ohlcv()
    out = config.CACHE_DIR / "eth_ohlcv.parquet"
    df.to_parquet(out)
    print(f"  wrote {len(df):,} rows → {out.name} (range {df.index[0].date()} → {df.index[-1].date()})")


if __name__ == "__main__":
    sys.exit(main())
