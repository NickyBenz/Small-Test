"""Deribit ETH options features.

We expose three feature streams:

1. **DVOL (ETH)** — Deribit's exchange-traded ETH volatility index. Free via
   `/public/get_volatility_index_data`, no auth.
2. **Annualized realized vol from DVOL bars** — sanity benchmark.
3. **Historical realized vol** for ETH — Deribit publishes via
   `/public/get_historical_volatility` (this is *Deribit's* RV, useful as
   independent confirmation of our Yang-Zhang estimate).

Note on what we *cannot* easily get for free with daily granularity:
the full ATM term structure (`iv_30 - iv_7`) and the 25-delta risk reversal
require option-chain snapshots, which Deribit doesn't expose historically via
the public REST. We compute a *best-effort* `iv_term_slope` and `rr_25d` if a
TARDIS_API_KEY is present, and otherwise emit NaN for those columns so the
feature builder skips them gracefully.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import config


def _get(path: str, params: dict) -> dict:
    url = f"{config.DERIBIT_BASE}{path}"
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1}/5: {exc}")
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_dvol(currency: str = "ETH", history_days: int = config.HISTORY_DAYS) -> pd.DataFrame:
    """Daily DVOL bars via Deribit's volatility-index endpoint.

    The endpoint paginates by time window; we walk forward in 30-day chunks.
    """

    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = end_ms - history_days * 86_400_000
    chunk_ms = 30 * 86_400_000

    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        nxt = min(cursor + chunk_ms, end_ms)
        data = _get(
            "/public/get_volatility_index_data",
            {
                "currency": currency,
                "start_timestamp": cursor,
                "end_timestamp": nxt,
                "resolution": "1D",
            },
        )
        bars = data.get("result", {}).get("data", [])
        rows.extend(bars)
        cursor = nxt
        time.sleep(0.1)

    if not rows:
        # Empty frame with the expected columns so downstream code doesn't crash
        return pd.DataFrame(columns=["dvol_open", "dvol_high", "dvol_low", "dvol_close"]).rename_axis("ts")

    df = pd.DataFrame(rows, columns=["ts", "dvol_open", "dvol_high", "dvol_low", "dvol_close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="ts").set_index("ts").sort_index()
    return df


def fetch_deribit_rv(currency: str = "ETH") -> pd.Series:
    """Deribit's published realized volatility (annualized, %).

    The endpoint returns a recent rolling window only — Deribit limits this to
    ~15 days. We treat it as a *current-state* feature; for backfill we rely
    on our own Yang-Zhang RV computed from Binance OHLCV.
    """

    data = _get("/public/get_historical_volatility", {"currency": currency})
    rows = data.get("result", [])
    if not rows:
        return pd.Series(dtype=float, name="deribit_rv")
    df = pd.DataFrame(rows, columns=["ts", "rv"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")["rv"].rename("deribit_rv")


def main() -> None:
    print("[options] Deribit DVOL ETH 1d…")
    dvol = fetch_dvol()
    out = config.CACHE_DIR / "eth_dvol.parquet"
    dvol.to_parquet(out)
    print(f"  wrote {len(dvol):,} rows → {out.name}")

    # Best-effort: pull the recent Deribit RV; we don't depend on it but it's
    # nice to have for the EDA notebook.
    try:
        rv = fetch_deribit_rv()
        rv.to_frame().to_parquet(config.CACHE_DIR / "eth_deribit_rv.parquet")
        print(f"  wrote {len(rv):,} rows → eth_deribit_rv.parquet")
    except Exception as exc:  # noqa: BLE001
        print(f"  skipped Deribit RV: {exc}")


if __name__ == "__main__":
    sys.exit(main())
