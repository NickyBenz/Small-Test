"""Project-wide configuration constants.

Centralizing these here means the rest of the code never hardcodes addresses,
window sizes, or rate-model knobs. Update here when the target reserve, chain,
or strategy parameters change.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# AAVE V3 — Ethereum mainnet
# ---------------------------------------------------------------------------
# Graph Network gateway. The subgraph id below is AAVE's official V3 Ethereum
# subgraph. Override SUBGRAPH_ID in .env if you want to point at e.g. Messari's
# standardized subgraph (id JCNWRypm7FYwV8fx5HhzZPSFaMxgkPuw4TnR3Gpi81zk).
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY", "")
SUBGRAPH_ID = os.getenv(
    "SUBGRAPH_ID",
    "Cd2gEDVeqnjBn1hSeqFMitw8Q1iiyV9FYUZkLNRcL87g",
)
SUBGRAPH_URL = (
    f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/{SUBGRAPH_ID}"
    if GRAPH_API_KEY
    else f"https://api.thegraph.com/subgraphs/id/{SUBGRAPH_ID}"
)

# Underlying token addresses (lowercase) on Ethereum mainnet
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

# AAVE V3 PoolAddressesProvider on Ethereum (lowercase). The reserve entity id
# in AAVE's subgraph schema is `<underlying><poolAddressesProvider>`.
POOL_ADDRESSES_PROVIDER = "0x2f39d218133afab8f2b819b1066c7e434ad94e9e"

WETH_RESERVE_ID = WETH + POOL_ADDRESSES_PROVIDER
USDC_RESERVE_ID = USDC + POOL_ADDRESSES_PROVIDER

# Token decimals
WETH_DECIMALS = 18
USDC_DECIMALS = 6

# ---------------------------------------------------------------------------
# Backtest window
# ---------------------------------------------------------------------------
HISTORY_DAYS = 540  # ~18 months — long enough for walk-forward, short enough to fetch fast

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
LABEL_HORIZON_DAYS = 7
ROLL_WINDOW = 30
TRAIN_WINDOW_DAYS = 365
RETRAIN_EVERY_DAYS = 30
EMBARGO_DAYS = LABEL_HORIZON_DAYS  # purge to prevent label leakage

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
P_ENTER = 0.60
P_EXIT = 0.40
MIN_HOLD_DAYS = 3
NOTIONAL_USD = 1_000_000.0
GAS_UNITS_PER_TX = 150_000  # supply or withdraw

# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
DERIBIT_BASE = "https://www.deribit.com/api/v2"
BINANCE_SYMBOL_SPOT = "ETH/USDT"
BINANCE_SYMBOL_PERP = "ETH/USDT:USDT"
