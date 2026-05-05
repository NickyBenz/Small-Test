"""Convenience runner: pull every data source in sequence."""

from __future__ import annotations

import sys

from data import fetch_funding, fetch_gas, fetch_ohlcv, fetch_options, fetch_subgraph


def main() -> None:
    fetch_subgraph.main()
    fetch_ohlcv.main()
    fetch_funding.main()
    fetch_options.main()
    fetch_gas.main()


if __name__ == "__main__":
    sys.exit(main())
