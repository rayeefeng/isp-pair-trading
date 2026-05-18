"""Fetch daily Yahoo Finance history for one or more crypto tickers → data/*.parquet.

Usage:
    python scripts/fetch.py
    python scripts/fetch.py --coins BTC-USD ETH-USD SOL-USD
    python scripts/fetch.py --days 730
"""

import argparse

import _bootstrap  # noqa: F401
from pairtrade.data import fetch_many


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coins", nargs="+", default=["BTC-USD", "ETH-USD"],
                   help="Yahoo Finance crypto tickers (e.g. BTC-USD ETH-USD SOL-USD)")
    p.add_argument("--days", default="max", help='"max" or an integer N')
    p.add_argument("--out", default="data")
    args = p.parse_args()

    print(f"Fetching {len(args.coins)} ticker(s) (days={args.days})...")
    fetch_many(args.coins, days=args.days, out_dir=args.out)


if __name__ == "__main__":
    main()
