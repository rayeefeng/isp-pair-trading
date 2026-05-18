"""Yahoo Finance daily history → parquet.

Tickers use Yahoo's crypto convention: BTC-USD, ETH-USD, SOL-USD, etc.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_coin(ticker: str, days: str | int = "max") -> pd.DataFrame:
    """Fetch one ticker's daily history. Returns date | coin | price | volume."""
    kwargs = {"interval": "1d", "auto_adjust": True, "progress": False}
    if days == "max":
        kwargs["period"] = "max"
    else:
        kwargs["start"] = (datetime.utcnow() - timedelta(days=int(days))).date().isoformat()

    raw = yf.download(ticker, **kwargs)
    if raw.empty:
        raise RuntimeError(f"No data returned for ticker {ticker!r} — check the symbol.")

    # yfinance returns a multiindex when given a list; flatten just in case
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()[["Date", "Close", "Volume"]].rename(
        columns={"Date": "date", "Close": "price", "Volume": "volume"}
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["coin"] = ticker
    return (
        df[["date", "coin", "price", "volume"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_many(tickers: list[str], days: str | int = "max", out_dir: str | Path = "data") -> dict[str, Path]:
    """Fetch a list of tickers, writing one parquet per ticker. Returns {ticker: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for ticker in tickers:
        df = fetch_coin(ticker, days=days)
        path = out / f"{ticker}_history.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        written[ticker] = path
        print(f"  {ticker:<10} {len(df):>5} rows  {df['date'].min().date()} → {df['date'].max().date()}")
    return written


def load_coin(ticker: str, data_dir: str | Path = "data") -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{ticker}_history.parquet")


def load_pair(a: str, b: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """Align two tickers on date, returning date | <a> | <b>."""
    da = load_coin(a, data_dir)[["date", "price"]].rename(columns={"price": a})
    db = load_coin(b, data_dir)[["date", "price"]].rename(columns={"price": b})
    return da.merge(db, on="date", how="inner").sort_values("date").reset_index(drop=True)
