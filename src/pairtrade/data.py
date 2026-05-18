"""CoinGecko daily history → parquet."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

COINGECKO = "https://api.coingecko.com/api/v3"
RATE_LIMIT_SLEEP = 2.5  # free tier ~30 calls/min


def fetch_coin(coin_id: str, days: str | int = "max", vs: str = "usd") -> pd.DataFrame:
    """Fetch one coin's daily history. Returns date | coin | price | market_cap | volume."""
    params = {"vs_currency": vs, "days": str(days)}
    if days == "max" or int(days) > 90:
        params["interval"] = "daily"

    r = requests.get(
        f"{COINGECKO}/coins/{coin_id}/market_chart",
        params=params,
        timeout=30,
        headers={"User-Agent": "pairtrade/1.0"},
    )
    r.raise_for_status()
    payload = r.json()

    prices = pd.DataFrame(payload["prices"], columns=["ts", "price"])
    mcaps = pd.DataFrame(payload["market_caps"], columns=["ts", "market_cap"])
    vols = pd.DataFrame(payload["total_volumes"], columns=["ts", "volume"])

    df = prices.merge(mcaps, on="ts").merge(vols, on="ts")
    df["date"] = (
        pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    )
    df["coin"] = coin_id
    return (
        df[["date", "coin", "price", "market_cap", "volume"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def fetch_many(coins: list[str], days: str | int = "max", out_dir: str | Path = "data") -> dict[str, Path]:
    """Fetch a list of coins, writing one parquet per coin. Returns {coin: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for i, coin in enumerate(coins):
        if i > 0:
            time.sleep(RATE_LIMIT_SLEEP)
        df = fetch_coin(coin, days=days)
        path = out / f"{coin}_history.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        written[coin] = path
        print(f"  {coin:<12} {len(df):>5} rows  {df['date'].min().date()} → {df['date'].max().date()}")
    return written


def load_coin(coin: str, data_dir: str | Path = "data") -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / f"{coin}_history.parquet")


def load_pair(a: str, b: str, data_dir: str | Path = "data") -> pd.DataFrame:
    """Align two coins on date, returning date | <a> | <b>."""
    da = load_coin(a, data_dir)[["date", "price"]].rename(columns={"price": a})
    db = load_coin(b, data_dir)[["date", "price"]].rename(columns={"price": b})
    return da.merge(db, on="date", how="inner").sort_values("date").reset_index(drop=True)
