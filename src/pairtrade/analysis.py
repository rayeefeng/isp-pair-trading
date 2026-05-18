"""Compare two coins: total return, correlation, plain-English trend summary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TrendSummary:
    a: str
    b: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_days: int
    start_price_a: float
    end_price_a: float
    return_a: float            # fractional, e.g. 0.42 = +42%
    start_price_b: float
    end_price_b: float
    return_b: float
    correlation: float         # of daily log returns

    @property
    def direction_a(self) -> str:
        return "up" if self.return_a >= 0 else "down"

    @property
    def direction_b(self) -> str:
        return "up" if self.return_b >= 0 else "down"

    @property
    def trend_verdict(self) -> str:
        if self.direction_a == self.direction_b:
            base = f"both trended {self.direction_a}"
        else:
            base = f"diverged: {self.a} {self.direction_a}, {self.b} {self.direction_b}"

        c = self.correlation
        if c >= 0.7:
            link = "moved tightly together"
        elif c >= 0.3:
            link = "moderately linked"
        elif c > -0.3:
            link = "largely independent day-to-day"
        else:
            link = "moved opposite each other day-to-day"
        return f"{base}; {link} (corr={c:.2f})"


def compare(df: pd.DataFrame, a: str, b: str) -> TrendSummary:
    """`df` is the output of data.load_pair(a, b): columns date | <a> | <b>."""
    df = df.sort_values("date").reset_index(drop=True)
    log_a = np.log(df[a])
    log_b = np.log(df[b])
    ret_a = log_a.diff()
    ret_b = log_b.diff()
    corr = float(ret_a.corr(ret_b))

    start_a, end_a = float(df[a].iloc[0]), float(df[a].iloc[-1])
    start_b, end_b = float(df[b].iloc[0]), float(df[b].iloc[-1])

    return TrendSummary(
        a=a,
        b=b,
        start_date=df["date"].iloc[0],
        end_date=df["date"].iloc[-1],
        n_days=len(df),
        start_price_a=start_a,
        end_price_a=end_a,
        return_a=end_a / start_a - 1,
        start_price_b=start_b,
        end_price_b=end_b,
        return_b=end_b / start_b - 1,
        correlation=corr,
    )


def print_summary(s: TrendSummary) -> None:
    print(f"\nPeriod: {s.start_date.date()} → {s.end_date.date()}  ({s.n_days} daily observations)")
    print(f"  {s.a:<12} ${s.start_price_a:>12,.2f}  →  ${s.end_price_a:>12,.2f}   ({s.return_a:+.1%})")
    print(f"  {s.b:<12} ${s.start_price_b:>12,.2f}  →  ${s.end_price_b:>12,.2f}   ({s.return_b:+.1%})")
    print(f"  Correlation of daily returns: {s.correlation:+.3f}")
    print(f"\nVerdict: {s.trend_verdict}")
