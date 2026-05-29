"""Rolling cumulative-response hedge ratio (β) at a chosen lead-lag delay.

Definition
----------
For a chosen lookahead K and rolling window W, the cumulative-response β at
time t is:

    β(t) = cov( ret_leader(s), Σ ret_follower(s+1 .. s+K) )    over s ∈ window
           ─────────────────────────────────────────────
                     var( ret_leader(s) )

Interpretation: "given a leader move of x today, the follower's cumulative
log return over the next K days is, on average, β·x." A β of 0.5 means the
follower captures half of the leader's move over the lookahead window.

The implementation is **look-ahead-safe**: β(t) only uses information that
was observable strictly before time t. (We compute the rolling cov/var on
windows ending at index s, then shift the result forward by K so that β at
the decision date never sees data from t+1 onward.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_cumulative_beta(
    df: pd.DataFrame,
    leader: str,
    follower: str,
    lookahead: int,
    window: int = 90,
) -> pd.DataFrame:
    """Return DataFrame: date | beta | n_obs (rolling sample size that was actually used)."""
    if lookahead < 1:
        raise ValueError("lookahead must be >= 1")
    if window < 10:
        raise ValueError("window must be >= 10")

    work = df.sort_values("date").reset_index(drop=True)
    ret_L = np.log(work[leader]).diff()
    ret_F = np.log(work[follower]).diff()

    # Cumulative follower log-return over [s+1, s+K], placed at index s.
    cum_F_forward = ret_F.rolling(lookahead).sum().shift(-lookahead)

    cov = ret_L.rolling(window).cov(cum_F_forward)
    var = ret_L.rolling(window).var()
    beta_at_s = cov / var

    # Shift forward by `lookahead` so β at time t uses only data through t-lookahead.
    beta = beta_at_s.shift(lookahead)

    # How many non-NaN obs went into the most recent fit at each t:
    n_obs = ret_L.rolling(window).count().shift(lookahead)

    return pd.DataFrame({
        "date": work["date"],
        "beta": beta,
        "n_obs": n_obs,
    })


def beta_at(beta_df: pd.DataFrame, when: pd.Timestamp) -> float:
    """Look up β as of a given date (returns NaN if unavailable)."""
    row = beta_df[beta_df["date"] == pd.Timestamp(when)]
    if row.empty:
        return float("nan")
    return float(row["beta"].iloc[0])


def summarize_beta(beta_df: pd.DataFrame) -> dict:
    """Summary stats of the rolling β time series."""
    b = beta_df["beta"].dropna()
    if b.empty:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "first_date": None, "last_date": None,
                "first_value": float("nan"), "last_value": float("nan")}
    valid = beta_df.dropna(subset=["beta"])
    return {
        "n": int(len(b)),
        "mean": float(b.mean()),
        "median": float(b.median()),
        "p10": float(b.quantile(0.10)),
        "p90": float(b.quantile(0.90)),
        "min": float(b.min()),
        "max": float(b.max()),
        "first_date": valid["date"].iloc[0],
        "last_date": valid["date"].iloc[-1],
        "first_value": float(valid["beta"].iloc[0]),
        "last_value": float(valid["beta"].iloc[-1]),
    }


def print_beta_summary(stats: dict, leader: str, follower: str,
                       lookahead: int, window: int) -> None:
    print(f"\n=== Rolling β  ({leader} → {follower}, lag={lookahead}d, window={window}d) ===")
    if stats["n"] == 0:
        print("  (insufficient data to fit β)")
        return
    print(f"  Observations:        {stats['n']:,}")
    print(f"  β  mean / median:    {stats['mean']:+.3f}  /  {stats['median']:+.3f}")
    print(f"  β  10th / 90th pct:  {stats['p10']:+.3f}  /  {stats['p90']:+.3f}")
    print(f"  β  range:            [{stats['min']:+.3f}, {stats['max']:+.3f}]")
    print(f"  Earliest β:          {stats['first_value']:+.3f}  ({stats['first_date'].date()})")
    print(f"  Latest β:            {stats['last_value']:+.3f}  ({stats['last_date'].date()})")
