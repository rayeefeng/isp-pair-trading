"""Lead-lag analysis: when one coin moves big, does the other follow?

Pipeline:
    1. daily_return_stats()    — distribution per coin (what's a normal move?)
    2. cross_correlation()     — corr(ret_a_t, ret_b_{t+k}) at each lag
    3. parameter_sweep()       — for many (threshold, lookahead, leader) combos,
                                 measure hit rate and average follow-through
    4. recommend_params()      — pick the best combination by signal strength
    5. find_events()           — list every event at the chosen parameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


# ---------- helpers ----------

def _log_returns(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Return a DataFrame with columns: date, ret_a, ret_b (log returns)."""
    out = pd.DataFrame({"date": df["date"].values})
    out["ret_a"] = np.log(df[a].values)
    out["ret_b"] = np.log(df[b].values)
    out["ret_a"] = out["ret_a"].diff()
    out["ret_b"] = out["ret_b"].diff()
    return out


# ---------- 1. return distribution ----------

def daily_return_stats(df: pd.DataFrame, coin: str) -> dict:
    """Percentiles of |daily log return| for one coin."""
    ret = np.log(df[coin]).diff().abs().dropna()
    return {
        "coin": coin,
        "n_days": int(len(ret)),
        "median": float(ret.median()),
        "p75": float(ret.quantile(0.75)),
        "p90": float(ret.quantile(0.90)),
        "p95": float(ret.quantile(0.95)),
        "p99": float(ret.quantile(0.99)),
        "max": float(ret.max()),
    }


# ---------- 2. cross-correlation ----------

def cross_correlation(df: pd.DataFrame, a: str, b: str, max_lag: int = 10) -> pd.DataFrame:
    """For each lag k in [-max_lag, +max_lag], compute corr(ret_a_t, ret_b_{t+k}).

    Positive k means: b's return at time t+k correlates with a's return at t
        → a leads b by k days.
    """
    r = _log_returns(df, a, b).dropna()
    ra, rb = r["ret_a"], r["ret_b"]
    rows = []
    for k in range(-max_lag, max_lag + 1):
        if k >= 0:
            shifted = rb.shift(-k)
            valid = ~(ra.isna() | shifted.isna())
            corr = ra[valid].corr(shifted[valid])
        else:
            shifted = ra.shift(k)  # negative k
            valid = ~(shifted.isna() | rb.isna())
            corr = shifted[valid].corr(rb[valid])
        rows.append({"lag": k, "correlation": float(corr)})
    return pd.DataFrame(rows)


def lead_lag_verdict(ccf: pd.DataFrame, a: str, b: str) -> str:
    """One-line verdict from a cross-correlation table."""
    best = ccf.iloc[ccf["correlation"].abs().idxmax()]
    lag, corr = int(best["lag"]), float(best["correlation"])
    if lag == 0:
        return f"Peak correlation is at lag 0 ({corr:+.3f}) — neither coin appears to lead."
    leader, follower = (a, b) if lag > 0 else (b, a)
    return (f"Peak correlation is at lag {lag:+d} ({corr:+.3f}) — "
            f"{leader} appears to lead {follower} by {abs(lag)} day(s).")


# ---------- 3. parameter sweep ----------

def _sweep_one(
    r: pd.DataFrame, leader_col: str, follower_col: str,
    threshold: float, lookahead: int
) -> dict:
    """For one (leader, threshold, lookahead): measure follow-through stats."""
    leader = r[leader_col]
    follower = r[follower_col]
    # Cumulative follower log-return over the next `lookahead` days.
    # rolling(K).sum().shift(-K) at index t equals follower[t+1] + ... + follower[t+K]
    fwd_cum = follower.rolling(lookahead).sum().shift(-lookahead)

    event_mask = leader.abs() >= threshold
    n_events = int(event_mask.sum())
    if n_events == 0:
        return dict(n_events=0, hit_rate=np.nan, mean_follow=np.nan,
                    sd_follow=np.nan, t_stat=np.nan, score=-np.inf)

    sign = np.sign(leader[event_mask])
    same_dir = (sign * fwd_cum[event_mask]).dropna()
    if same_dir.empty:
        return dict(n_events=n_events, hit_rate=np.nan, mean_follow=np.nan,
                    sd_follow=np.nan, t_stat=np.nan, score=-np.inf)

    mean = float(same_dir.mean())
    sd = float(same_dir.std(ddof=1)) if len(same_dir) > 1 else np.nan
    hit_rate = float((same_dir > 0).mean())
    t_stat = mean / (sd / np.sqrt(len(same_dir))) if sd and sd > 0 else np.nan
    return dict(
        n_events=int(len(same_dir)),
        hit_rate=hit_rate,
        mean_follow=mean,           # mean same-direction log return over K days
        sd_follow=sd,
        t_stat=float(t_stat) if not np.isnan(t_stat) else np.nan,
        # Score balances effect size and sample size; higher = stronger signal.
        score=float(mean * np.sqrt(len(same_dir))) if not np.isnan(mean) else -np.inf,
    )


def parameter_sweep(
    df: pd.DataFrame, a: str, b: str,
    thresholds: Iterable[float] = (0.02, 0.03, 0.05, 0.07, 0.10),
    lookaheads: Iterable[int] = (1, 2, 3, 5, 7),
) -> pd.DataFrame:
    """Test every (direction, threshold, lookahead) combo and return stats."""
    r = _log_returns(df, a, b).dropna().reset_index(drop=True)
    rows = []
    for leader, follower, name_l, name_f in [
        ("ret_a", "ret_b", a, b),
        ("ret_b", "ret_a", b, a),
    ]:
        for T in thresholds:
            for K in lookaheads:
                stats = _sweep_one(r, leader, follower, T, K)
                rows.append({
                    "leader": name_l, "follower": name_f,
                    "threshold": T, "lookahead": K, **stats,
                })
    return pd.DataFrame(rows)


def recommend_params(sweep: pd.DataFrame, min_events: int = 10) -> pd.Series | None:
    """Pick the (direction, threshold, lookahead) with the strongest signal,
    requiring at least `min_events` events for statistical credibility."""
    eligible = sweep[(sweep["n_events"] >= min_events) & sweep["score"].notna()]
    if eligible.empty:
        return None
    return eligible.sort_values("score", ascending=False).iloc[0]


# ---------- 4. find events ----------

@dataclass
class EventRow:
    event_date: pd.Timestamp
    leader_return: float
    follower_best_date: pd.Timestamp
    follower_best_return: float
    delay_days: int
    follower_cum_return: float
    same_direction: bool


def find_events(
    df: pd.DataFrame, leader: str, follower: str,
    threshold: float, lookahead: int,
) -> pd.DataFrame:
    """List every event: date where |ret_leader| ≥ threshold, with the
    follower's biggest same-direction single-day move within `lookahead` days
    and its cumulative same-direction return over the window."""
    r = _log_returns(df, leader, follower).dropna().reset_index(drop=True)
    ret_l = r["ret_a"].values
    ret_f = r["ret_b"].values
    dates = r["date"].values
    n = len(r)

    rows = []
    for t in range(n):
        if abs(ret_l[t]) < threshold:
            continue
        s = 1 if ret_l[t] >= 0 else -1
        window_end = min(t + lookahead, n - 1)
        if t + 1 > window_end:
            continue
        window = ret_f[t + 1 : window_end + 1]
        if window.size == 0:
            continue
        signed = s * window
        best_idx = int(np.argmax(signed))
        best_signed = float(signed[best_idx])
        cum_signed = float(signed.sum())
        rows.append({
            "event_date": pd.Timestamp(dates[t]),
            "leader_return": float(ret_l[t]),
            "follower_best_date": pd.Timestamp(dates[t + 1 + best_idx]),
            "follower_best_return": float(s * best_signed),
            "delay_days": best_idx + 1,
            "follower_cum_return": float(s * cum_signed),
            "same_direction": best_signed > 0,
        })
    return pd.DataFrame(rows)


# ---------- 5. printable summaries ----------

def print_return_distribution(stats_list: list[dict]) -> None:
    print("\n=== Daily return distribution (|log return|) ===")
    print(f"  {'coin':<10} {'days':>6} {'median':>8} {'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    for s in stats_list:
        print(f"  {s['coin']:<10} {s['n_days']:>6} "
              f"{s['median']:>7.2%} {s['p75']:>7.2%} {s['p90']:>7.2%} "
              f"{s['p95']:>7.2%} {s['p99']:>7.2%} {s['max']:>7.2%}")


def print_sweep_table(sweep: pd.DataFrame, top: int = 8) -> None:
    show = (sweep.dropna(subset=["score"])
                 .sort_values("score", ascending=False)
                 .head(top)
                 .copy())
    if show.empty:
        print("\n(no eligible parameter combinations)")
        return
    show["threshold"] = (show["threshold"] * 100).map(lambda x: f"{x:.1f}%")
    show["hit_rate"] = show["hit_rate"].map(lambda x: f"{x:.0%}")
    show["mean_follow"] = show["mean_follow"].map(lambda x: f"{x*100:+.2f}%")
    show["t_stat"] = show["t_stat"].map(lambda x: f"{x:+.2f}")
    show["score"] = show["score"].map(lambda x: f"{x:.4f}")
    print("\n=== Top parameter combinations by signal strength ===")
    print(show[["leader", "follower", "threshold", "lookahead",
                "n_events", "hit_rate", "mean_follow", "t_stat", "score"]]
              .to_string(index=False))


def print_recommendation(rec: pd.Series, sweep: pd.DataFrame) -> None:
    if rec is None:
        print("\nNo parameter combination produced enough events to recommend.")
        return
    print("\n=== Recommended parameters ===")
    print(f"  Direction:       {rec['leader']} → {rec['follower']}")
    print(f"  Threshold:       {rec['threshold']*100:.1f}% daily move in the leader")
    print(f"  Lookahead:       {int(rec['lookahead'])} day(s)")
    print(f"  Events:          {int(rec['n_events'])}")
    print(f"  Hit rate:        {rec['hit_rate']:.0%}   (follower moved same direction)")
    print(f"  Mean follow:     {rec['mean_follow']*100:+.2f}% cumulative log return")
    print(f"  t-statistic:     {rec['t_stat']:+.2f}   (>2 ≈ statistically meaningful)")


def print_events(events: pd.DataFrame, leader: str, follower: str,
                 threshold: float, lookahead: int, show: int = 12) -> None:
    if events.empty:
        print("\nNo events found at the chosen parameters.")
        return
    n_total = len(events)
    n_same = int(events["same_direction"].sum())
    avg_delay = events.loc[events["same_direction"], "delay_days"].mean() if n_same else np.nan
    avg_follow = events.loc[events["same_direction"], "follower_best_return"].mean() if n_same else np.nan
    avg_leader_abs = events["leader_return"].abs().mean()

    print(f"\n=== Events: {leader} moves ≥ {threshold:.0%}, "
          f"look {lookahead} day(s) ahead in {follower} ===")
    print(f"  Total events:              {n_total}")
    print(f"  Same-direction follow:     {n_same}  ({n_same/n_total:.0%})")
    if n_same:
        print(f"  Avg delay (same-dir):      {avg_delay:.1f} day(s)")
        print(f"  Avg leader move size:      {avg_leader_abs*100:+.1f}% (absolute)")
        print(f"  Avg follower move size:    {avg_follow*100:+.1f}% (same direction, peak day)")
    print(f"\n  Most recent {min(show, n_total)} events:")
    recent = events.sort_values("event_date", ascending=False).head(show)
    for r in recent.itertuples():
        arrow = "→" if r.same_direction else "✗"
        print(f"    {r.event_date.date()}  {leader} {r.leader_return*100:+6.2f}%   "
              f"{arrow} {follower} {r.follower_best_return*100:+6.2f}% "
              f"on {r.follower_best_date.date()}  ({r.delay_days}d delay)")
