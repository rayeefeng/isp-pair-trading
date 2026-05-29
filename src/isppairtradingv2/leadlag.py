"""Hedge-ratio-aware event analysis.

Re-uses v1's parameter sweep (in `pairtrade.leadlag`) and adds a per-event
`predicted_follower_move` (using rolling β at the decision date) and a
`capture_ratio = actual / predicted`. Capture > 1 means the follower
over-shot; < 1 means it under-shot; < 0 means it moved the wrong way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hedge_ratio import rolling_cumulative_beta


def find_events_with_capture(
    df: pd.DataFrame,
    leader: str,
    follower: str,
    threshold: float,
    lookahead: int,
    beta_window: int = 90,
) -> pd.DataFrame:
    """List every event with β-based prediction and capture ratio.

    Columns:
        event_date, leader_return, beta_at_event,
        predicted_follower_cum, actual_follower_cum, capture_ratio,
        same_direction
    """
    work = df.sort_values("date").reset_index(drop=True)
    ret_L = np.log(work[leader]).diff().values
    ret_F = np.log(work[follower]).diff().values
    dates = work["date"].values
    n = len(work)

    beta_df = rolling_cumulative_beta(work, leader, follower, lookahead, beta_window)
    beta = beta_df["beta"].values

    rows = []
    for t in range(n):
        if not np.isfinite(ret_L[t]) or abs(ret_L[t]) < threshold:
            continue
        if not np.isfinite(beta[t]):
            continue
        end = t + lookahead
        if end >= n:
            continue

        # actual cumulative follower return over [t+1, t+lookahead]
        window_returns = ret_F[t + 1 : end + 1]
        if window_returns.size == 0 or not np.all(np.isfinite(window_returns)):
            continue
        actual = float(window_returns.sum())
        predicted = float(beta[t] * ret_L[t])
        # capture: actual / predicted (sign-aware). Guard against tiny predicted.
        if abs(predicted) < 1e-9:
            capture = float("nan")
        else:
            capture = actual / predicted

        rows.append({
            "event_date": pd.Timestamp(dates[t]),
            "leader_return": float(ret_L[t]),
            "beta_at_event": float(beta[t]),
            "predicted_follower_cum": predicted,
            "actual_follower_cum": actual,
            "capture_ratio": float(capture) if np.isfinite(capture) else float("nan"),
            # "Same direction" uses sign of actual vs leader (not vs predicted), so it
            # still measures what the v1 hit-rate metric measured.
            "same_direction": (np.sign(ret_L[t]) == np.sign(actual)) and actual != 0,
        })

    return pd.DataFrame(rows)


def summarize_capture(events: pd.DataFrame) -> dict:
    if events.empty:
        return {"n_events": 0}
    cap = events["capture_ratio"].dropna()
    n_total = len(events)
    n_same = int(events["same_direction"].sum())
    return {
        "n_events": n_total,
        "n_same_direction": n_same,
        "hit_rate": n_same / n_total,
        "capture_mean": float(cap.mean()) if not cap.empty else float("nan"),
        "capture_median": float(cap.median()) if not cap.empty else float("nan"),
        "capture_p25": float(cap.quantile(0.25)) if not cap.empty else float("nan"),
        "capture_p75": float(cap.quantile(0.75)) if not cap.empty else float("nan"),
        "frac_over_predicted": float((cap > 1).mean()) if not cap.empty else float("nan"),
        "frac_under_predicted_positive": float(((cap >= 0) & (cap < 1)).mean()) if not cap.empty else float("nan"),
        "frac_wrong_direction": float((cap < 0).mean()) if not cap.empty else float("nan"),
        "mean_leader_move_abs": float(events["leader_return"].abs().mean()),
        "mean_actual_follow": float(events["actual_follower_cum"].mean()),
        "mean_predicted_follow": float(events["predicted_follower_cum"].mean()),
    }


def print_capture_summary(stats: dict, leader: str, follower: str,
                          threshold: float, lookahead: int) -> None:
    if stats.get("n_events", 0) == 0:
        print("\nNo events found at the chosen parameters.")
        return
    print(f"\n=== Capture analysis ({leader} → {follower}, "
          f"threshold={threshold:.0%}, lookahead={lookahead}d) ===")
    print(f"  Total events:                     {stats['n_events']}")
    print(f"  Same-direction follow-through:    {stats['n_same_direction']}  "
          f"({stats['hit_rate']:.0%})")
    print(f"  Mean leader move size (|·|):      {stats['mean_leader_move_abs']*100:.2f}%")
    print(f"  Mean predicted follower cum:      {stats['mean_predicted_follow']*100:+.2f}%")
    print(f"  Mean actual follower cum:         {stats['mean_actual_follow']*100:+.2f}%")
    print(f"  Capture ratio  mean / median:     {stats['capture_mean']:+.2f}  /  {stats['capture_median']:+.2f}")
    print(f"  Capture ratio  25th / 75th pct:   {stats['capture_p25']:+.2f}  /  {stats['capture_p75']:+.2f}")
    print(f"  Events that over-shot β:          {stats['frac_over_predicted']:.0%}")
    print(f"  Events that under-shot (still right direction): {stats['frac_under_predicted_positive']:.0%}")
    print(f"  Events that moved wrong direction: {stats['frac_wrong_direction']:.0%}")


def print_events_with_capture(events: pd.DataFrame, leader: str, follower: str,
                              show: int = 12) -> None:
    if events.empty:
        return
    recent = events.sort_values("event_date", ascending=False).head(show)
    print(f"\n  Most recent {len(recent)} events  "
          f"(predicted = β(t) × leader move, capture = actual / predicted):")
    for r in recent.itertuples():
        cap_str = f"{r.capture_ratio:+.2f}x" if np.isfinite(r.capture_ratio) else "  n/a"
        print(f"    {r.event_date.date()}  "
              f"{leader} {r.leader_return*100:+6.2f}%  "
              f"β={r.beta_at_event:+.2f}  "
              f"pred {r.predicted_follower_cum*100:+6.2f}%  "
              f"actual {r.actual_follower_cum*100:+6.2f}%  "
              f"capture {cap_str}")
