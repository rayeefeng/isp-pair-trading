"""Event-driven backtest of the lead-lag follower trade, with transaction costs.

Strategy
--------
When the leader's daily return is >= `threshold` in magnitude on day t, take a
same-direction position in the FOLLOWER. Enter at the close of day
(t + entry_delay), hold `hold` days, exit at the close.

  - entry_delay=1 (default): you act the day AFTER the signal forms — realistic,
    since the signal is only known at day-t close. Conservative: misses the
    immediate t→t+1 follow-through.
  - entry_delay=0: enter at the signal-day close. Aggressive: assumes you can
    trade at the very bar that produced the signal.

Costs: `cost` is charged per side (entry and exit), so a round trip costs
2 × cost. It bundles fees + slippage.

Positions are non-overlapping by default (one trade at a time); a signal that
fires while a position is open is ignored unless allow_overlap=True.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    leader: str,
    follower: str,
    threshold: float,
    lookahead: int,
    cost: float = 0.001,
    entry_delay: int = 1,
    hold: int | None = None,
    allow_overlap: bool = False,
) -> dict:
    """Return {'daily': DataFrame(date, position, ret, equity), 'trades': DataFrame}."""
    hold = int(hold if hold is not None else lookahead)
    work = df.sort_values("date").reset_index(drop=True)
    n = len(work)

    ret_L = np.log(work[leader]).diff().values          # signal (log) returns
    price_F = work[follower].values
    simple_F = work[follower].pct_change().values        # follower P&L returns

    position = np.zeros(n)
    daily_ret = np.zeros(n)
    trades = []

    t = 0
    while t < n:
        if np.isfinite(ret_L[t]) and abs(ret_L[t]) >= threshold:
            direction = 1 if ret_L[t] > 0 else -1
            entry_idx = t + entry_delay
            exit_idx = entry_idx + hold
            if exit_idx >= n:
                break
            entry_price = price_F[entry_idx]
            exit_price = price_F[exit_idx]
            if not (np.isfinite(entry_price) and np.isfinite(exit_price)) or entry_price <= 0:
                t += 1
                continue

            gross = direction * (exit_price - entry_price) / entry_price
            net = gross - 2 * cost

            # Daily attribution: market return accrues from entry_idx+1..exit_idx;
            # fees charged at entry and exit days.
            for k in range(entry_idx + 1, exit_idx + 1):
                if np.isfinite(simple_F[k]):
                    position[k] = direction
                    daily_ret[k] += direction * simple_F[k]
            daily_ret[entry_idx] -= cost
            daily_ret[exit_idx] -= cost
            position[entry_idx] = direction

            trades.append({
                "entry_date": work["date"].iloc[entry_idx],
                "exit_date": work["date"].iloc[exit_idx],
                "direction": direction,
                "leader_return": float(ret_L[t]),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_return": float(gross),
                "net_return": float(net),
            })

            t = (entry_idx + 1) if allow_overlap else (exit_idx + 1)
        else:
            t += 1

    daily = pd.DataFrame({
        "date": work["date"],
        "position": position,
        "ret": daily_ret,
    })
    daily["equity"] = (1.0 + daily["ret"]).cumprod()
    return {"daily": daily, "trades": pd.DataFrame(trades)}
