"""Walk-forward validation.

The honest test: never let the strategy see its own future. We split the
history into an initial training block plus several sequential test windows.
For each test window we (1) pick the best (threshold, lookahead, direction)
using ONLY the data before the window — via v1's parameter sweep — then
(2) backtest those params on the window. The out-of-sample equity curve is the
concatenation of all test windows.

If out-of-sample Sharpe collapses versus in-sample, the signal was overfit.
If the selected parameters jump around between windows, it's unstable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairtrade.leadlag import parameter_sweep, recommend_params

from .backtest import run_backtest
from .metrics import performance_metrics


def walk_forward(
    df: pd.DataFrame,
    a: str,
    b: str,
    train_frac: float = 0.5,
    n_windows: int = 5,
    cost: float = 0.001,
    entry_delay: int = 1,
    min_events: int = 10,
) -> dict:
    """Return {'oos_daily', 'oos_metrics', 'folds'(DataFrame)}."""
    work = df.sort_values("date").reset_index(drop=True)
    n = len(work)
    train_end = int(n * train_frac)
    if train_end < 60 or train_end >= n:
        raise ValueError("Not enough data for the chosen train_frac.")

    test_chunks = [c for c in np.array_split(np.arange(train_end, n), n_windows) if len(c)]

    oos_daily_parts = []
    folds = []
    for w, idx in enumerate(test_chunks, start=1):
        test_start, test_end = int(idx[0]), int(idx[-1])
        train_df = work.iloc[:test_start]

        sweep = parameter_sweep(train_df, a, b)
        rec = recommend_params(sweep, min_events=min_events)

        fold = {
            "window": w,
            "test_start": work["date"].iloc[test_start],
            "test_end": work["date"].iloc[test_end],
            "test_days": test_end - test_start + 1,
        }

        if rec is None:
            fold.update(leader=None, follower=None, threshold=np.nan,
                        lookahead=np.nan, sharpe=np.nan, total_return=0.0,
                        n_trades=0)
            folds.append(fold)
            # flat (cash) over this window
            flat = work.iloc[test_start:test_end + 1][["date"]].copy()
            flat["position"] = 0.0
            flat["ret"] = 0.0
            oos_daily_parts.append(flat)
            continue

        # Backtest on all data up to the window end, then keep only the test slice.
        bt = run_backtest(work.iloc[:test_end + 1], rec["leader"], rec["follower"],
                          float(rec["threshold"]), int(rec["lookahead"]),
                          cost=cost, entry_delay=entry_delay)
        slice_daily = bt["daily"].iloc[test_start:test_end + 1][["date", "position", "ret"]].copy()
        oos_daily_parts.append(slice_daily)

        slice_eq = slice_daily.copy()
        slice_eq["equity"] = (1 + slice_eq["ret"]).cumprod()
        wm = performance_metrics(slice_eq, bt["trades"])
        fold.update(
            leader=rec["leader"], follower=rec["follower"],
            threshold=float(rec["threshold"]), lookahead=int(rec["lookahead"]),
            sharpe=wm["sharpe"], total_return=wm["total_return"],
            n_trades=wm["n_trades"],
        )
        folds.append(fold)

    oos_daily = pd.concat(oos_daily_parts).sort_values("date").reset_index(drop=True)
    oos_daily["equity"] = (1 + oos_daily["ret"]).cumprod()

    # Rebuild an approximate OOS trade log from daily P&L is messy; use daily metrics.
    oos_metrics = performance_metrics(oos_daily, _pseudo_trades(oos_daily))
    return {"oos_daily": oos_daily, "oos_metrics": oos_metrics, "folds": pd.DataFrame(folds)}


def _pseudo_trades(daily: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct trades from the daily position series so trade-level stats
    (win rate, profit factor) work on the stitched out-of-sample curve."""
    trades = []
    pos = daily["position"].values
    ret = daily["ret"].values
    i = 0
    n = len(daily)
    while i < n:
        if pos[i] != 0:
            j = i
            cum = 1.0
            direction = pos[i]
            while j < n and pos[j] == direction:
                cum *= (1 + ret[j])
                j += 1
            trades.append({"net_return": cum - 1, "direction": direction})
            i = j
        else:
            i += 1
    return pd.DataFrame(trades)


def print_walkforward(result: dict) -> None:
    folds = result["folds"]
    print("\n=== Walk-forward windows (out-of-sample) ===")
    print(f"  {'win':>3} {'test period':<25} {'leader→follower':<22} "
          f"{'thr':>5} {'look':>4} {'trades':>6} {'return':>8} {'sharpe':>7}")
    for r in folds.itertuples():
        pair = f"{r.leader}→{r.follower}" if r.leader else "(no signal)"
        thr = f"{r.threshold*100:.1f}%" if r.leader else "  -"
        look = f"{int(r.lookahead)}" if r.leader else " -"
        ret = f"{r.total_return*100:+.1f}%"
        sharpe = f"{r.sharpe:+.2f}" if not np.isnan(r.sharpe) else "   -"
        period = f"{r.test_start.date()}→{r.test_end.date()}"
        print(f"  {r.window:>3} {period:<25} {pair:<22} {thr:>5} {look:>4} "
              f"{r.n_trades:>6} {ret:>8} {sharpe:>7}")

    # Parameter stability note
    chosen = folds.dropna(subset=["threshold"])
    if not chosen.empty:
        n_unique = chosen[["leader", "follower", "threshold", "lookahead"]].drop_duplicates().shape[0]
        print(f"\n  Distinct parameter sets chosen across windows: {n_unique} "
              f"(fewer = more stable / less overfit)")
