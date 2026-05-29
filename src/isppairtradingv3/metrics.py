"""Performance metrics from a daily strategy-return series and a trade log."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365  # crypto trades every calendar day


def performance_metrics(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict:
    """Compute headline metrics.

    `daily` must have columns: date, position, ret (simple daily strategy return),
    and equity (cumulative product of 1+ret). `trades` has one row per closed
    trade with a `net_return` column.
    """
    ret = daily["ret"].fillna(0.0)
    equity = daily["equity"]
    n_days = len(ret)
    years = n_days / periods_per_year if periods_per_year else np.nan

    total_return = float(equity.iloc[-1] - 1) if n_days else np.nan
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years and years > 0 else np.nan

    mean, std = float(ret.mean()), float(ret.std(ddof=1)) if n_days > 1 else np.nan
    sharpe = float(mean / std * np.sqrt(periods_per_year)) if std and std > 0 else np.nan
    downside = ret[ret < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sortino = float(mean / dstd * np.sqrt(periods_per_year)) if dstd and dstd > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = float(drawdown.min()) if n_days else np.nan
    calmar = float(cagr / abs(max_dd)) if max_dd and max_dd < 0 else np.nan

    exposure = float((daily["position"] != 0).mean()) if n_days else np.nan

    # trade-level
    if trades is not None and not trades.empty:
        nr = trades["net_return"]
        n_trades = int(len(trades))
        win_rate = float((nr > 0).mean())
        avg_trade = float(nr.mean())
        gross_win = float(nr[nr > 0].sum())
        gross_loss = float(-nr[nr < 0].sum())
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else np.inf
        avg_win = float(nr[nr > 0].mean()) if (nr > 0).any() else np.nan
        avg_loss = float(nr[nr < 0].mean()) if (nr < 0).any() else np.nan
    else:
        n_trades = 0
        win_rate = avg_trade = profit_factor = avg_win = avg_loss = np.nan

    return {
        "n_days": n_days,
        "years": years,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "exposure": exposure,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def buy_and_hold_metrics(df: pd.DataFrame, asset: str,
                         periods_per_year: int = PERIODS_PER_YEAR) -> dict:
    """Benchmark: buy `asset` on day 1, hold to the end."""
    work = df.sort_values("date").reset_index(drop=True)
    ret = work[asset].pct_change().fillna(0.0)
    equity = (1 + ret).cumprod()
    daily = pd.DataFrame({"date": work["date"], "position": 1.0, "ret": ret, "equity": equity})
    m = performance_metrics(daily, pd.DataFrame(), periods_per_year)
    m["n_trades"] = 1
    return m


def print_metrics(m: dict, title: str) -> None:
    def pct(x):
        return "   n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:+.2f}%"

    def num(x):
        if x is None or (isinstance(x, float) and (np.isnan(x))):
            return "   n/a"
        if np.isinf(x):
            return "    inf"
        return f"{x:.2f}"

    print(f"\n=== {title} ===")
    print(f"  Period:            {m['n_days']} days  (~{m['years']:.1f} years)")
    print(f"  Total return:      {pct(m['total_return'])}")
    print(f"  CAGR:              {pct(m['cagr'])}")
    print(f"  Sharpe (ann.):     {num(m['sharpe'])}")
    print(f"  Sortino (ann.):    {num(m['sortino'])}")
    print(f"  Max drawdown:      {pct(m['max_drawdown'])}")
    print(f"  Calmar:            {num(m['calmar'])}")
    print(f"  Time in market:    {pct(m['exposure'])}")
    print(f"  Trades:            {m['n_trades']}")
    print(f"  Win rate:          {pct(m['win_rate'])}")
    print(f"  Avg trade:         {pct(m['avg_trade'])}")
    print(f"  Avg win / loss:    {pct(m['avg_win'])} / {pct(m['avg_loss'])}")
    print(f"  Profit factor:     {num(m['profit_factor'])}")
