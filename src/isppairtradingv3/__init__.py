"""isppairtradingv3 — realistic backtest + walk-forward validation.

Where v1/v2 measure whether the lead-lag signal *exists*, v3 asks the only
question that matters for trading: **does it make money after costs, out of
sample?**

  - backtest.py     event-driven P&L simulation with transaction costs
  - metrics.py      Sharpe, Sortino, max drawdown, win rate, profit factor
  - walkforward.py  pick params in-sample, test out-of-sample, roll forward
  - plots.py        equity curve, drawdown, walk-forward comparison
"""

from . import backtest, metrics, plots, walkforward

__all__ = ["backtest", "metrics", "plots", "walkforward"]
