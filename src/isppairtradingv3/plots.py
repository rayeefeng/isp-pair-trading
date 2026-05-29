"""v3 charts: equity curve (vs buy & hold), drawdown, walk-forward comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity(daily: pd.DataFrame, benchmark: pd.DataFrame | None = None,
                title: str = "Strategy equity curve", benchmark_label: str = "buy & hold"):
    """Strategy equity (and optional benchmark) on a log y-axis."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily["date"], daily["equity"], color="tab:blue", linewidth=1.4, label="strategy")
    if benchmark is not None and not benchmark.empty:
        ax.plot(benchmark["date"], benchmark["equity"], color="tab:grey",
                linewidth=1.1, alpha=0.8, label=benchmark_label)
    ax.axhline(1.0, color="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("equity (×, log scale)")
    ax.set_xlabel("date")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_drawdown(daily: pd.DataFrame, title: str = "Drawdown"):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    dd = daily["equity"] / daily["equity"].cummax() - 1
    ax.fill_between(daily["date"], dd * 100, 0, color="tab:red", alpha=0.4)
    ax.set_ylabel("drawdown (%)")
    ax.set_xlabel("date")
    ax.set_title(f"{title}  (max {dd.min()*100:.1f}%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_walkforward(folds: pd.DataFrame):
    """Per-window out-of-sample return and Sharpe."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [f"W{int(r.window)}" for r in folds.itertuples()]

    rets = folds["total_return"].fillna(0).values * 100
    colors = ["tab:green" if v >= 0 else "tab:red" for v in rets]
    ax1.bar(labels, rets, color=colors)
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_ylabel("out-of-sample return (%)")
    ax1.set_title("OOS return per window")
    ax1.grid(alpha=0.3, axis="y")

    sharpes = folds["sharpe"].fillna(0).values
    colors2 = ["tab:green" if v >= 0 else "tab:red" for v in sharpes]
    ax2.bar(labels, sharpes, color=colors2)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("annualized Sharpe")
    ax2.set_title("OOS Sharpe per window")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    return fig
