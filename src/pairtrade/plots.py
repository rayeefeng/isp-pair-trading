"""Side-by-side trend comparison chart for two coins."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_comparison(df: pd.DataFrame, a: str, b: str, normalize: bool = True):
    """Both coins on one chart.

    normalize=True (default): rebase each series to 100 at the start so trends
    are directly comparable on a single y-axis.
    normalize=False: dual y-axes showing raw prices.
    """
    df = df.sort_values("date").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 5))

    if normalize:
        series_a = df[a] / df[a].iloc[0] * 100
        series_b = df[b] / df[b].iloc[0] * 100
        ax.plot(df["date"], series_a, label=a, color="tab:blue", linewidth=1.5)
        ax.plot(df["date"], series_b, label=b, color="tab:orange", linewidth=1.5)
        ax.axhline(100, color="grey", linewidth=0.6, linestyle="--")
        ax.set_ylabel("price (rebased to 100 at start)")
        ax.set_title(f"{a} vs {b} — rebased trend comparison")
        ax.legend(loc="best")
    else:
        ax.plot(df["date"], df[a], label=a, color="tab:blue", linewidth=1.5)
        ax.set_ylabel(f"{a} price (USD)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(df["date"], df[b], label=b, color="tab:orange", linewidth=1.5)
        ax2.set_ylabel(f"{b} price (USD)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax.set_title(f"{a} vs {b} — raw prices (dual axis)")

    ax.set_xlabel("date")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
