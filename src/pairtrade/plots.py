"""Side-by-side trend comparison chart for two coins."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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


def plot_cross_correlation(ccf: pd.DataFrame, a: str, b: str):
    """Bar chart of corr(ret_a_t, ret_b_{t+k}) at each lag k.

    Positive lag = `a` leads `b`. The tallest bar tells you the dominant lag.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["tab:green" if v > 0 else "tab:red" for v in ccf["correlation"]]
    ax.bar(ccf["lag"], ccf["correlation"], color=colors, alpha=0.8)
    best = ccf.iloc[ccf["correlation"].abs().idxmax()]
    ax.axvline(best["lag"], color="black", linestyle="--", linewidth=0.8,
               label=f"peak lag = {int(best['lag']):+d}  (r={best['correlation']:+.2f})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"lag k  (positive = {a} leads {b})")
    ax.set_ylabel(f"corr(ret_{a}(t), ret_{b}(t+k))")
    ax.set_title(f"Cross-correlation of daily returns: {a} vs {b}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sweep_heatmap(sweep: pd.DataFrame, leader: str, follower: str, metric: str = "mean_follow"):
    """Heatmap of `metric` over (threshold, lookahead) for one direction.

    `metric` is one of: mean_follow, hit_rate, t_stat, score.
    """
    sub = sweep[(sweep["leader"] == leader) & (sweep["follower"] == follower)]
    if sub.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, f"no data for {leader} → {follower}",
                ha="center", va="center")
        ax.set_axis_off()
        return fig

    pivot = sub.pivot(index="threshold", columns="lookahead", values=metric).sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = "RdYlGn"
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v*100:.1f}%" for v in pivot.index])
    ax.set_xlabel("lookahead (days)")
    ax.set_ylabel("leader move threshold")
    ax.set_title(f"{metric} for {leader} → {follower}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            txt = f"{v*100:+.1f}%" if metric in ("mean_follow", "hit_rate") else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def plot_events_scatter(events: pd.DataFrame, leader: str, follower: str):
    """Scatter of leader return vs the follower's biggest same-direction move
    in the lookahead window. Green = same direction, red = opposite."""
    if events.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    fig, ax = plt.subplots(figsize=(7, 6))
    same = events[events["same_direction"]]
    diff = events[~events["same_direction"]]
    ax.scatter(same["leader_return"] * 100, same["follower_best_return"] * 100,
               color="tab:green", alpha=0.7, label=f"same direction (n={len(same)})")
    ax.scatter(diff["leader_return"] * 100, diff["follower_best_return"] * 100,
               color="tab:red", alpha=0.7, label=f"opposite direction (n={len(diff)})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"{leader} daily return on event day (%)")
    ax.set_ylabel(f"{follower} best same-direction move in window (%)")
    ax.set_title(f"Leader vs follower response  —  {leader} → {follower}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
