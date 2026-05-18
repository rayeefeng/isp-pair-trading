"""Matplotlib charts for pair-trading analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_pair(out: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.5):
    """Three stacked panels: prices, spread, z-score with entry bands and positions."""
    a, b = out.attrs["a"], out.attrs["b"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    ax = axes[0]
    ax.plot(out["date"], out[a], label=a, color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(out["date"], out[b], label=b, color="tab:orange", alpha=0.8)
    ax.set_ylabel(f"{a} price", color="tab:blue")
    ax2.set_ylabel(f"{b} price", color="tab:orange")
    ax.set_title(f"{a} vs {b}  (β={out.attrs['beta']:.3f}, window={out.attrs['window']})")

    axes[1].plot(out["date"], out["spread"], color="tab:purple")
    axes[1].plot(out["date"], out["spread_mean"], color="black", linewidth=0.8, label="rolling mean")
    axes[1].set_ylabel("spread = log(A) − β·log(B)")
    axes[1].legend(loc="upper left")

    z = axes[2]
    z.plot(out["date"], out["zscore"], color="tab:green")
    z.axhline(entry_z, color="red", linestyle="--", linewidth=0.8, label=f"±{entry_z} entry")
    z.axhline(-entry_z, color="red", linestyle="--", linewidth=0.8)
    z.axhline(exit_z, color="grey", linestyle=":", linewidth=0.8, label=f"±{exit_z} exit")
    z.axhline(-exit_z, color="grey", linestyle=":", linewidth=0.8)
    z.axhline(0, color="black", linewidth=0.5)

    long_mask = out["position"] == 1
    short_mask = out["position"] == -1
    z.fill_between(out["date"], -10, 10, where=long_mask, alpha=0.08, color="green")
    z.fill_between(out["date"], -10, 10, where=short_mask, alpha=0.08, color="red")
    z.set_ylim(out["zscore"].min() - 0.5, out["zscore"].max() + 0.5)
    z.set_ylabel("z-score")
    z.legend(loc="upper left")
    z.set_xlabel("date")

    fig.tight_layout()
    return fig


def plot_screen(ranked: pd.DataFrame, top: int = 15):
    """Horizontal bar chart of cointegration p-values for the top N pairs."""
    df = ranked.head(top).iloc[::-1]
    labels = [f"{r.a}/{r.b}" for r in df.itertuples()]
    colors = ["tab:green" if c else "lightgrey" for c in df["cointegrated"]]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(df))))
    ax.barh(labels, df["coint_pvalue"], color=colors)
    ax.axvline(0.05, color="red", linestyle="--", linewidth=0.8, label="p = 0.05")
    ax.set_xlabel("cointegration p-value (lower = stronger)")
    ax.set_title(f"Top {len(df)} pairs by cointegration")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
