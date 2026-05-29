"""v2 charts: rolling β over time, predicted-vs-actual scatter, capture-ratio histogram."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_rolling_beta(beta_df: pd.DataFrame, leader: str, follower: str,
                      lookahead: int, window: int):
    """β over time. Useful to see when the relationship strengthened/weakened."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    valid = beta_df.dropna(subset=["beta"])
    ax.plot(valid["date"], valid["beta"], color="tab:blue", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.6)
    if not valid.empty:
        mean_beta = valid["beta"].mean()
        ax.axhline(mean_beta, color="grey", linestyle="--", linewidth=0.7,
                   label=f"mean β = {mean_beta:+.2f}")
        ax.legend()
    ax.set_xlabel("date")
    ax.set_ylabel(f"β (cumulative response over {lookahead}d)")
    ax.set_title(f"Rolling {window}-day hedge ratio: {leader} → {follower}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(events: pd.DataFrame, leader: str, follower: str):
    """Scatter: predicted follower cum return vs actual. Diagonal = perfect prediction."""
    if events.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    fig, ax = plt.subplots(figsize=(7, 6))
    pred = events["predicted_follower_cum"].values * 100
    act = events["actual_follower_cum"].values * 100
    colors = np.where(events["same_direction"].values, "tab:green", "tab:red")
    ax.scatter(pred, act, c=colors, alpha=0.7)

    lo = min(np.nanmin(pred), np.nanmin(act))
    hi = max(np.nanmax(pred), np.nanmax(act))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=0.8,
            label="actual = predicted")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlabel(f"predicted {follower} cumulative return (%)  [β × leader move]")
    ax.set_ylabel(f"actual {follower} cumulative return (%)")
    ax.set_title(f"β-based prediction vs reality  —  {leader} → {follower}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_capture_ratio_hist(events: pd.DataFrame, leader: str, follower: str):
    """Histogram of capture ratios. 1.0 = exactly matched β prediction."""
    if events.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no events", ha="center", va="center")
        ax.set_axis_off()
        return fig

    cap = events["capture_ratio"].dropna().values
    # Clip extreme outliers for readability (record raw values are kept in the parquet)
    clipped = np.clip(cap, -3.0, 3.0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(clipped, bins=40, color="tab:blue", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.6, label="0 (wrong direction)")
    ax.axvline(1, color="tab:green", linestyle="--", linewidth=0.9, label="1.0 (exactly as predicted)")
    median = float(np.median(cap))
    ax.axvline(median, color="tab:orange", linestyle="-.", linewidth=0.9, label=f"median = {median:+.2f}")
    ax.set_xlabel("capture ratio = actual / predicted  (clipped to [-3, 3] for display)")
    ax.set_ylabel("count")
    ax.set_title(f"Capture ratio distribution  —  {leader} → {follower}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
