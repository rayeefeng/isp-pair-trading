"""Lead-lag analysis v2 — adds rolling hedge ratio (β) at the recommended lag.

Pipeline:
    1. Show daily return distribution (v1 logic)
    2. Cross-correlation of returns (v1 logic)
    3. Parameter sweep + recommend (threshold, lookahead, direction)  (v1 logic)
    4. NEW: fit a rolling cumulative-response β at the recommended lag
    5. NEW: find events with predicted follower move and capture ratio
    6. NEW: charts for rolling β, predicted-vs-actual, capture-ratio histogram

The β fit is look-ahead-safe: β(t) only uses data observable strictly before t.

Usage:
    python scripts/leadlagv2.py --a BTC-USD --b ETH-USD
    python scripts/leadlagv2.py --a BTC-USD --b ETH-USD --beta-window 60 --recent 20
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt

from pairtrade.data import load_pair
from pairtrade.leadlag import (
    cross_correlation,
    daily_return_stats,
    lead_lag_verdict,
    parameter_sweep,
    print_recommendation,
    print_return_distribution,
    print_sweep_table,
    recommend_params,
)

from isppairtradingv2.hedge_ratio import (
    print_beta_summary,
    rolling_cumulative_beta,
    summarize_beta,
)
from isppairtradingv2.leadlag import (
    find_events_with_capture,
    print_capture_summary,
    print_events_with_capture,
    summarize_capture,
)
from isppairtradingv2.plots import (
    plot_capture_ratio_hist,
    plot_predicted_vs_actual,
    plot_rolling_beta,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="First ticker (e.g. BTC-USD)")
    p.add_argument("--b", required=True, help="Second ticker (e.g. ETH-USD)")
    p.add_argument("--data", default="data")
    p.add_argument("--max-lag", type=int, default=10,
                   help="Max lag (days) for cross-correlation")
    p.add_argument("--min-events", type=int, default=10,
                   help="Min events required for a (threshold, lookahead) to be recommendable")
    p.add_argument("--beta-window", type=int, default=90,
                   help="Rolling window in days for the β fit")
    p.add_argument("--recent", type=int, default=12,
                   help="How many recent events to list")
    p.add_argument("--plots-dir", default=".",
                   help="Directory to save chart PNGs (use 'none' to skip plotting)")
    args = p.parse_args()

    df = load_pair(args.a, args.b, args.data)

    # 1. Return distribution
    print_return_distribution([
        daily_return_stats(df, args.a),
        daily_return_stats(df, args.b),
    ])

    # 2. Cross-correlation
    ccf = cross_correlation(df, args.a, args.b, max_lag=args.max_lag)
    print("\n=== Cross-correlation of daily returns ===")
    print(lead_lag_verdict(ccf, args.a, args.b))

    # 3. Sweep + recommend
    sweep = parameter_sweep(df, args.a, args.b)
    print_sweep_table(sweep, top=8)
    rec = recommend_params(sweep, min_events=args.min_events)
    print_recommendation(rec, sweep)

    if rec is None:
        print("\nNo recommended parameters → cannot fit β or analyze capture.")
        return

    leader = rec["leader"]
    follower = rec["follower"]
    threshold = float(rec["threshold"])
    lookahead = int(rec["lookahead"])

    # 4. Rolling β at the recommended lag
    beta_df = rolling_cumulative_beta(df, leader, follower, lookahead, args.beta_window)
    beta_stats = summarize_beta(beta_df)
    print_beta_summary(beta_stats, leader, follower, lookahead, args.beta_window)

    # 5. Events with capture
    events = find_events_with_capture(df, leader, follower, threshold, lookahead,
                                      beta_window=args.beta_window)
    cap_stats = summarize_capture(events)
    print_capture_summary(cap_stats, leader, follower, threshold, lookahead)
    print_events_with_capture(events, leader, follower, show=args.recent)

    # 6. Save outputs
    if not events.empty:
        events.to_parquet("leadlagv2_events.parquet", index=False)
        print(f"\nEvents table saved → leadlagv2_events.parquet  ({len(events)} rows)")
    beta_df.dropna(subset=["beta"]).to_parquet("rolling_beta.parquet", index=False)
    print(f"Rolling β saved     → rolling_beta.parquet")

    if args.plots_dir.lower() != "none":
        out_dir = Path(args.plots_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig1 = plot_rolling_beta(beta_df, leader, follower, lookahead, args.beta_window)
        fig1.savefig(out_dir / "rolling_beta.png", dpi=110)
        plt.close(fig1)

        if not events.empty:
            fig2 = plot_predicted_vs_actual(events, leader, follower)
            fig2.savefig(out_dir / "predicted_vs_actual.png", dpi=110)
            plt.close(fig2)

            fig3 = plot_capture_ratio_hist(events, leader, follower)
            fig3.savefig(out_dir / "capture_ratio_hist.png", dpi=110)
            plt.close(fig3)

        print(f"Plots saved         → {out_dir}/  "
              f"(rolling_beta.png, predicted_vs_actual.png, capture_ratio_hist.png)")


if __name__ == "__main__":
    main()
