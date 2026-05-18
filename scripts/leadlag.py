"""Lead-lag analysis: when one coin moves big, does the other follow?

Pipeline:
    1. Show daily return distribution for both coins (what's a normal move?)
    2. Cross-correlation of returns at every lag from -10 to +10
    3. Parameter sweep over (threshold, lookahead, leader) combinations
    4. Recommend the strongest combination, then list its events

Usage:
    python scripts/leadlag.py --a BTC-USD --b ETH-USD
    python scripts/leadlag.py --a BTC-USD --b ETH-USD --max-lag 14 --recent 20
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt

from pairtrade.data import load_pair
from pairtrade.leadlag import (
    cross_correlation,
    daily_return_stats,
    find_events,
    lead_lag_verdict,
    parameter_sweep,
    print_events,
    print_recommendation,
    print_return_distribution,
    print_sweep_table,
    recommend_params,
)
from pairtrade.plots import (
    plot_cross_correlation,
    plot_events_scatter,
    plot_sweep_heatmap,
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
    p.add_argument("--recent", type=int, default=12,
                   help="How many recent events to list")
    p.add_argument("--plots-dir", default=".",
                   help="Directory to save chart PNGs (use 'none' to skip plotting)")
    args = p.parse_args()

    df = load_pair(args.a, args.b, args.data)

    # 1. Daily return distribution
    stats_a = daily_return_stats(df, args.a)
    stats_b = daily_return_stats(df, args.b)
    print_return_distribution([stats_a, stats_b])

    # 2. Cross-correlation
    ccf = cross_correlation(df, args.a, args.b, max_lag=args.max_lag)
    print("\n=== Cross-correlation of daily returns ===")
    print(lead_lag_verdict(ccf, args.a, args.b))

    # 3. Parameter sweep
    sweep = parameter_sweep(df, args.a, args.b)
    print_sweep_table(sweep, top=8)

    # 4. Recommend + find events
    rec = recommend_params(sweep, min_events=args.min_events)
    print_recommendation(rec, sweep)

    events = None
    if rec is not None:
        events = find_events(df, rec["leader"], rec["follower"],
                             threshold=float(rec["threshold"]),
                             lookahead=int(rec["lookahead"]))
        print_events(events, rec["leader"], rec["follower"],
                     threshold=float(rec["threshold"]),
                     lookahead=int(rec["lookahead"]),
                     show=args.recent)

    # 5. Plots
    if args.plots_dir.lower() != "none":
        out_dir = Path(args.plots_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig1 = plot_cross_correlation(ccf, args.a, args.b)
        fig1.savefig(out_dir / "ccf.png", dpi=110)
        plt.close(fig1)

        if rec is not None:
            fig2 = plot_sweep_heatmap(sweep, rec["leader"], rec["follower"], metric="mean_follow")
            fig2.savefig(out_dir / "sweep_heatmap.png", dpi=110)
            plt.close(fig2)

            if events is not None and not events.empty:
                fig3 = plot_events_scatter(events, rec["leader"], rec["follower"])
                fig3.savefig(out_dir / "events_scatter.png", dpi=110)
                plt.close(fig3)

        print(f"\nPlots saved → {out_dir}/  (ccf.png, sweep_heatmap.png, events_scatter.png)")

    # 6. Save the events table
    if events is not None and not events.empty:
        events.to_parquet("leadlag_events.parquet", index=False)
        print(f"Events table saved → leadlag_events.parquet  ({len(events)} rows)")


if __name__ == "__main__":
    main()
