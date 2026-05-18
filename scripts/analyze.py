"""Analyze one pair: print summary, save parquet + plot.

Usage:
    python scripts/analyze.py --a bitcoin --b ethereum
    python scripts/analyze.py --a bitcoin --b ethereum --window 90 --entry 2.5 --exit 0.3
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt

from pairtrade.analysis import analyze, print_summary, summarize
from pairtrade.data import load_pair
from pairtrade.plots import plot_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--entry", type=float, default=2.0)
    p.add_argument("--exit", type=float, default=0.5)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="analysis.parquet")
    p.add_argument("--plot", default="analysis.png", help="PNG path, or 'none' to skip")
    args = p.parse_args()

    df = load_pair(args.a, args.b, args.data)
    result = analyze(df, args.a, args.b, args.window, args.entry, args.exit)
    result.to_parquet(args.out, index=False)
    print_summary(summarize(result))
    print(f"\nAnalysis saved → {args.out}")

    if args.plot.lower() != "none":
        fig = plot_pair(result, entry_z=args.entry, exit_z=args.exit)
        fig.savefig(args.plot, dpi=110)
        plt.close(fig)
        print(f"Plot saved     → {args.plot}")


if __name__ == "__main__":
    main()
