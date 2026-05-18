"""Compare two coins: trend direction summary + comparison chart.

Usage:
    python scripts/analyze.py --a bitcoin --b ethereum
    python scripts/analyze.py --a bitcoin --b ethereum --raw    # raw prices (dual axis)
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt

from pairtrade.analysis import compare, print_summary
from pairtrade.data import load_pair
from pairtrade.plots import plot_comparison


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="First coin id (e.g. bitcoin)")
    p.add_argument("--b", required=True, help="Second coin id (e.g. ethereum)")
    p.add_argument("--data", default="data")
    p.add_argument("--plot", default="comparison.png", help="PNG path, or 'none' to skip")
    p.add_argument("--raw", action="store_true",
                   help="Plot raw prices on dual axes (default: rebased to 100)")
    args = p.parse_args()

    df = load_pair(args.a, args.b, args.data)
    print_summary(compare(df, args.a, args.b))

    if args.plot.lower() != "none":
        fig = plot_comparison(df, args.a, args.b, normalize=not args.raw)
        fig.savefig(args.plot, dpi=110)
        plt.close(fig)
        print(f"\nChart saved → {args.plot}")


if __name__ == "__main__":
    main()
