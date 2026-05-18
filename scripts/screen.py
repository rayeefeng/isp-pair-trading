"""Screen a basket of coins: rank every pair by cointegration p-value.

Usage:
    python scripts/screen.py --coins bitcoin ethereum solana cardano ripple dogecoin
    python scripts/screen.py --coins bitcoin ethereum solana --window 90 --top 10
"""

import argparse

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt

from pairtrade.plots import plot_screen
from pairtrade.screening import screen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="screen.parquet")
    p.add_argument("--plot", default="screen.png", help="PNG path, or 'none' to skip")
    p.add_argument("--top", type=int, default=15, help="Rows printed/plotted")
    args = p.parse_args()

    ranked = screen(args.coins, data_dir=args.data, window=args.window)
    if ranked.empty:
        print("No pairs to rank — did you fetch the data first?")
        return

    ranked.to_parquet(args.out, index=False)
    with pd_display_options():
        print(ranked.head(args.top).to_string(index=False))
    print(f"\nFull ranking saved → {args.out}")

    if args.plot.lower() != "none":
        fig = plot_screen(ranked, top=args.top)
        fig.savefig(args.plot, dpi=110)
        plt.close(fig)
        print(f"Plot saved         → {args.plot}")


class pd_display_options:
    def __enter__(self):
        import pandas as pd
        self._old = {
            "display.float_format": pd.get_option("display.float_format"),
            "display.width": pd.get_option("display.width"),
        }
        pd.set_option("display.float_format", lambda x: f"{x:.4f}")
        pd.set_option("display.width", 140)

    def __exit__(self, *args):
        import pandas as pd
        for k, v in self._old.items():
            pd.set_option(k, v)


if __name__ == "__main__":
    main()
