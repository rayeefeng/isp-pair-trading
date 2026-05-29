"""Backtest the lead-lag follower trade with realistic costs, then validate
it out-of-sample with walk-forward windows.

Two backtests are run:
  1. In-sample: pick params on the FULL history, backtest on the full history.
     (Optimistic — the params saw the whole period. This is the number most
     people quote and most people are fooled by.)
  2. Walk-forward: pick params using only past data for each test window, then
     test on the unseen window. (Honest — this is closer to live performance.)

A buy-and-hold benchmark of the follower is shown for context.

Usage:
    python scripts/backtest.py --a BTC-USD --b ETH-USD
    python scripts/backtest.py --a BTC-USD --b ETH-USD --cost 0.0026 --entry-delay 0
    python scripts/backtest.py --a BTC-USD --b ETH-USD --windows 6 --train-frac 0.4
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt

from pairtrade.data import load_pair
from pairtrade.leadlag import parameter_sweep, print_recommendation, recommend_params

from isppairtradingv3.backtest import run_backtest
from isppairtradingv3.metrics import buy_and_hold_metrics, performance_metrics, print_metrics
from isppairtradingv3.plots import plot_drawdown, plot_equity, plot_walkforward
from isppairtradingv3.walkforward import print_walkforward, walk_forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="First ticker (e.g. BTC-USD)")
    p.add_argument("--b", required=True, help="Second ticker (e.g. ETH-USD)")
    p.add_argument("--data", default="data")
    p.add_argument("--cost", type=float, default=0.001,
                   help="Transaction cost per side (fees+slippage). Default 0.001 = 0.10%%.")
    p.add_argument("--entry-delay", type=int, default=1,
                   help="Days after the signal to enter (1=conservative, 0=aggressive)")
    p.add_argument("--train-frac", type=float, default=0.5,
                   help="Fraction of history used as the initial training block")
    p.add_argument("--windows", type=int, default=5, help="Number of walk-forward test windows")
    p.add_argument("--min-events", type=int, default=10)
    p.add_argument("--plots-dir", default=".", help="Where to save PNGs ('none' to skip)")
    args = p.parse_args()

    df = load_pair(args.a, args.b, args.data)
    print(f"Loaded {len(df)} aligned daily observations: "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Transaction cost: {args.cost*100:.3f}% per side "
          f"({args.cost*2*100:.3f}% round trip), entry delay: {args.entry_delay} day(s)")

    # ---- 1. In-sample backtest ----
    sweep = parameter_sweep(df, args.a, args.b)
    rec = recommend_params(sweep, min_events=args.min_events)
    print_recommendation(rec, sweep)
    if rec is None:
        print("\nNo tradeable signal found — nothing to backtest.")
        return

    leader, follower = rec["leader"], rec["follower"]
    threshold, lookahead = float(rec["threshold"]), int(rec["lookahead"])

    bt = run_backtest(df, leader, follower, threshold, lookahead,
                      cost=args.cost, entry_delay=args.entry_delay)
    is_metrics = performance_metrics(bt["daily"], bt["trades"])
    print_metrics(is_metrics, f"IN-SAMPLE backtest: {leader}→{follower} "
                              f"(thr {threshold:.0%}, hold {lookahead}d)")

    bh = buy_and_hold_metrics(df, follower)
    print_metrics(bh, f"Benchmark: buy & hold {follower}")

    # ---- 2. Walk-forward (out-of-sample) ----
    wf = walk_forward(df, args.a, args.b, train_frac=args.train_frac,
                      n_windows=args.windows, cost=args.cost,
                      entry_delay=args.entry_delay, min_events=args.min_events)
    print_walkforward(wf)
    print_metrics(wf["oos_metrics"], "OUT-OF-SAMPLE (stitched walk-forward)")

    # ---- Honest read ----
    is_sh, oos_sh = is_metrics["sharpe"], wf["oos_metrics"]["sharpe"]
    print("\n=== Reality check ===")
    if oos_sh is None or (oos_sh != oos_sh):  # nan
        print("  Out-of-sample Sharpe is undefined (too few trades). Treat as no edge.")
    elif oos_sh <= 0:
        print(f"  Out-of-sample Sharpe is {oos_sh:+.2f} ≤ 0 — no demonstrated edge after costs.")
    elif is_sh and oos_sh < 0.5 * is_sh:
        print(f"  In-sample Sharpe {is_sh:.2f} but out-of-sample only {oos_sh:.2f} — "
              f"strong overfitting; be very skeptical.")
    else:
        print(f"  In-sample {is_sh:.2f} vs out-of-sample {oos_sh:.2f} — holds up "
              f"reasonably. Still paper-trade before risking capital.")

    # ---- Save outputs ----
    if not bt["trades"].empty:
        bt["trades"].to_parquet("backtest_trades.parquet", index=False)
        print("\nTrades saved → backtest_trades.parquet")

    if args.plots_dir.lower() != "none":
        out = Path(args.plots_dir)
        out.mkdir(parents=True, exist_ok=True)

        bench = buy_and_hold_equity(df, follower)
        fig1 = plot_equity(bt["daily"], benchmark=bench,
                           title=f"In-sample equity: {leader}→{follower}",
                           benchmark_label=f"buy & hold {follower}")
        fig1.savefig(out / "equity_insample.png", dpi=110); plt.close(fig1)

        fig2 = plot_drawdown(bt["daily"], title="In-sample drawdown")
        fig2.savefig(out / "drawdown_insample.png", dpi=110); plt.close(fig2)

        fig3 = plot_equity(wf["oos_daily"], title="Out-of-sample (walk-forward) equity")
        fig3.savefig(out / "equity_oos.png", dpi=110); plt.close(fig3)

        fig4 = plot_walkforward(wf["folds"])
        fig4.savefig(out / "walkforward.png", dpi=110); plt.close(fig4)

        print(f"Plots saved → {out}/  (equity_insample, drawdown_insample, "
              f"equity_oos, walkforward)")


def buy_and_hold_equity(df, asset):
    import pandas as pd
    work = df.sort_values("date").reset_index(drop=True)
    ret = work[asset].pct_change().fillna(0.0)
    return pd.DataFrame({"date": work["date"], "equity": (1 + ret).cumprod()})


if __name__ == "__main__":
    main()
