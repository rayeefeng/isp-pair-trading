# Crypto Trend Comparison

A small Python tool to fetch the daily price history of two cryptocurrencies
from Yahoo Finance and compare their trends — both visually (a chart with both
coins overlaid) and as a plain-English summary.

## Why Parquet?

The fetched price data is stored as Parquet rather than CSV/JSON:

| Concern                 | CSV                       | Parquet                            |
|-------------------------|---------------------------|------------------------------------|
| File size               | 1×                        | ~0.1–0.2× (snappy compressed)      |
| Read speed into pandas  | 1×                        | 10–50× faster                      |
| Date columns            | Re-parsed every load      | Stored as native `datetime64`      |
| Schema                  | Lost (everything is text) | Preserved                          |

## Layout

```
src/pairtrade/                  v1 library
  data.py                       Yahoo Finance fetcher + parquet I/O
  analysis.py                   per-coin returns + correlation + verdict
  leadlag.py                    parameter sweep + event detection (no β)
  plots.py                      comparison / CCF / sweep / event-scatter charts
src/isppairtradingv2/           v2 library  (adds rolling hedge ratio)
  hedge_ratio.py                look-ahead-safe rolling cumulative-response β
  leadlag.py                    events augmented with predicted move + capture ratio
  plots.py                      rolling β, predicted-vs-actual, capture histogram
src/isppairtradingv3/           v3 library  (backtest + walk-forward)
  backtest.py                   event-driven P&L sim with transaction costs
  metrics.py                    Sharpe, Sortino, drawdown, win rate, profit factor
  walkforward.py                pick params in-sample, test out-of-sample, roll
  plots.py                      equity curve, drawdown, walk-forward comparison
scripts/
  fetch.py                      fetch one or more tickers
  analyze.py                    long-run trend comparison
  leadlag.py                    v1 lead-lag / opportunity detection
  leadlagv2.py                  v2 (β-aware) lead-lag analysis
  backtest.py                   v3 backtest + walk-forward validation
notebooks/
  exploration.ipynb             interactive workflow
data/                           parquet outputs (gitignored)
```

## Setup

```powershell
pip install -r requirements.txt
```

## Workflow

### 1. Fetch historical data

```powershell
python scripts/fetch.py --coins BTC-USD ETH-USD
python scripts/fetch.py --coins BTC-USD ETH-USD SOL-USD ADA-USD    # multiple
python scripts/fetch.py --days 730                                 # last 2 years
```

Writes `data/<ticker>_history.parquet`.

### 2. Compare two coins

```powershell
python scripts/analyze.py --a BTC-USD --b ETH-USD
python scripts/analyze.py --a BTC-USD --b ETH-USD --raw            # raw $ on dual axes
```

Prints a summary like:

```
Period: 2014-09-17 → 2026-05-18  (4263 daily observations)
  BTC-USD      $      457.33  →  $  103,221.40   (+22473.0%)
  ETH-USD      $        ...   →  $    4,082.10   (...)
  Correlation of daily returns: +0.785

Verdict: both trended up; moved tightly together (corr=0.78)
```

And saves `comparison.png` — both coins on one chart, rebased to 100 at the
start so trends are directly comparable on a single y-axis.

### 3. Lead-lag analysis (find day-to-day trading opportunities)

Instead of the overall trend, look at *when one coin moves big, does the
other follow with a delay?* — and let the data pick the threshold and
lookahead window.

```powershell
python scripts/leadlag.py --a BTC-USD --b ETH-USD
```

What it does:

1. Reports the daily return distribution for each coin (so "big move" has
   a concrete meaning — e.g. *"5% is BTC's 95th percentile day"*).
2. Computes the cross-correlation of daily returns at lags from −10 to +10
   to see which coin tends to lead the other and by how many days.
3. Sweeps a grid of *(threshold, lookahead, direction)* combinations against
   the entire history and ranks them by signal strength — the metric is
   `mean_follow × √n_events`, which rewards both effect size and sample size.
4. Picks the strongest combination (requiring ≥ 10 events for credibility),
   then lists every event with date, leader move, follower's biggest
   same-direction move within the window, and delay.

Example output (BTC-USD vs ETH-USD):

```
=== Recommended parameters ===
  Direction:       BTC-USD → ETH-USD
  Threshold:       5.0% daily move in the leader
  Lookahead:       2 day(s)
  Events:          47
  Hit rate:        68%   (follower moved same direction)
  Mean follow:     +1.85% cumulative log return
  t-statistic:     +3.21   (>2 ≈ statistically meaningful)

=== Events: BTC-USD moves ≥ 5%, look 2 day(s) ahead in ETH-USD ===
  Total events:              47
  Same-direction follow:     32  (68%)
  Avg delay (same-dir):      1.3 day(s)
  Avg leader move size:      +6.4% (absolute)
  Avg follower move size:    +3.8% (same direction, peak day)

  Most recent events:
    2025-08-12  BTC-USD  +6.20%   → ETH-USD  +3.10% on 2025-08-13  (1d delay)
    2025-07-04  BTC-USD  -8.10%   → ETH-USD  -4.50% on 2025-07-05  (1d delay)
    ...
```

Also saves three charts:

- `ccf.png` — cross-correlation bar chart (which lag is dominant)
- `sweep_heatmap.png` — mean follower return for each (threshold, lookahead)
- `events_scatter.png` — leader move vs follower response, colored by direction

And the events table as `leadlag_events.parquet` for further inspection.

### 4. v2 — same analysis with a rolling hedge ratio

`leadlagv2.py` runs the same pipeline (distribution → cross-correlation →
sweep → recommendation) and then fits a **rolling hedge ratio** β at the
recommended lag. For each event you then see how much of the move the
follower *should* have made vs how much it actually made.

```powershell
python scripts/leadlagv2.py --a BTC-USD --b ETH-USD
python scripts/leadlagv2.py --a BTC-USD --b ETH-USD --beta-window 60
```

The β used is the **cumulative-response** form, fit only on past data:

```
β(t) = cov( ret_leader(s), Σ ret_follower(s+1..s+K) ) / var( ret_leader(s) )
       over s in the trailing 90-day window
```

Interpretation: "given a leader move of `x` today, the follower's
cumulative return over the next `K` days is, on average, `β·x`." A
β of 0.5 means the follower captures half the move; β of 1.2 means it
over-shoots. β is recomputed every day from a rolling window, so you see
how the relationship has evolved.

For each event we then report:

- **predicted follower move**:  `β(event_date) × leader_return`
- **actual follower move**:    cumulative log return over the lookahead window
- **capture ratio**:           `actual / predicted`
  - `≈ 1.0` → follower moved exactly as β predicted
  - `> 1.0` → over-shoot (potentially still tradeable, but momentum is overshooting)
  - `0 < r < 1` → under-shoot (the typical "lag still has room to run" case)
  - `< 0`     → moved opposite of prediction (β-broken event)

Three extra charts get saved:

- `rolling_beta.png` — β over time (when did the relationship strengthen / break?)
- `predicted_vs_actual.png` — scatter; diagonal = perfect β prediction
- `capture_ratio_hist.png` — distribution of capture ratios

And the augmented events table as `leadlagv2_events.parquet`, the β series
as `rolling_beta.parquet`.

**v1 vs v2 in one sentence:** v1 tells you *whether the follower moved
same direction*; v2 tells you *whether the size of the follower's move
matched what its historical relationship with the leader predicts*.

### 5. v3 — does it actually make money? (backtest + walk-forward)

v1/v2 tell you the signal *exists*. v3 asks the only question that matters
for trading: **does it make money after costs, out of sample?**

```powershell
python scripts/backtest.py --a BTC-USD --b ETH-USD
python scripts/backtest.py --a BTC-USD --b ETH-USD --cost 0.0026 --entry-delay 0
```

It runs two backtests of the lead-lag follower trade (enter the follower in
the leader's direction after a big leader move, hold for the recommended
lookahead, exit):

1. **In-sample** — params picked on the full history, tested on the full
   history. Optimistic; this is the number that fools people.
2. **Walk-forward (out-of-sample)** — for each test window, params are picked
   using *only* prior data, then tested on the unseen window. This is the
   honest estimate of live performance.

Both report Sharpe, Sortino, max drawdown, CAGR, win rate, and profit
factor, with a buy-&-hold benchmark for context. A closing "reality check"
compares in-sample vs out-of-sample Sharpe and flags overfitting.

Key flags:

- `--cost` — transaction cost **per side** (fees + slippage). Default `0.001`
  (0.10%); round trip is `2×`. Try `0.0005` and `0.0026` to bracket it.
- `--entry-delay` — `1` (default, conservative: act the day after the signal)
  or `0` (aggressive: trade at the signal-bar close).
- `--train-frac` / `--windows` — size of the initial training block and the
  number of walk-forward test windows.

Saves `equity_insample.png`, `drawdown_insample.png`, `equity_oos.png`,
`walkforward.png`, and `backtest_trades.parquet`.

> ⚠️ Even a good walk-forward result is not a guarantee. Costs, slippage in
> fast markets, and regime changes can still sink a live strategy. Paper-trade
> before risking real money.

### 6. (Optional) Interactive notebook

```powershell
jupyter notebook notebooks/exploration.ipynb
```

## What "verdict" means

The verdict line summarizes two things:

- **Trend direction**: did each coin end above or below where it started?
  Either *"both trended up/down"* or *"diverged: A up, B down"*.
- **Co-movement**: the Pearson correlation of daily log returns —
  - `|r| ≥ 0.7` → *"moved tightly together"* (or *"opposite each other"*)
  - `0.3 ≤ |r| < 0.7` → *"moderately linked"*
  - `|r| < 0.3` → *"largely independent day-to-day"*

Trend direction tells you the *long-run* path. Correlation tells you whether
they *jiggle in sync day-to-day*. Two coins can both trend up while having
weak day-to-day correlation, or have strong day-to-day correlation while one
ends up and the other down (rare but possible over short windows).

## Ticker format

Yahoo Finance uses the format `<SYMBOL>-USD` for crypto:

| Coin      | Ticker     |
|-----------|------------|
| Bitcoin   | `BTC-USD`  |
| Ethereum  | `ETH-USD`  |
| Solana    | `SOL-USD`  |
| Cardano   | `ADA-USD`  |
| XRP       | `XRP-USD`  |
| Dogecoin  | `DOGE-USD` |
| Litecoin  | `LTC-USD`  |

Browse all crypto symbols at https://finance.yahoo.com/crypto/.

## Notes

- Yahoo Finance is free, no API key needed.
- Daily candles. Bitcoin history goes back to 2014-09-17 on Yahoo;
  other coins start from their respective listing dates.
