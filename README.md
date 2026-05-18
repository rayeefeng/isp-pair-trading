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
src/pairtrade/
  data.py           Yahoo Finance fetcher + parquet I/O
  analysis.py       per-coin returns + correlation + plain-English verdict
  leadlag.py        daily lead-lag analysis + parameter sweep + event detection
  plots.py          comparison chart, cross-correlation, sweep heatmap, event scatter
scripts/
  fetch.py          CLI: fetch one or more tickers
  analyze.py        CLI: long-run trend comparison of two tickers
  leadlag.py        CLI: lead-lag / opportunity detection
notebooks/
  exploration.ipynb interactive workflow
data/               parquet outputs (gitignored)
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

### 4. (Optional) Interactive notebook

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
