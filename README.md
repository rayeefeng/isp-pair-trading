# Crypto Trend Comparison

A small Python tool to fetch the daily price history of two cryptocurrencies
from CoinGecko and compare their trends — both visually (a chart with both
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
  data.py           CoinGecko fetcher + parquet I/O
  analysis.py       per-coin returns + correlation + plain-English verdict
  plots.py          single comparison chart (rebased or dual-axis)
scripts/
  fetch.py          CLI: fetch one or more coins
  analyze.py        CLI: compare two coins
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
python scripts/fetch.py --coins bitcoin ethereum
python scripts/fetch.py --coins bitcoin ethereum solana cardano   # multiple
python scripts/fetch.py --days 730                                # last 2 years
```

Writes `data/<coin>_history.parquet`.

### 2. Compare two coins

```powershell
python scripts/analyze.py --a bitcoin --b ethereum
python scripts/analyze.py --a bitcoin --b ethereum --raw          # raw $ on dual axes
```

Prints a summary like:

```
Period: 2017-08-17 → 2026-05-18  (3197 daily observations)
  bitcoin      $    4,261.94  →  $  103,221.40   (+2322.0%)
  ethereum     $      301.55  →  $    4,082.10   (+1253.8%)
  Correlation of daily returns: +0.785

Verdict: both trended up; moved tightly together (corr=0.78)
```

And saves `comparison.png` — both coins on one chart, rebased to 100 at the
start so trends are directly comparable on a single y-axis.

### 3. (Optional) Interactive notebook

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

## Coin ids

CoinGecko uses slugs, not tickers: `bitcoin`, `ethereum`, `solana`, `cardano`,
`ripple`, `dogecoin`, `litecoin`, etc. Full list at
https://api.coingecko.com/api/v3/coins/list

## Notes

- CoinGecko free tier is rate-limited (~30 calls/min); the fetcher sleeps 2.5s
  between coins.
- Daily data only.
