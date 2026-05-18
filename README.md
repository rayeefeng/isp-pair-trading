# Crypto Pair Trading Toolkit

A small research toolkit for statistical-arbitrage style **pair trading** on
crypto. Fetches daily history from CoinGecko, runs cointegration and z-score
signal analysis on any two coins, and screens a basket of coins for the best
candidate pairs.

## Why Parquet?

Pair trading is column-heavy work (rolling stats, cointegration, z-scores).
Parquet's columnar layout matches that access pattern:

| Concern                 | CSV                       | Parquet                            |
|-------------------------|---------------------------|------------------------------------|
| File size               | 1×                        | ~0.1–0.2× (snappy compressed)      |
| Read speed into pandas  | 1×                        | 10–50× faster                      |
| Date columns            | Re-parsed every load      | Stored as native `datetime64`      |
| Column subsetting       | Reads entire file         | Reads only requested columns       |
| Schema                  | Lost (everything is text) | Preserved                          |

## Layout

```
src/pairtrade/      reusable library
  data.py           CoinGecko fetcher + parquet I/O
  analysis.py       β, spread, z-score, signals, cointegration
  screening.py      rank all C(N, 2) pairs in a basket
  plots.py          matplotlib charts
scripts/            CLI entry points
  fetch.py
  analyze.py
  screen.py
notebooks/
  exploration.ipynb interactive workflow
data/               parquet outputs (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Fetch historical data

```bash
python scripts/fetch.py                                      # BTC + ETH, max history
python scripts/fetch.py --coins bitcoin ethereum solana cardano ripple
python scripts/fetch.py --days 730                           # last 2 years
```

Writes `data/<coin>_history.parquet`.

### 2. Analyze one pair

```bash
python scripts/analyze.py --a bitcoin --b ethereum
python scripts/analyze.py --a bitcoin --b ethereum --window 90 --entry 2.5 --exit 0.3
```

Prints a summary (β, correlation, cointegration p-value, signal count), writes
`analysis.parquet` with the full time series, and saves a 3-panel chart
(`analysis.png`: prices, spread, z-score with entry bands).

### 3. Screen a basket

```bash
python scripts/screen.py --coins bitcoin ethereum solana cardano ripple dogecoin litecoin
```

Runs every pair, ranks by cointegration p-value, writes `screen.parquet` and a
horizontal bar chart `screen.png`.

### 4. Interactive exploration

```bash
jupyter notebook notebooks/exploration.ipynb
```

Imports the same library modules and renders plots inline. Good for tweaking
windows / thresholds and re-running single cells.

## What the analysis produces

For each day:
- `log_<coin>` and `ret_<coin>` — log prices and log returns
- `rolling_corr` — N-day correlation of returns
- `spread = log(A) − β·log(B)` — mean-reverting if the pair is cointegrated
- `zscore` — rolling z-score of the spread
- `position` — `−1` (short spread), `+1` (long spread), `0` (flat), driven by a
  state machine that enters at `|z| > entry_z` and exits at `|z| < exit_z`

## Reading the data later

```python
import pandas as pd
btc = pd.read_parquet("data/bitcoin_history.parquet")
analysis = pd.read_parquet("analysis.parquet")
ranked = pd.read_parquet("screen.parquet")
```

## Coin ids

CoinGecko uses slugs, not tickers: `bitcoin`, `ethereum`, `solana`, `cardano`,
`ripple`, `dogecoin`, `litecoin`, etc. Full list at
https://api.coingecko.com/api/v3/coins/list

## Notes

- The fetcher uses CoinGecko's free tier, which is rate-limited to ~30 calls/min.
  `fetch_many` sleeps 2.5s between coins to stay polite.
- β is computed once over the full history. For a production system you'd want
  a rolling β; the static version is fine for research.
- This is a research toolkit, not a trading system — no order routing, no
  transaction costs, no slippage model.
