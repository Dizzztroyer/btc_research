# BTC Research Platform

A fully local, production-grade BTC/USDT strategy research and backtesting
system. Downloads OHLCV data from Binance (or any ccxt exchange), stores it
locally in Parquet files, builds a rich feature set, and runs multiple
strategy families across all timeframes with proper robustness validation.

---

## Installation
```bash
# Clone or unzip the project
cd btc_research

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart

### Step 1 — Download all historical data
```bash
python download_all_timeframes.py
```

This will:
- Auto-detect all timeframes supported by Binance
- Download BTC/USDT OHLCV from 2018-01-01 to today
- Save each timeframe to `data/raw/BTCUSDT/<tf>.parquet`
- Save metadata to `data/metadata/BTCUSDT_metadata.json`

### Step 2 — Update data (append new candles only)
```bash
python update_all_timeframes.py
```

Run this daily/weekly to keep your local data fresh.

### Step 3 — Build features
```bash
python build_features.py
```

Or for specific timeframes:
```bash
python build_features.py --timeframes 1h 4h 1d
```

Features are saved to `data/features/BTCUSDT/<tf>_features.parquet`.

### Step 4 — Run research
```bash
# Full research run (all strategies, all timeframes, with walk-forward)
python run_research.py

# Faster: specific timeframes and strategy families
python run_research.py --timeframes 1h 4h 1d --strategies trend_following mean_reversion

# Skip walk-forward (much faster, for quick exploratory runs)
python run_research.py --no-wf
```

---

## Output Structure