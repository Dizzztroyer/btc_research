# BTC Research Platform

A fully local, production-grade BTC/USDT strategy research and backtesting framework.

---

## What This Does

- Downloads BTC/USDT OHLCV candles for **all exchange-supported timeframes** and stores them locally as Parquet files
- Computes a comprehensive set of technical features and caches them locally
- Backtests **6 strategy families** (trend, mean-reversion, breakout, structure, regime, ensemble) across all timeframes
- Optimises parameters with IS/OOS splitting to avoid overfitting
- Validates results with **walk-forward testing**
- Runs **stress tests** with inflated fees/slippage
- Produces **year-by-year breakdowns**, ranking tables, equity curve plots, and an HTML report
- Ranks strategies by **robustness**, not just raw return

---

## Project Structure

```
btc_research/
├── config/
│   └── config.yaml              ← All settings live here
├── data/
│   ├── raw/BTCUSDT/             ← Raw OHLCV parquet files (one per timeframe)
│   ├── features/BTCUSDT/        ← Feature-enriched parquet files
│   └── metadata/                ← Download metadata JSON
├── src/
│   ├── data/                    ← Download + validation logic
│   ├── features/                ← Feature computation
│   ├── backtest/                ← Simulation engine + metrics
│   ├── strategies/              ← All strategy families
│   ├── research/                ← Optimiser, walk-forward, reporter
│   └── utils/                   ← Config loader, logger
├── outputs/
│   ├── reports/                 ← HTML report
│   ├── rankings/                ← CSV result tables
│   ├── plots/                   ← Equity curves, heatmaps
│   └── logs/                    ← Run logs
├── download_all_timeframes.py
├── update_all_timeframes.py
├── build_features.py
├── run_research.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone or copy the project folder
cd btc_research

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Python 3.9+ required.**

---

## Step-by-Step Usage

### 1. Download Data

Downloads all timeframes supported by Binance (auto-detected):

```bash
python download_all_timeframes.py
```

Options:
```bash
# Force full re-download even if files exist
python download_all_timeframes.py --force

# Download only specific timeframes
python download_all_timeframes.py --tf 1h 4h 1d

# Use a different config file
python download_all_timeframes.py --config config/config.yaml
```

After running, you will see files like:
```
data/raw/BTCUSDT/1m.parquet
data/raw/BTCUSDT/5m.parquet
data/raw/BTCUSDT/1h.parquet
...
data/metadata/BTCUSDT_metadata.json
```

### 2. Update Data (Incremental)

Fetches only new candles since the last stored timestamp:

```bash
python update_all_timeframes.py

# Update only specific timeframes
python update_all_timeframes.py --tf 1h 4h
```

### 3. Build Features

Reads raw parquet files and computes all technical features:

```bash
python build_features.py

# Build specific timeframes only
python build_features.py --tf 1h 4h 1d

# Force rebuild even if feature files exist
python build_features.py --force
```

Feature files are saved to `data/features/BTCUSDT/<tf>_features.parquet`.

Features include:
- Returns and log returns (1, 2, 5, 10 bar)
- Candle anatomy (body size, wicks, range)
- ATR (7, 14, 21 period)
- Rolling volatility (10, 20, 40 period)
- EMA (5, 8, 13, 21, 34, 55, 89, 144, 200)
- SMA (10, 20, 50, 100, 200)
- RSI (7, 14, 21)
- ADX (10, 14)
- Bollinger Bands (multiple parameter sets)
- Donchian Channels (10, 20, 40, 55)
- Breakout distances
- Rolling high/low breaks
- Volume z-score
- Squeeze indicator (BB inside KC)
- Calendar features (hour, day of week, month, cyclical encoding)
- Session markers (Asia, Europe, US)
- EMA crossover signals
- Market regime labels (trend, range, high vol, neutral)

### 4. Run Research

Runs all strategies on all timeframes with full optimisation and validation:

```bash
python run_research.py
```

Options:
```bash
# Restrict to specific timeframes
python run_research.py --tf 1h 4h 1d

# Restrict to specific strategy families
python run_research.py --strategy trend mean_reversion

# Long-only
python run_research.py --direction long

# Skip walk-forward (much faster, useful for quick exploration)
python run_research.py --no-wf

# Skip stress tests
python run_research.py --no-stress

# Combine options
python run_research.py --tf 4h --strategy trend --direction long
```

### 5. Run One Strategy on One Timeframe

You can run a focused research job:

```bash
python run_research.py --tf 4h --strategy trend --no-wf --no-stress
```

Or interact programmatically:

```python
from src.utils.config_loader import load_config
from src.features.feature_engine import FeatureEngine
from src.backtest.engine import BacktestEngine, SimConfig
from src.strategies.trend import EMACrossStrategy

cfg      = load_config()
engine   = FeatureEngine(cfg)
df       = engine.load("4h")

strategy = EMACrossStrategy()
params   = {"fast": 13, "slow": 55, "atr_len": 14, "sl_mult": 2.0, "tp_mult": 3.0}
df_sig   = strategy.generate_signals(df, params)

sim = SimConfig(fees=0.00075, slippage=0.0003, direction="both")
bt  = BacktestEngine(sim)
res = bt.run(df_sig, strategy_name="ema_cross", timeframe="4h", params=params)

print(res.metrics)
print(res.yearly)
```

---

## Where Outputs Are Saved

```
outputs/
├── reports/research_report.html    ← Main HTML report (open in browser)
├── rankings/
│   ├── all_results.csv             ← All tested combinations ranked by robustness
│   ├── wf_summary.csv              ← Walk-forward results per strategy/timeframe
│   ├── stress_test.csv             ← Stress test results
│   └── yearly_best_strategy.csv    ← Year-by-year breakdown of best strategy
├── plots/
│   ├── equity_*.png                ← Equity + drawdown curves
│   └── heatmap_*.png               ← Strategy × timeframe heatmaps
└── logs/
    ├── research.log
    ├── download.log
    ├── update.log
    └── features.log
```

Open `outputs/reports/research_report.html` in any browser for a summary.

---

## How Validation Works

### IS/OOS Split

The data is split chronologically into three non-overlapping segments:
- **In-sample (IS)**: 60% — used for parameter optimisation
- **Out-of-sample (OOS)**: 20% — used to evaluate candidate parameter sets
- **Test**: 20% — held out; used only for stress tests

Parameters are selected by maximising a **robustness score** on the OOS period,
not IS performance. The robustness score combines:
- OOS profit factor (×2 weight)
- OOS Sharpe ratio (×1.5 weight)
- IS→OOS Sharpe decay ratio (×1 weight)
- OOS max drawdown penalty (×0.5 weight)

### Walk-Forward Validation

The full dataset is split into N sequential windows (default: 5).
For each window:
1. Optimise all parameters on the IS portion of that window
2. Run the best parameters on the OOS portion of that window (never seen during optimisation)
3. Concatenate all OOS portions → walk-forward equity curve

This prevents cherry-picking a lucky parameter set that happened to work on one period.

### Rejection Filters

Strategies are **automatically rejected** if:
- OOS trade count < `min_trades` (default: 30)
- OOS profit factor < `min_profit_factor` (default: 1.0)
- Robustness score < 0

Rejected strategies are listed in the HTML report and in `all_results.csv`.

### Stress Test

The best parameters are re-run with:
- Fees × `stress_fee_mult` (default: 2×)
- Slippage × `stress_slip_mult` (default: 2×)

Strategies that collapse under stress are flagged.

---

## Strategy Families

| Family | Strategies |
|--------|-----------|
| trend | EMACross, DonchianBreakout, PullbackTrend |
| mean_reversion | RSIReversion, BollingerReversion, EMADeviation |
| breakout | SqueezeBreakout, ConsolidationBreakout, ATRExpansionBreakout |
| structure | SwingBreakout, LiquiditySweep, BOS |
| regime | RegimeSwitch (routes between trend and MR by ADX) |
| ensemble | MajorityVote, WeightedEnsemble |

All strategies are **fully algorithmic** — every rule has an explicit, objective, reproducible definition. No visual confirmation, no discretionary logic.

---

## Configuration

Edit `config/config.yaml` to change:

```yaml
exchange: binance          # ccxt exchange id
symbol: BTC/USDT
start_date: "2018-01-01"
end_date: null             # null = up to now

# null = auto-detect all supported timeframes
enabled_timeframes: null

fees: 0.00075              # 0.075% per side
slippage: 0.0003           # 0.03% per side
leverage: 1.0
risk_per_trade: 0.01       # 1% of equity per trade

validation:
  is_ratio: 0.60
  oos_ratio: 0.20
  wf_windows: 5
  min_trades: 30
  min_profit_factor: 1.0
  stress_fee_mult: 2.0
  stress_slip_mult: 2.0
```

---

## Adding a New Strategy

1. Create a new class in `src/strategies/` inheriting from `BaseStrategy`
2. Implement `param_grid()` and `generate_signals()`
3. Register it in `src/strategies/__init__.py` under the appropriate family
4. It will automatically appear in all research runs

```python
from src.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def param_grid(self):
        return [{"fast": 10, "slow": 50}, {"fast": 20, "slow": 100}]

    def generate_signals(self, df, params):
        df = df.copy()
        self._add_signal_columns(df)
        # ... your logic here ...
        # Must set df["signal"] (1, -1, 0), df["sl_price"], df["tp_price"]
        return df
```

---

## Adding a New Symbol

Change `symbol` and `symbol_file` in `config.yaml`:

```yaml
symbol: ETH/USDT
symbol_file: ETHUSDT
```

Then re-run the full pipeline (download → features → research).

---

## Known Limitations

1. **1m data volume**: Downloading full history at 1-minute resolution is very large (~500MB+ uncompressed) and slow. Consider restricting with `--tf` if you don't need 1m.

2. **Exchange rate limits**: The downloader respects ccxt rate limits, but very aggressive polling may still trigger temporary bans. The exponential backoff handles most cases.

3. **Ensemble optimisation**: The ensemble strategies use fixed default sub-strategy parameters rather than jointly optimising all sub-parameters (which would be combinatorially explosive). For best results, first find optimal sub-strategy parameters, then run the ensemble.

4. **Structure strategies on 1m**: Swing-high/low detection and BOS work best on 1h+ timeframes. On 1m, noise dominates. Results below the 1h level for structure strategies should be treated with extra scepticism.

5. **Walk-forward on small datasets**: If you restrict to a narrow date range, walk-forward windows may be too small. Increase `start_date` range or reduce `wf_windows`.

6. **No live trading**: This platform is research-only. Do not connect it to a live exchange account.

---

## Research Principles Followed

- All timestamps in UTC
- No lookahead bias: signals on bar N act on bar N+1 open
- No data leakage between IS and OOS
- Raw candles, features, and results stored separately
- Strategies ranked by robustness (OOS metrics), not IS max profit
- Every strategy rule is explicit and reproducible
- Realistic fees (0.075%) and slippage (0.03%) included in every backtest
