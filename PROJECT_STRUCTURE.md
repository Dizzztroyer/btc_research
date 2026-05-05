# BTC Research Platform — Project Structure

## Entry Point Scripts (`run_*.py`)

| Script | Purpose | Status |
|---|---|---|
| `run_research.py` | Main grid search across strategies × TFs × params. Outputs `all_results.csv` + `wf_summary.csv`. Supports `--reset-checkpoint`, `--random-n`, `--tf`, `--strategy` | Active |
| `run_enhanced_backtest.py` | Compares BASE vs ML_SIZE for given strategy/TF. Outputs `enhanced_backtest_summary.csv` + `ml_sizer_diagnostics.csv` | Active |
| `run_atr_regime_study.py` | Tests ATR regime modifier. Versions: BASE, ATR_SIZE, ML_SIZE, ML_ATR. Key flags: `--low-mult`, `--high-mult`, `--no-filter` | Active |
| `run_wf_validation.py` | Walk-forward validation of BASE vs ML_SIZE vs ML_ATR across N sequential windows. Expanding IS. Outputs per-window and aggregate CSV | Active |
| `run_volume_study.py` | Volume confirmation study. Versions: BASE, ML_SIZE, VOL_CONFIRM, ML_VOL. Key flag: `--vol-thresh` | Active |
| `run_session_study.py` | Session filter + sizing study. Concluded: no edge. Kept for reference | Concluded |
| `run_ml_enhanced.py` | ML sizer with extended feature set (51 features) + strict quality gates | Concluded |
| `run_ml_filter.py` | Legacy binary entry/skip ML filter. Concluded: harmful | Deprecated |
| `run_inversion_study.py` | Tests if losing strategies become profitable when signal is inverted | Concluded |
| `run_multiples_of_3.py` | Research on timeframes divisible by 3 (3h, 3d, 6h, 12h) | Concluded |
| `aggregate_3h.py` | Resamples 1h OHLCV → 3h. Run before `run_research.py --tf 3h` | Utility |

## Source Modules (`src/`)

### `src/backtest/`
- **`engine.py`** — `BacktestEngine.run()` executes simulation. `SimConfig` holds fees, slippage, risk_per_trade, direction. `BacktestResult` contains metrics, trades DataFrame, equity/drawdown series, yearly breakdown.
- **`fast_engine.py`** — Numba-compiled inner loop for parallel param search. Called by `optimizer.py`.
- **`metrics.py`** — `compute_metrics()` computes PF, Sharpe, Sortino, Calmar, MDD, win_rate, expectancy. `yearly_breakdown()` splits by calendar year.

### `src/data/`
- **`downloader.py`** — Downloads OHLCV from Binance via ccxt. Handles pagination, gaps, incremental updates. Output: `data/raw/BTCUSDT/{tf}.parquet`.
- **`validator.py`** — Checks for gaps, duplicates, OHLC consistency.

### `src/features/`
- **`feature_engine.py`** — `FeatureEngine.build(tf)` reads raw parquet, computes 140 features, saves to `data/features/BTCUSDT/{tf}_features.parquet`. `FeatureEngine.load(tf)` returns feature DataFrame (builds if missing). Features include: log returns (1/2/5/10 bars), ATR (7/14/21), EMA (5–200), RSI (7/14/21), ADX (10/14), Donchian channels (10/20/40/55), Bollinger bands, vol_zscore (10/20/40), squeeze, range_ratio, session one-hot, regime labels.

### `src/ml/`
- **`ml_sizer_v2.py`** — `MLSizerV2.fit()` trains XGBoost on IS trade outcomes with rank-normalized scoring and sample weighting. `MLSizerV2.predict()` returns per-bar size multiplier Series. `score_to_mult()` maps rank [0,1] → mult [0.5, 1.5]. Auto-disables if std(mult) < 0.05 or trades < 150.
- **`dynamic_engine.py`** — `run_dynamic()` runs backtest with per-bar `size_mults` Series. Identical logic to `BacktestEngine` but applies `base_risk × mult` per trade. Takes `ml_mults × atr_mults` combined Series.
- **`ml_filter.py`** — Legacy binary filter (enter/skip). Not used. Kept for reference.

### `src/research/`
- **`optimizer.py`** — `optimize()` runs parallel param grid search. For each param set: generate signals → backtest → OOS filter (PF≥1.0, trades≥30, robustness>0). Returns sorted results.
- **`reporter.py`** — Saves `all_results.csv`, `wf_summary.csv`, `yearly_best_strategy.csv`, HTML report.
- **`walk_forward.py`** — `walk_forward()` runs expanding-window WF validation. Called after OOS acceptance.

### `src/strategies/`
All strategies inherit from `BaseStrategy` with `generate_signals(df, params) → df_with_signals`.

- **`structure.py`** — `SwingBreakoutStrategy` ← PRIMARY. Detects swing highs/lows (left/right bars), enters on close breakout. Also: `LiquiditySweepStrategy`, `BOSStrategy`.
- **`trend.py`** — `DonchianBreakoutStrategy` (Donchian channel breakout), `EMACrossStrategy`, `PullbackTrendStrategy`.
- **`breakout.py`** — `SqueezeBreakoutStrategy`, `ConsolidationBreakoutStrategy`, `ATRExpansionBreakoutStrategy`.
- **`mean_reversion.py`** — RSI, Bollinger, EMA deviation. All rejected on BTC.
- **`ensemble.py`** — `WeightedEnsemble`, `MajorityVote`. Marginal results.
- **`regime.py`** — `RegimeSwitchStrategy`. Rejected (overfit).

### `src/utils/`
- **`config_loader.py`** — `load_config(path)` → `Config` dataclass with all paths, trading params, IS/OOS ratios.
- **`logger.py`** — Structured logger with timestamped format, optional file output.

## Data Flow

```
Binance API
    ↓ download_all_timeframes.py / downloader.py
data/raw/BTCUSDT/{tf}.parquet
    ↓ feature_engine.py
data/features/BTCUSDT/{tf}_features.parquet
    ↓ run_research.py / optimizer.py
outputs/rankings/all_results.csv
outputs/rankings/wf_summary.csv
    ↓ run_enhanced_backtest.py + run_atr_regime_study.py
        uses: ml_sizer_v2.py (train on IS)
              dynamic_engine.py (run on OOS with per-bar mults)
outputs/rankings/enhanced_backtest_summary.csv
outputs/rankings/atr_regime_results.csv
    ↓ run_wf_validation.py
outputs/rankings/wf_validation_detail.csv
outputs/rankings/wf_validation_summary.csv
```

## Key Config Values (`config/config.yaml`)

```yaml
fees: 0.00075          # 0.075% per trade (Binance taker)
slippage: 0.0003       # 0.03% per fill
risk_per_trade: 0.01   # 1% capital at risk per trade
leverage: 1.0          # No leverage in base config
validation:
  is_ratio: 0.60       # 60% IS, 20% OOS, 20% test
  oos_ratio: 0.20
```