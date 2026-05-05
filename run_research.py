#!/usr/bin/env python3
"""
run_research.py
───────────────
Main research orchestrator.

Speed improvements vs v1:
  1. Numba JIT backtest loop         → 15-50× faster per backtest
  2. Parallel param search (joblib)  → N_cores × faster per strategy
  3. Checkpoint/resume               → survives reboots
  4. Random-N sampling               → 50 params instead of 1458
  5. Partial results saved instantly → nothing lost on crash

Typical timing with all improvements:
  Full run (6 families × 12 TF, random-n=50, no-wf) → 20-60 min
  vs original: days

Usage
─────
    python run_research.py --tf 2h 4h 6h 8h 12h 1d --no-wf --no-stress
    python run_research.py --tf 4h 8h 12h 1d --random-n 30 --no-wf
    python run_research.py --reset-checkpoint
    python run_research.py --direction long
    python run_research.py --jobs 4   # explicit core count
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config, Config
from src.utils.logger import get_logger
from src.features.feature_engine import FeatureEngine
from src.backtest.engine import BacktestEngine, SimConfig
from src.backtest.metrics import yearly_breakdown
from src.strategies import STRATEGY_FAMILIES
from src.strategies.base import BaseStrategy
from src.research.optimizer import ParameterOptimizer
from src.research.walk_forward import WalkForwardEngine
from src.research.reporter import Reporter

logger = get_logger(__name__, log_file=Path("outputs/logs/research.log"))

CHECKPOINT_FILE      = Path("outputs/logs/checkpoint.json")
PARTIAL_RESULTS_FILE = Path("outputs/rankings/all_results_partial.csv")


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        try:
            data = json.loads(CHECKPOINT_FILE.read_text())
            done = set(tuple(x) for x in data.get("done", []))
            logger.info(f"Checkpoint: {len(done)} runs already done — skipping")
            return done
        except Exception:
            pass
    return set()


def _save_checkpoint(done: set) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps({"done": [list(x) for x in sorted(done)]}, indent=2)
    )


def _append_partial(df: pd.DataFrame) -> None:
    PARTIAL_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PARTIAL_RESULTS_FILE.exists():
        try:
            existing = pd.read_csv(PARTIAL_RESULTS_FILE)
            combined = pd.concat([existing, df], ignore_index=True)
        except Exception:
            combined = df
    else:
        combined = df
    combined.to_csv(PARTIAL_RESULTS_FILE, index=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC Research — parallel + checkpoint")
    p.add_argument("--config",    default="config/config.yaml")
    p.add_argument("--tf",        nargs="+", default=None)
    p.add_argument("--strategy",  nargs="+", default=None)
    p.add_argument("--direction", default="both", choices=["long","short","both"])
    p.add_argument("--no-wf",     action="store_true")
    p.add_argument("--no-stress", action="store_true")
    p.add_argument("--top-n",     type=int, default=None)
    p.add_argument("--random-n",  type=int, default=50,
                   help="Random param sets per run (default 50, 0=full grid)")
    p.add_argument("--jobs",      type=int, default=-1,
                   help="Parallel workers for param search (-1=all cores)")
    p.add_argument("--reset-checkpoint", action="store_true")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _available_timeframes(cfg: Config, requested: Optional[List[str]]) -> List[str]:
    paths = sorted(cfg.features_dir.glob("*_features.parquet"))
    avail = [p.stem.replace("_features", "") for p in paths]
    if not avail:
        avail = [p.stem for p in sorted(cfg.raw_dir.glob("*.parquet"))]
    if requested:
        missing = [t for t in requested if t not in avail]
        if missing:
            logger.warning(f"Not found locally: {missing}")
        return [t for t in requested if t in avail]
    return avail


def _load_features(cfg: Config, timeframe: str) -> pd.DataFrame:
    df = FeatureEngine(cfg).load(timeframe)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if cfg.start_date:
        df = df[df["timestamp"] >= pd.Timestamp(cfg.start_date, tz="UTC")]
    if cfg.end_date:
        df = df[df["timestamp"] <= pd.Timestamp(cfg.end_date, tz="UTC")]
    return df.reset_index(drop=True)


def _stress_test(strategy, df, params, cfg, direction) -> dict:
    sim = SimConfig(
        fees           = cfg.fees    * cfg.validation.stress_fee_mult,
        slippage       = cfg.slippage * cfg.validation.stress_slip_mult,
        leverage       = cfg.leverage,
        risk_per_trade = cfg.risk_per_trade,
        direction      = direction,
    )
    n       = len(df)
    ts      = int(n * (cfg.validation.is_ratio + cfg.validation.oos_ratio))
    df_test = df.iloc[ts:] if ts < n and n - ts > 30 else df
    try:
        df_sig = strategy.generate_signals(df_test.copy(), params)
        return BacktestEngine(sim).run(df_sig, strategy.name, "", params).metrics
    except Exception:
        return {}


def _extract_best_params(opt_df: pd.DataFrame) -> dict:
    if opt_df.empty:
        return {}
    row = opt_df.iloc[0]
    return {c[2:]: row[c] for c in row.index if c.startswith("p_")}


def _estimate_bpy(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 365.0
    ts    = pd.to_datetime(df["timestamp"])
    total = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    return 365.25 * 86_400 / (total / (len(ts)-1)) if total > 0 else 365.0


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_research(
    cfg:              Config,
    timeframes:       List[str],
    families:         List[str],
    direction:        str,
    skip_wf:          bool,
    skip_stress:      bool,
    top_n:            int,
    random_n:         int,
    n_jobs:           int,
    reset_checkpoint: bool,
) -> None:

    optimizer = ParameterOptimizer(cfg)
    wf_engine = WalkForwardEngine(cfg)
    reporter  = Reporter(cfg)

    done_set = set()
    if not reset_checkpoint:
        done_set = _load_checkpoint()
    elif CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint reset")

    all_results:  List[pd.DataFrame] = []
    wf_rows:      List[dict] = []
    stress_rows:  List[dict] = []
    equity_plots: List[Path] = []
    rejected:     List[dict] = []

    best_yearly     = pd.DataFrame()
    best_robustness = -999999.0

    if PARTIAL_RESULTS_FILE.exists() and not reset_checkpoint:
        try:
            prev = pd.read_csv(PARTIAL_RESULTS_FILE)
            all_results.append(prev)
            logger.info(f"Loaded {len(prev)} rows from partial results")
        except Exception:
            pass

    total = sum(len(STRATEGY_FAMILIES[f]) for f in families if f in STRATEGY_FAMILIES) \
            * len(timeframes)

    # Log system info
    import multiprocessing
    n_cpu = multiprocessing.cpu_count()
    from src.backtest.fast_engine import NUMBA_AVAILABLE
    try:
        from joblib import Parallel
        joblib_ok = True
    except ImportError:
        joblib_ok = False

    logger.info(f"CPU cores     : {n_cpu}")
    logger.info(f"Numba JIT     : {'ON' if NUMBA_AVAILABLE else 'OFF (pip install numba)'}")
    logger.info(f"Joblib        : {'ON' if joblib_ok else 'OFF (pip install joblib)'}")
    logger.info(f"Research plan : {len(families)} families × {len(timeframes)} TFs = {total} runs")
    logger.info(f"Random-N      : {random_n if random_n > 0 else 'FULL GRID'}")
    logger.info(f"Jobs/run      : {n_jobs} ({'all cores' if n_jobs==-1 else str(n_jobs)})")
    logger.info(f"Already done  : {len(done_set)}")

    # Warm up numba on first run (avoids first-call delay mid-loop)
    if NUMBA_AVAILABLE:
        logger.info("Warming up Numba JIT (one-time, ~5s) …")
        from src.backtest.fast_engine import get_fast_loop
        get_fast_loop()
        logger.info("Numba ready.")

    run_count = 0

    for family_name in families:
        if family_name not in STRATEGY_FAMILIES:
            continue

        for StratClass in STRATEGY_FAMILIES[family_name]:
            strategy = StratClass()
            logger.info(f"\n{'='*60}")
            logger.info(f"Strategy: {strategy.name}  (family: {family_name})")
            logger.info(f"{'='*60}")

            for timeframe in timeframes:
                run_count += 1
                key = (strategy.name, timeframe)

                if key in done_set:
                    logger.info(f"[{run_count}/{total}] SKIP {strategy.name}/{timeframe}")
                    continue

                logger.info(f"\n[{run_count}/{total}] {strategy.name} / {timeframe}")
                t0 = time.time()

                # ── Load features ──────────────────────────────────────────────
                try:
                    df = _load_features(cfg, timeframe)
                except Exception as exc:
                    logger.error(f"  Feature load failed: {exc}")
                    done_set.add(key); _save_checkpoint(done_set)
                    continue

                if len(df) < 200:
                    logger.warning(f"  Only {len(df)} rows — skip")
                    done_set.add(key); _save_checkpoint(done_set)
                    continue

                # ── Parallel param optimisation ────────────────────────────────
                try:
                    rn     = random_n if random_n > 0 else None
                    opt_df = optimizer.optimize(
                        strategy  = strategy,
                        df        = df,
                        timeframe = timeframe,
                        direction = direction,
                        random_n  = rn,
                        n_jobs    = n_jobs,
                    )
                except Exception as exc:
                    logger.error(f"  Optimisation failed: {exc}\n{traceback.format_exc()}")
                    done_set.add(key); _save_checkpoint(done_set)
                    continue

                if opt_df.empty:
                    logger.warning("  No valid results")
                    rejected.append({"strategy": strategy.name, "timeframe": timeframe,
                                     "reason": "empty"})
                    done_set.add(key); _save_checkpoint(done_set)
                    continue

                _append_partial(opt_df)
                all_results.append(opt_df)

                best_params = _extract_best_params(opt_df)
                best_row    = opt_df.iloc[0]
                oos_pf      = best_row.get("oos_pf",     0) or 0
                oos_trades  = best_row.get("oos_trades",  0) or 0
                robustness  = best_row.get("robustness", -9999) or -9999

                # ── Reject ────────────────────────────────────────────────────
                if oos_trades < cfg.validation.min_trades:
                    logger.warning(f"  Rejected: trades={oos_trades} < {cfg.validation.min_trades}")
                    rejected.append({"strategy": strategy.name, "timeframe": timeframe,
                                     "reason": f"trades={oos_trades}<min"})
                    done_set.add(key); _save_checkpoint(done_set)
                    continue

                if oos_pf < cfg.validation.min_profit_factor:
                    logger.warning(f"  Rejected: PF={oos_pf:.2f}")
                    rejected.append({"strategy": strategy.name, "timeframe": timeframe,
                                     "reason": f"PF={oos_pf:.2f}<min"})
                    done_set.add(key); _save_checkpoint(done_set)
                    continue

                logger.info(
                    f"  ✓ Accepted | PF={oos_pf:.2f} | "
                    f"Sharpe={best_row.get('oos_sharpe', float('nan')):.2f} | "
                    f"Robustness={robustness:.2f}"
                )

                # ── Walk-forward ───────────────────────────────────────────────
                if not skip_wf:
                    try:
                        wf  = wf_engine.run(strategy, df, timeframe, direction)
                        wm  = wf.combined_metrics
                        wf_rows.append({
                            "strategy":  strategy.name, "timeframe": timeframe,
                            "wf_sharpe": wm.get("sharpe",        np.nan),
                            "wf_pf":     wm.get("profit_factor", np.nan),
                            "wf_return": wm.get("total_return",  np.nan),
                            "wf_mdd":    wm.get("max_drawdown",  np.nan),
                            "wf_trades": wm.get("trade_count",   0),
                            "n_windows": len(wf.windows),
                        })
                    except Exception as exc:
                        logger.warning(f"  WF failed: {exc}")

                # ── Stress test ────────────────────────────────────────────────
                if not skip_stress:
                    try:
                        sm = _stress_test(strategy, df, best_params, cfg, direction)
                        stress_rows.append({
                            "strategy": strategy.name, "timeframe": timeframe,
                            "stress_sharpe": sm.get("sharpe",        np.nan),
                            "stress_pf":     sm.get("profit_factor", np.nan),
                            "stress_return": sm.get("total_return",  np.nan),
                            "stress_mdd":    sm.get("max_drawdown",  np.nan),
                            "stress_trades": sm.get("trade_count",   0),
                        })
                    except Exception as exc:
                        logger.warning(f"  Stress failed: {exc}")

                # ── Equity plot ────────────────────────────────────────────────
                if cfg.reporting.plot_equity_curves:
                    try:
                        sim    = SimConfig(fees=cfg.fees, slippage=cfg.slippage,
                                           leverage=cfg.leverage,
                                           risk_per_trade=cfg.risk_per_trade,
                                           direction=direction)
                        eng    = BacktestEngine(sim)
                        df_sig = strategy.generate_signals(df.copy(), best_params)
                        res    = eng.run(df_sig, strategy.name, timeframe, best_params)
                        if res.ok:
                            p = reporter.plot_equity_curve(
                                equity     = res.equity,
                                drawdown   = res.drawdown,
                                name       = f"{strategy.name}_{timeframe}",
                                timestamps = df_sig.get("timestamp"),
                            )
                            equity_plots.append(p)
                            if robustness > best_robustness:
                                best_robustness = robustness
                                best_yearly = yearly_breakdown(
                                    trades          = res.trades,
                                    equity          = res.equity,
                                    timestamps      = df_sig["timestamp"],
                                    bars_per_year   = _estimate_bpy(df),
                                    initial_capital = cfg.risk_per_trade,
                                )
                    except Exception as exc:
                        logger.warning(f"  Plot failed: {exc}")

                done_set.add(key)
                _save_checkpoint(done_set)
                logger.info(f"  Done in {time.time()-t0:.1f}s")

    # ── Final reports ─────────────────────────────────────────────────────────
    if not all_results:
        logger.error("No results. Check logs.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    key_cols = ["strategy","timeframe"] + [c for c in combined.columns if c.startswith("p_")]
    combined = (
        combined
        .drop_duplicates(subset=key_cols)
        .sort_values("robustness", ascending=False)
        .reset_index(drop=True)
    )

    rej_df = pd.DataFrame(rejected)
    reporter.save_rankings(combined, "all_results.csv")
    if wf_rows:    reporter.save_wf_summary(wf_rows)
    if stress_rows:reporter.save_stress_summary(stress_rows)
    if not best_yearly.empty:
        reporter.save_yearly_breakdown(best_yearly, "best_strategy")

    heatmaps: List[Path] = []
    if cfg.reporting.plot_heatmaps and "strategy" in combined.columns:
        for metric in ["oos_sharpe", "oos_pf", "oos_return"]:
            try:
                p = reporter.plot_tf_strategy_heatmap(combined, metric=metric)
                if p and p.exists():
                    heatmaps.append(p)
            except Exception:
                pass

    if cfg.reporting.html_report:
        try:
            reporter.generate_html_report(
                all_results=combined, wf_rows=wf_rows,
                best_yearly=best_yearly, rejected=rej_df,
                equity_plots=equity_plots, heatmap_plots=heatmaps, top_n=top_n,
            )
        except Exception as exc:
            logger.error(f"HTML failed: {exc}")

    logger.info("\n" + "="*60)
    logger.info("RESEARCH COMPLETE")
    logger.info("="*60)
    logger.info(f"Results   : {len(combined)}")
    logger.info(f"Rejected  : {len(rejected)}")
    logger.info(f"WF runs   : {len(wf_rows)}")
    cols = [c for c in ["strategy","timeframe","oos_sharpe","oos_pf",
                         "oos_return","oos_drawdown","oos_trades","robustness"]
            if c in combined.columns]
    with pd.option_context("display.max_columns", None, "display.width", 130):
        logger.info("\nTOP 10:\n" + combined[cols].head(10).to_string(index=False))
    logger.info(f"\nOutputs → {cfg.output_dir}")


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(Path(args.config))

    timeframes = _available_timeframes(cfg, args.tf)
    if not timeframes:
        logger.error("No feature files. Run build_features.py first.")
        sys.exit(1)

    families = args.strategy or cfg.enabled_strategies
    families = [f for f in families if f in STRATEGY_FAMILIES]
    if not families:
        logger.error("No valid families.")
        sys.exit(1)

    logger.info(f"Timeframes : {timeframes}")
    logger.info(f"Families   : {families}")
    logger.info(f"Direction  : {args.direction}")
    logger.info(f"WF         : {'off' if args.no_wf else 'on'}")
    logger.info(f"Stress     : {'off' if args.no_stress else 'on'}")
    logger.info(f"Random-N   : {args.random_n}")
    logger.info(f"Jobs       : {args.jobs}")

    run_research(
        cfg              = cfg,
        timeframes       = timeframes,
        families         = families,
        direction        = args.direction,
        skip_wf          = args.no_wf,
        skip_stress      = args.no_stress,
        top_n            = args.top_n or cfg.reporting.top_n,
        random_n         = args.random_n,
        n_jobs           = args.jobs,
        reset_checkpoint = args.reset_checkpoint,
    )


if __name__ == "__main__":
    main()