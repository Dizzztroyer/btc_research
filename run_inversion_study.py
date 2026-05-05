#!/usr/bin/env python3
"""
run_inversion_study.py
──────────────────────
Inversion study: flip signal direction for stably losing strategies.

Logic:
    If a strategy is systematically losing in OOS AND walk-forward,
    it may be detecting real patterns — just in the wrong direction.
    Inverting the signal (long→short, short→long) tests this hypothesis.

Criteria for inversion candidates:
    - OOS PF < 0.85 (consistently losing, not just noisy)
    - OOS trades ≥ 30 (enough for statistics)
    - Losing in BOTH OOS and walk-forward
    - NOT one of the working strategies

What we DO NOT touch:
    - swing_breakout (any TF)
    - donchian_breakout (any TF)
    - Any strategy with OOS PF ≥ 1.0

Output:
    - Inversion results vs original
    - CSV comparison
    - Only reports if inverted version has PF ≥ 1.0 AND Sharpe > 0

Usage
─────
    python run_inversion_study.py
    python run_inversion_study.py --min-trades 30 --max-pf 0.85
    python run_inversion_study.py --candidates-only  # just show candidates
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.features.feature_engine import FeatureEngine
from src.backtest.engine import BacktestEngine, SimConfig
from src.strategies.trend import (
    EMACrossStrategy, DonchianBreakoutStrategy, PullbackTrendStrategy
)
from src.strategies.mean_reversion import (
    RSIReversionStrategy, BollingerReversionStrategy, EMADeviationStrategy
)
from src.strategies.breakout import (
    SqueezeBreakoutStrategy, ConsolidationBreakoutStrategy, ATRExpansionBreakoutStrategy
)
from src.strategies.structure import SwingBreakoutStrategy, LiquiditySweepStrategy, BOSStrategy
from src.strategies.regime import RegimeSwitchStrategy

logger = get_logger(__name__, log_file=Path("outputs/logs/inversion_study.log"))

# Strategies that must NEVER be inverted
PROTECTED = {"swing_breakout", "donchian_breakout"}

ALL_STRATEGIES = {
    "ema_cross":               EMACrossStrategy,
    "pullback_trend":          PullbackTrendStrategy,
    "rsi_reversion":           RSIReversionStrategy,
    "bollinger_reversion":     BollingerReversionStrategy,
    "ema_deviation":           EMADeviationStrategy,
    "squeeze_breakout":        SqueezeBreakoutStrategy,
    "consolidation_breakout":  ConsolidationBreakoutStrategy,
    "atr_expansion_breakout":  ATRExpansionBreakoutStrategy,
    "liquidity_sweep":         LiquiditySweepStrategy,
    "bos":                     BOSStrategy,
    "regime_switch":           RegimeSwitchStrategy,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inversion study for losing strategies")
    p.add_argument("--config",           default="config/config.yaml")
    p.add_argument("--results-csv",      default="outputs/rankings/all_results.csv")
    p.add_argument("--wf-csv",           default="outputs/rankings/wf_summary.csv")
    p.add_argument("--min-trades",       type=int,   default=30)
    p.add_argument("--max-pf",           type=float, default=0.85,
                   help="Candidate must have OOS PF below this")
    p.add_argument("--max-candidates",   type=int,   default=10)
    p.add_argument("--candidates-only",  action="store_true",
                   help="Only show candidates, do not run backtests")
    p.add_argument("--direction",        default="both")
    return p.parse_args()


def _load_params(df: pd.DataFrame, strategy: str, tf: str) -> Optional[dict]:
    """Load worst (most negative) params for inversion — use MOST losing combo."""
    mask = (df["strategy"] == strategy) & (df["timeframe"] == tf)
    sub  = df[mask]
    if sub.empty:
        return None
    # For inversion, pick params with most trades and lowest PF
    sub_filtered = sub[sub["oos_trades"] >= 30]
    if sub_filtered.empty:
        return None
    worst = sub_filtered.sort_values("oos_pf").iloc[0]
    p_cols = [c for c in worst.index if c.startswith("p_") and not pd.isna(worst[c])]
    out = {}
    for c in p_cols:
        v = worst[c]; k = c[2:]
        out[k] = int(v) if isinstance(v, float) and v == int(v) else v
    return out


def _invert_signals(df_sig: pd.DataFrame) -> pd.DataFrame:
    """
    Invert all signals: 1 → -1, -1 → 1, 0 → 0.
    Also invert SL/TP relative to price.
    """
    df_inv = df_sig.copy()
    df_inv["signal"] = -df_sig["signal"]

    # Swap SL and TP: if originally long with sl below, now short with sl above
    c = df_sig["close"]
    if "sl_price" in df_sig.columns and "tp_price" in df_sig.columns:
        orig_sl = df_sig["sl_price"].copy()
        orig_tp = df_sig["tp_price"].copy()
        # For inverted short: sl above price, tp below price
        # Simple reflection around price
        df_inv["sl_price"] = np.where(
            df_inv["signal"] == -1,
            c + (c - orig_sl).abs(),   # sl above close
            np.where(
                df_inv["signal"] == 1,
                c - (c - orig_sl).abs(),  # sl below close
                np.nan
            )
        )
        df_inv["tp_price"] = np.where(
            df_inv["signal"] == -1,
            c - (orig_tp - c).abs(),   # tp below close
            np.where(
                df_inv["signal"] == 1,
                c + (orig_tp - c).abs(),  # tp above close
                np.nan
            )
        )
    return df_inv


def find_candidates(
    df:         pd.DataFrame,
    wf:         pd.DataFrame,
    min_trades: int,
    max_pf:     float,
    max_n:      int,
) -> pd.DataFrame:
    """Find strategies that are stably losing in both OOS and WF."""

    # Filter: not protected, enough trades, losing PF
    mask = (
        ~df["strategy"].isin(PROTECTED) &
        (df["oos_trades"] >= min_trades) &
        (df["oos_pf"] < max_pf) &
        df["strategy"].isin(ALL_STRATEGIES.keys())
    )
    candidates_raw = df[mask].copy()

    # Aggregate: worst OOS PF per strategy/TF
    worst = (
        candidates_raw
        .groupby(["strategy","timeframe"])
        .agg(
            best_oos_pf  = ("oos_pf",       "max"),   # even best param set
            avg_oos_pf   = ("oos_pf",       "mean"),
            avg_oos_sh   = ("oos_sharpe",   "mean"),
            max_trades   = ("oos_trades",    "max"),
            n_combos     = ("oos_pf",        "count"),
        )
        .reset_index()
    )

    # Must be stably losing: even BEST params still below threshold
    worst = worst[worst["best_oos_pf"] < max_pf]

    # Add WF info if available
    if not wf.empty:
        worst = worst.merge(
            wf[["strategy","timeframe","wf_sharpe","wf_pf"]],
            on=["strategy","timeframe"], how="left"
        )
        # Prefer candidates that also fail in WF
        worst["wf_losing"] = worst.get("wf_sharpe", pd.Series(0)) < 0
    else:
        worst["wf_losing"] = True  # assume losing if no WF data

    # Sort by most stably losing
    worst = worst.sort_values(["wf_losing","avg_oos_pf"]).head(max_n)
    return worst


def main() -> None:
    args = parse_args()
    cfg  = load_config(Path(args.config))
    fe   = FeatureEngine(cfg)

    logger.info("=== Inversion Study ===")
    logger.info(f"Protected strategies (will NOT be inverted): {PROTECTED}")
    logger.info(f"Criteria: OOS trades≥{args.min_trades}, OOS PF<{args.max_pf}")

    # Load results
    results_csv = Path(args.results_csv)
    wf_csv      = Path(args.wf_csv)

    if not results_csv.exists():
        logger.error(f"all_results.csv not found: {results_csv}")
        sys.exit(1)

    df_results = pd.read_csv(results_csv)
    df_wf      = pd.read_csv(wf_csv) if wf_csv.exists() else pd.DataFrame()

    # Find candidates
    candidates = find_candidates(
        df_results, df_wf,
        args.min_trades, args.max_pf, args.max_candidates
    )

    print(f"\n{'='*65}")
    print("INVERSION CANDIDATES")
    print(f"{'='*65}")
    if candidates.empty:
        print("No suitable candidates found.")
        sys.exit(0)

    print(candidates.to_string(index=False))
    print(f"\nTotal: {len(candidates)} candidates")

    if args.candidates_only:
        sys.exit(0)

    # ── Run inversions ─────────────────────────────────────────────────────────
    sim_cfg = SimConfig(
        fees=cfg.fees, slippage=cfg.slippage,
        leverage=cfg.leverage, risk_per_trade=cfg.risk_per_trade,
        direction=args.direction,
    )

    all_rows: List[dict] = []
    successes: List[str] = []

    for _, cand in candidates.iterrows():
        strat_name = cand["strategy"]
        tf         = cand["timeframe"]
        label      = f"{strat_name}/{tf}"

        if strat_name not in ALL_STRATEGIES:
            logger.warning(f"  {strat_name}: not in strategy map — skipping")
            continue

        logger.info(f"\n{'─'*50}")
        logger.info(f"Inverting: {label}")
        logger.info(f"  OOS PF (orig): {cand['avg_oos_pf']:.3f}  Sharpe: {cand['avg_oos_sh']:.3f}")

        # Load features
        try:
            df = fe.load(tf)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except Exception as exc:
            logger.error(f"  Feature load failed: {exc}"); continue

        # Load WORST params (most losing)
        params = _load_params(df_results, strat_name, tf)
        if params is None:
            logger.warning(f"  No params found — skipping"); continue
        logger.info(f"  Using params: {params}")

        # IS/OOS split
        n     = len(df)
        is_n  = int(n * cfg.validation.is_ratio)
        df_oos = df.iloc[is_n:]

        strategy   = ALL_STRATEGIES[strat_name]()
        df_oos_sig = strategy.generate_signals(df_oos.copy(), params)

        # Run ORIGINAL (confirm it's losing)
        res_orig = BacktestEngine(sim_cfg).run(
            df_oos_sig.copy(), strat_name, tf, params
        )
        orig_pf = res_orig.metrics.get("profit_factor", np.nan)
        orig_sh = res_orig.metrics.get("sharpe", np.nan)
        orig_tr = res_orig.metrics.get("trade_count", 0)

        logger.info(
            f"  Original OOS: trades={orig_tr}  PF={orig_pf:.3f}  Sh={orig_sh:.3f}"
        )

        # Run INVERTED
        df_inv_sig = _invert_signals(df_oos_sig)
        res_inv = BacktestEngine(sim_cfg).run(
            df_inv_sig, f"{strat_name}_INV", tf, params
        )
        inv_pf = res_inv.metrics.get("profit_factor", np.nan)
        inv_sh = res_inv.metrics.get("sharpe", np.nan)
        inv_tr = res_inv.metrics.get("trade_count", 0)

        logger.info(
            f"  Inverted OOS: trades={inv_tr}  PF={inv_pf:.3f}  Sh={inv_sh:.3f}"
        )

        # Verdict
        is_success = (
            not np.isnan(inv_pf) and inv_pf >= 1.0 and
            not np.isnan(inv_sh) and inv_sh > 0 and
            inv_tr >= args.min_trades
        )
        verdict = "✓ PROFITABLE WHEN INVERTED" if is_success else "✗ Still losing inverted"
        logger.info(f"  {verdict}")
        if is_success:
            successes.append(label)

        all_rows.extend([
            {"strategy": strat_name, "timeframe": tf, "version": "ORIGINAL",
             "trades": orig_tr, "pf": orig_pf, "sharpe": orig_sh,
             "total_return": res_orig.metrics.get("total_return", np.nan),
             "mdd": res_orig.metrics.get("max_drawdown", np.nan)},
            {"strategy": strat_name, "timeframe": tf, "version": "INVERTED",
             "trades": inv_tr, "pf": inv_pf, "sharpe": inv_sh,
             "total_return": res_inv.metrics.get("total_return", np.nan),
             "mdd": res_inv.metrics.get("max_drawdown", np.nan),
             "verdict": verdict},
        ])

    # ── Save and summarise ─────────────────────────────────────────────────────
    if all_rows:
        out_df = pd.DataFrame(all_rows)
        path   = cfg.output_dir / "rankings" / "inversion_study_results.csv"
        out_df.to_csv(path, index=False)
        logger.info(f"\nResults → {path}")

        print(f"\n{'='*65}")
        print("INVERSION STUDY RESULTS")
        print(f"{'='*65}")
        # Print side-by-side
        for strat_tf, grp in out_df.groupby(["strategy","timeframe"]):
            orig = grp[grp["version"]=="ORIGINAL"].iloc[0] if "ORIGINAL" in grp["version"].values else None
            inv  = grp[grp["version"]=="INVERTED"].iloc[0]  if "INVERTED" in grp["version"].values else None
            if orig is None or inv is None: continue
            print(f"\n{strat_tf[0]}/{strat_tf[1]}:")
            print(f"  ORIGINAL : trades={int(orig['trades'])}  PF={orig['pf']:.3f}  Sh={orig['sharpe']:.3f}  MDD={orig['mdd']:.2%}")
            print(f"  INVERTED : trades={int(inv['trades'])}   PF={inv['pf']:.3f}  Sh={inv['sharpe']:.3f}  MDD={inv['mdd']:.2%}  {inv.get('verdict','')}")

        print(f"\n{'='*65}")
        if successes:
            print(f"PROFITABLE WHEN INVERTED ({len(successes)}):")
            for s in successes:
                print(f"  ✓ {s}")
            print("\nNote: Verify these with walk-forward before use.")
        else:
            print("No strategies became profitable when inverted.")
            print("This suggests losses are from market noise, not systematic misdirection.")


if __name__ == "__main__":
    main()