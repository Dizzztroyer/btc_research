#!/usr/bin/env python3
"""
run_multiples_of_3.py
──────────────────────
Step 1: Check if Binance supports 3h and 3M timeframes.
Step 2: Download missing ones (3m already present — skip).
Step 3: Build features for newly downloaded TFs.
Step 4: Run research (swing_breakout + donchian_breakout) on:
        3h, 3d, 6h, 12h, and 3M if available.
Step 5: Compare results vs benchmark.

Usage
─────
    python run_multiples_of_3.py
    python run_multiples_of_3.py --skip-download   # if data already present
    python run_multiples_of_3.py --research-only   # skip check/download
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.data.downloader import OHLCVDownloader, _parquet_path, _load_existing

logger = get_logger(__name__, log_file=Path("outputs/logs/multiples_of_3.log"))

# ── Benchmark for comparison ───────────────────────────────────────────────────
BENCHMARK = {
    "swing_breakout/8h+ML":  {"pf": 1.849, "sharpe": 1.762, "mdd": -0.1321, "trades": 246},
    "swing_breakout/12h+ML": {"pf": 1.890, "sharpe": 1.736, "mdd": -0.0851, "trades": 209},
}

# Target timeframes (multiples of 3, excluding 3m which already exists)
TARGET_TFS = ["3h", "3d", "6h", "12h"]  # 3M added if supported


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiples-of-3 TF research")
    p.add_argument("--config",         default="config/config.yaml")
    p.add_argument("--skip-download",  action="store_true")
    p.add_argument("--research-only",  action="store_true")
    p.add_argument("--no-wf",          action="store_true")
    p.add_argument("--random-n",       type=int, default=0)
    return p.parse_args()


# ── Step 1: Check exchange support ────────────────────────────────────────────

def check_exchange_support(cfg) -> dict:
    """
    Query Binance for supported timeframes.
    Returns dict: {timeframe: supported (bool)}
    """
    import ccxt
    logger.info("Checking Binance supported timeframes...")
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        exchange.load_markets()
        supported = set(exchange.timeframes.keys()) if exchange.timeframes else set()
        logger.info(f"Binance supports {len(supported)} timeframes")
    except Exception as exc:
        logger.warning(f"Could not query exchange: {exc}")
        # Conservative fallback based on known Binance timeframes
        supported = {
            "1m","3m","5m","15m","30m",
            "1h","2h","4h","6h","8h","12h",
            "1d","3d","1w","1M",
        }
        logger.info("Using known Binance timeframe list as fallback")

    check_tfs = ["3h", "3M"]
    result = {}
    for tf in check_tfs:
        is_supported = tf in supported
        result[tf] = is_supported
        status = "✓ SUPPORTED" if is_supported else "✗ NOT SUPPORTED"
        logger.info(f"  {tf:>4s}: {status}")

    return result


# ── Step 2: Download missing TFs ──────────────────────────────────────────────

def download_missing(cfg, supported: dict) -> list:
    """Download 3h and 3M if supported and not already present."""
    to_download = []

    # 3h
    if supported.get("3h", False):
        path = _parquet_path(cfg, "3h")
        if path.exists():
            df = _load_existing(path)
            logger.info(f"  3h: already present ({len(df):,} rows) — skipping download")
        else:
            to_download.append("3h")
            logger.info("  3h: will download")
    else:
        logger.info("  3h: not supported by exchange — skipping")

    # 3M
    if supported.get("3M", False):
        path = _parquet_path(cfg, "3M")
        if path.exists():
            df = _load_existing(path)
            logger.info(f"  3M: already present ({len(df):,} rows) — skipping download")
        else:
            to_download.append("3M")
            logger.info("  3M: will download")
    else:
        logger.info("  3M: not supported by exchange — skipping")

    if to_download:
        logger.info(f"Downloading: {to_download}")
        downloader = OHLCVDownloader(cfg)
        results = downloader.download_all(timeframes=to_download, force_full=False)
        for tf, vr in results.items():
            if vr.ok:
                logger.info(f"  {tf}: downloaded successfully | gaps={len(vr.gaps)}")
            else:
                logger.error(f"  {tf}: download failed — {vr.errors}")

    return to_download


# ── Step 3: Build features ────────────────────────────────────────────────────

def build_features_for(cfg, timeframes: list) -> list:
    """Build feature files for given timeframes."""
    from src.features.feature_engine import FeatureEngine
    engine = FeatureEngine(cfg)
    built  = []
    for tf in timeframes:
        raw_path  = cfg.raw_dir / f"{tf}.parquet"
        feat_path = cfg.features_dir / f"{tf}_features.parquet"
        if not raw_path.exists():
            logger.warning(f"  {tf}: raw file not found — skipping features")
            continue
        if feat_path.exists():
            logger.info(f"  {tf}: feature file already exists — skipping")
            built.append(tf)
            continue
        try:
            df = engine.build(tf)
            logger.info(f"  {tf}: built {df.shape[1]} features over {len(df):,} rows")
            built.append(tf)
        except Exception as exc:
            logger.error(f"  {tf}: feature build failed: {exc}")
    return built


# ── Step 4: Run research ──────────────────────────────────────────────────────

def run_research_on_tfs(cfg, timeframes: list, no_wf: bool, random_n: int) -> None:
    """Run swing_breakout + donchian_breakout on given timeframes."""
    import subprocess

    available = [
        tf for tf in timeframes
        if (cfg.features_dir / f"{tf}_features.parquet").exists()
    ]
    if not available:
        logger.error("No feature files available for research")
        return

    logger.info(f"Running research on: {available}")

    cmd = [
        sys.executable, "run_research.py",
        "--tf", *available,
        "--strategy", "structure", "trend",
        "--random-n", str(random_n),
        "--reset-checkpoint",
    ]
    if no_wf:
        cmd.append("--no-wf")

    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        logger.error("Research run failed")
    else:
        logger.info("Research complete")


# ── Step 5: Compare vs benchmark ─────────────────────────────────────────────

def compare_vs_benchmark(cfg, target_tfs: list) -> None:
    """Load results and compare swing/donchian on target TFs vs benchmark."""
    results_path = cfg.output_dir / "rankings" / "all_results.csv"
    wf_path      = cfg.output_dir / "rankings" / "wf_summary.csv"

    if not results_path.exists():
        logger.warning("all_results.csv not found — skipping comparison")
        return

    df = pd.read_csv(results_path)
    wf = pd.read_csv(wf_path) if wf_path.exists() else pd.DataFrame()

    focus_strats = ["swing_breakout", "donchian_breakout"]
    mask = (
        df["strategy"].isin(focus_strats) &
        df["timeframe"].isin(target_tfs) &
        (df["robustness"] > 0) &
        (df["oos_pf"] >= 1.0) &
        (df["oos_trades"] >= 30)
    )
    accepted = df[mask].copy()

    if accepted.empty:
        logger.warning("No accepted results for multiples-of-3 TFs")
        return

    best = (
        accepted
        .groupby(["strategy","timeframe"])
        .agg(
            best_oos_pf     = ("oos_pf",     "max"),
            best_oos_sharpe = ("oos_sharpe",  "max"),
            best_oos_return = ("oos_return",  "max"),
            oos_mdd         = ("oos_drawdown","min"),
            n_combos        = ("robustness",  "count"),
        )
        .reset_index()
    )

    # Add WF
    if not wf.empty:
        best = best.merge(
            wf[wf["strategy"].isin(focus_strats)][
                ["strategy","timeframe","wf_sharpe","wf_pf","wf_return","wf_mdd"]
            ],
            on=["strategy","timeframe"], how="left"
        )

    # Save
    out_path = cfg.output_dir / "rankings" / "multiples_of_3_results.csv"
    best.to_csv(out_path, index=False)
    logger.info(f"Results → {out_path}")

    # Print
    print(f"\n{'='*75}")
    print("MULTIPLES-OF-3 RESULTS vs BENCHMARK")
    print(f"{'='*75}")
    print(f"\nBENCHMARK:")
    for name, b in BENCHMARK.items():
        print(f"  {name}: PF={b['pf']:.3f}  Sharpe={b['sharpe']:.3f}  MDD={b['mdd']:.2%}")

    print(f"\nNEW RESULTS:")
    cols_show = ["strategy","timeframe","best_oos_pf","best_oos_sharpe",
                 "best_oos_return","oos_mdd","n_combos"]
    if "wf_sharpe" in best.columns:
        cols_show += ["wf_sharpe","wf_pf"]
    available_cols = [c for c in cols_show if c in best.columns]
    print(best[available_cols].sort_values("best_oos_sharpe", ascending=False).to_string(index=False))

    # Highlight any that beat the benchmark
    print(f"\nBEATS BENCHMARK (Sharpe > 1.736)?")
    if "wf_sharpe" in best.columns:
        winners = best[best["wf_sharpe"] > 1.736]
    else:
        winners = best[best["best_oos_sharpe"] > 1.736]
    if winners.empty:
        print("  None — benchmark holds")
    else:
        for _, r in winners.iterrows():
            print(f"  ✓ {r['strategy']}/{r['timeframe']}: OOS Sharpe={r['best_oos_sharpe']:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(Path(args.config))

    logger.info("=== Multiples-of-3 TF Research ===")
    logger.info(f"Target TFs: {TARGET_TFS} + 3M if supported")

    supported = {"3h": False, "3M": False}

    if not args.research_only:
        # Step 1: Check support
        logger.info("\n--- Step 1: Checking exchange support ---")
        supported = check_exchange_support(cfg)

        if not args.skip_download:
            # Step 2: Download
            logger.info("\n--- Step 2: Downloading missing TFs ---")
            download_missing(cfg, supported)

        # Step 3: Features
        logger.info("\n--- Step 3: Building features ---")
        tfs_to_build = list(TARGET_TFS)
        if supported.get("3M"):
            tfs_to_build.append("3M")
        build_features_for(cfg, tfs_to_build)

    # Determine which TFs are available for research
    research_tfs = [
        tf for tf in TARGET_TFS
        if (cfg.features_dir / f"{tf}_features.parquet").exists()
    ]
    if supported.get("3M") and (cfg.features_dir / "3M_features.parquet").exists():
        research_tfs.append("3M")

    if not research_tfs:
        logger.error("No feature files found for target TFs")
        sys.exit(1)

    logger.info(f"\nTFs available for research: {research_tfs}")

    # Step 4: Research
    logger.info("\n--- Step 4: Running research ---")
    run_research_on_tfs(cfg, research_tfs, args.no_wf, args.random_n)

    # Step 5: Compare
    logger.info("\n--- Step 5: Comparing vs benchmark ---")
    compare_vs_benchmark(cfg, research_tfs)

    logger.info("\nDone. Check outputs/rankings/multiples_of_3_results.csv")


if __name__ == "__main__":
    main()