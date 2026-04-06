#!/usr/bin/env python3
"""
build_features.py
─────────────────
Reads raw parquet files and builds feature-enriched parquet files for
every locally stored timeframe.

Usage
─────
    python build_features.py                    # build all
    python build_features.py --tf 1h 4h 1d     # build specific timeframes
    python build_features.py --force            # rebuild even if feature file exists
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.features.feature_engine import FeatureEngine

logger = get_logger(__name__, log_file=Path("outputs/logs/features.log"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build local feature files from raw OHLCV data.")
    p.add_argument("--tf", nargs="+", metavar="TIMEFRAME", default=None)
    p.add_argument("--force", action="store_true", help="Rebuild even if feature file exists")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = load_config(Path(args.config))
    engine = FeatureEngine(cfg)

    if args.tf:
        timeframes = args.tf
    else:
        # Discover all raw timeframes
        timeframes = [p.stem for p in sorted(cfg.raw_dir.glob("*.parquet"))]

    if not timeframes:
        logger.error("No raw timeframe files found. Run download_all_timeframes.py first.")
        sys.exit(1)

    logger.info(f"Building features for: {timeframes}")

    success, fail = 0, 0
    for tf in timeframes:
        feat_path = cfg.features_dir / f"{tf}_features.parquet"
        if feat_path.exists() and not args.force:
            logger.info(f"[{tf}] Feature file already exists — skipping (use --force to rebuild)")
            success += 1
            continue
        try:
            df = engine.build(tf)
            logger.info(f"[{tf}] Built {df.shape[1]} features over {len(df):,} rows")
            success += 1
        except Exception as exc:
            logger.error(f"[{tf}] Failed: {exc}")
            fail += 1

    logger.info(f"\n=== Feature Build Summary ===")
    logger.info(f"Success : {success}")
    logger.info(f"Failed  : {fail}")

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
