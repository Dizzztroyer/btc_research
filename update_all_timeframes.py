#!/usr/bin/env python3
"""
update_all_timeframes.py
────────────────────────
Incrementally updates all locally stored timeframes with new candles only.
Reads existing local parquet files to determine the last stored timestamp,
then fetches only missing candles.

Usage
─────
    python update_all_timeframes.py
    python update_all_timeframes.py --tf 1h 4h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.data.downloader import OHLCVDownloader

logger = get_logger(__name__, log_file=Path("outputs/logs/update.log"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incrementally update BTC/USDT OHLCV data (only fetches new candles)."
    )
    p.add_argument(
        "--tf",
        nargs="+",
        metavar="TIMEFRAME",
        default=None,
        help="Restrict update to specific timeframes, e.g. --tf 1h 4h 1d",
    )
    p.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(Path(args.config))
    logger.info("=== BTC Research — Incremental Update ===")
    logger.info(f"Exchange : {cfg.exchange}")
    logger.info(f"Symbol   : {cfg.symbol}")

    downloader = OHLCVDownloader(cfg)

    # force_full=False → incremental update only
    results = downloader.download_all(
        timeframes=args.tf,
        force_full=False,
    )

    passed = sum(1 for v in results.values() if v.ok)
    failed = len(results) - passed

    logger.info("")
    logger.info("=== Update Summary ===")
    logger.info(f"Total timeframes : {len(results)}")
    logger.info(f"Passed           : {passed}")
    logger.info(f"Failed           : {failed}")

    for tf, vr in results.items():
        status = "OK" if vr.ok else "FAIL"
        gaps   = f"gaps={len(vr.gaps)}" if vr.gaps else ""
        logger.info(f"  [{status}] {tf:>4s}  {gaps}")

    if failed:
        logger.warning("Some updates had errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
