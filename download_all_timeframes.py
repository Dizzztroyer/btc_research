#!/usr/bin/env python3
"""
download_all_timeframes.py
──────────────────────────
Initial full download of BTC/USDT OHLCV data for ALL exchange-supported timeframes.

Usage
─────
    python download_all_timeframes.py
    python download_all_timeframes.py --force     # re-download even if local file exists
    python download_all_timeframes.py --tf 1h 4h  # restrict to specific timeframes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Ensure project root is on the path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.data.downloader import OHLCVDownloader

logger = get_logger(__name__, log_file=Path("outputs/logs/download.log"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download BTC/USDT OHLCV data for all supported timeframes."
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force full re-download even if local parquet files already exist.",
    )
    p.add_argument(
        "--tf",
        nargs="+",
        metavar="TIMEFRAME",
        default=None,
        help="Restrict download to specific timeframes, e.g. --tf 1h 4h 1d",
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
    logger.info("=== BTC Research — Full Download ===")
    logger.info(f"Exchange : {cfg.exchange}")
    logger.info(f"Symbol   : {cfg.symbol}")
    logger.info(f"From     : {cfg.start_date}")
    logger.info(f"To       : {cfg.end_date or 'now'}")
    logger.info(f"Force    : {args.force}")

    downloader = OHLCVDownloader(cfg)

    results = downloader.download_all(
        timeframes=args.tf,  # None → auto-detect from exchange
        force_full=args.force,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for v in results.values() if v.ok)
    failed = len(results) - passed

    logger.info("")
    logger.info("=== Download Summary ===")
    logger.info(f"Total timeframes : {len(results)}")
    logger.info(f"Passed           : {passed}")
    logger.info(f"Failed/warnings  : {failed}")

    for tf, vr in results.items():
        status = "OK" if vr.ok else "FAIL"
        gaps   = f"gaps={len(vr.gaps)}" if vr.gaps else ""
        logger.info(f"  [{status}] {tf:>4s}  {gaps}")

    if failed:
        logger.warning("Some timeframes had errors — check the log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
