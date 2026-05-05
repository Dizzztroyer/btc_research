#!/usr/bin/env python3
"""
aggregate_3h.py
───────────────
Builds 3h OHLCV candles by resampling existing 1h data.

Binance supports 3h as a custom interval (confirmed on UI),
but ccxt does not expose it. This script creates the equivalent
data by grouping 1h candles into 3h bars.

Aggregation rules:
    open   = first bar's open
    high   = max of all highs
    low    = min of all lows
    close  = last bar's close
    volume = sum of all volumes

Output: data/raw/BTCUSDT/3h.parquet (same format as other TF files)

Usage
─────
    python aggregate_3h.py
    python aggregate_3h.py --source 1h  # default
    python aggregate_3h.py --verify     # compare against known 1h data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate 1h candles into 3h")
    p.add_argument("--config",  default="config/config.yaml")
    p.add_argument("--source",  default="1h", help="Source timeframe (default: 1h)")
    p.add_argument("--verify",  action="store_true", help="Print sample for manual check")
    return p.parse_args()


def aggregate_to_3h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1h OHLCV → 3h OHLCV.

    Groups every 3 consecutive 1h bars starting from UTC midnight.
    Incomplete groups (< 3 bars) at the end are dropped.
    """
    df = df_1h.copy()

    # Ensure UTC datetime index
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Set timestamp as index for resample
    df_idx = df.set_index("timestamp")

    # Resample to 3h — offset='0h' aligns to midnight UTC
    agg = df_idx.resample("3h", closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    })

    # Drop bars with no data (gaps)
    agg = agg.dropna(subset=["open", "close"])

    # Only keep complete 3h bars (where we had all 3 input bars)
    # Count how many 1h bars fell into each 3h bucket
    counts = df_idx["close"].resample("3h", closed="left", label="left").count()
    complete = counts[counts == 3].index
    agg = agg.loc[agg.index.intersection(complete)]

    agg = agg.reset_index()
    agg.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    agg["timestamp"] = agg["timestamp"].dt.tz_localize(None)  # strip tz for parquet consistency

    return agg


def main() -> None:
    args = parse_args()
    cfg  = load_config(Path(args.config))

    # Paths
    raw_dir   = Path(cfg.raw_dir)
    src_path  = raw_dir / f"{args.source}.parquet"
    out_path  = raw_dir / "3h.parquet"

    if not src_path.exists():
        logger.error(f"Source file not found: {src_path}")
        sys.exit(1)

    logger.info(f"Loading {args.source} data from {src_path}...")
    df_src = pd.read_parquet(src_path)
    logger.info(f"  Loaded {len(df_src):,} rows | {df_src['timestamp'].min()} → {df_src['timestamp'].max()}")

    logger.info("Aggregating to 3h...")
    df_3h = aggregate_to_3h(df_src)
    logger.info(f"  Result: {len(df_3h):,} rows | {df_3h['timestamp'].min()} → {df_3h['timestamp'].max()}")

    # Sanity checks
    expected_ratio = len(df_src) / len(df_3h)
    logger.info(f"  Ratio 1h/3h = {expected_ratio:.2f} (expected ~3.0)")
    if not (2.8 <= expected_ratio <= 3.2):
        logger.warning(f"  Unexpected ratio {expected_ratio:.2f} — check for gaps in 1h data")

    # Check for duplicate timestamps
    dupes = df_3h["timestamp"].duplicated().sum()
    if dupes > 0:
        logger.warning(f"  {dupes} duplicate timestamps — dropping")
        df_3h = df_3h.drop_duplicates(subset=["timestamp"])

    # Check OHLC consistency
    bad_hl = (df_3h["high"] < df_3h["low"]).sum()
    bad_oh = (df_3h["open"] > df_3h["high"]).sum()
    bad_ol = (df_3h["open"] < df_3h["low"]).sum()
    if bad_hl > 0 or bad_oh > 0 or bad_ol > 0:
        logger.warning(f"  OHLC issues: H<L={bad_hl}, O>H={bad_oh}, O<L={bad_ol}")

    if args.verify:
        print("\n=== Sample 3h bars (first 10) ===")
        print(df_3h.head(10).to_string(index=False))
        print("\n=== Corresponding 1h bars (first 30) ===")
        df_check = df_src.copy()
        df_check["timestamp"] = pd.to_datetime(df_check["timestamp"])
        print(df_check.head(30)[["timestamp","open","high","low","close","volume"]].to_string(index=False))
        print("\n=== Last 5 3h bars ===")
        print(df_3h.tail(5).to_string(index=False))

    # Save
    df_3h.to_parquet(out_path, index=False)
    logger.info(f"Saved → {out_path}")

    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"File size: {size_mb:.1f} MB")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. python -m src.features.build_features --tf 3h")
    logger.info("  2. python run_research.py --tf 3h --strategy structure trend --random-n 0")


if __name__ == "__main__":
    main()