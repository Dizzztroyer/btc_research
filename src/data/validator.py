"""
validator.py
────────────
Validates downloaded OHLCV DataFrames:
- Schema check
- Timestamp uniqueness and monotonicity
- Gap detection
- OHLCV sanity (non-negative prices, volume >= 0, high >= low, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd


# ── Expected schema ────────────────────────────────────────────────────────────
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# ccxt timeframe string → approximate milliseconds per candle
TF_MS: dict[str, int] = {
    "1m":  60_000,
    "3m":  3 * 60_000,
    "5m":  5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h":  3_600_000,
    "2h":  2 * 3_600_000,
    "4h":  4 * 3_600_000,
    "6h":  6 * 3_600_000,
    "8h":  8 * 3_600_000,
    "12h": 12 * 3_600_000,
    "1d":  86_400_000,
    "3d":  3 * 86_400_000,
    "1w":  7 * 86_400_000,
    "1M":  30 * 86_400_000,  # approximate
}


@dataclass
class ValidationResult:
    ok: bool = True
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    gaps:     List[Tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        lines = []
        lines.append(f"Validation {'PASSED' if self.ok else 'FAILED'}")
        for e in self.errors:
            lines.append(f"  ERROR   : {e}")
        for w in self.warnings:
            lines.append(f"  WARNING : {w}")
        if self.gaps:
            lines.append(f"  GAPS    : {len(self.gaps)}")
            for g in self.gaps[:5]:
                lines.append(f"    {g[0]} → {g[1]}")
            if len(self.gaps) > 5:
                lines.append(f"    ... and {len(self.gaps)-5} more")
        return "\n".join(lines)


def validate_ohlcv(df: pd.DataFrame, timeframe: str) -> ValidationResult:
    """
    Validate an OHLCV DataFrame.

    Parameters
    ----------
    df        : DataFrame with columns timestamp, open, high, low, close, volume
    timeframe : ccxt timeframe string, e.g. '1h'

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    # ── Schema ─────────────────────────────────────────────────────────────────
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        result.add_error(f"Missing columns: {missing}")
        return result  # cannot proceed

    if df.empty:
        result.add_error("DataFrame is empty")
        return result

    # ── Timestamp type ─────────────────────────────────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        result.add_error("'timestamp' column is not datetime64")
        return result

    # ── Duplicates ─────────────────────────────────────────────────────────────
    dup_count = df["timestamp"].duplicated().sum()
    if dup_count > 0:
        result.add_error(f"{dup_count} duplicate timestamps found")

    # ── Monotonicity ───────────────────────────────────────────────────────────
    if not df["timestamp"].is_monotonic_increasing:
        result.add_error("Timestamps are not sorted in ascending order")

    # ── OHLCV sanity ───────────────────────────────────────────────────────────
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            result.add_error(f"Non-positive values in column '{col}'")

    if (df["volume"] < 0).any():
        result.add_error("Negative volume values found")

    if (df["high"] < df["low"]).any():
        result.add_error("high < low in some rows")

    if (df["close"] > df["high"]).any():
        result.add_warning("Some close prices exceed high")

    if (df["close"] < df["low"]).any():
        result.add_warning("Some close prices are below low")

    # ── NaN check ─────────────────────────────────────────────────────────────
    nan_counts = df[REQUIRED_COLUMNS].isna().sum()
    for col, cnt in nan_counts.items():
        if cnt > 0:
            result.add_warning(f"{cnt} NaN values in column '{col}'")

    # ── Gap detection ─────────────────────────────────────────────────────────
    expected_ms = TF_MS.get(timeframe)
    if expected_ms is not None and len(df) > 1:
        ts_ms = df["timestamp"].astype(np.int64) // 1_000_000
        diffs = ts_ms.diff().iloc[1:]

        # Allow up to 2× the expected interval before flagging as a gap
        gap_threshold_ms = expected_ms * 2

        gap_mask = diffs > gap_threshold_ms
        if gap_mask.any():
            gap_indices = diffs[gap_mask].index
            for idx in gap_indices:
                gap_start = df.loc[idx - 1, "timestamp"]
                gap_end   = df.loc[idx,     "timestamp"]
                result.gaps.append((gap_start, gap_end))
            result.add_warning(
                f"{len(result.gaps)} gap(s) detected in timeframe {timeframe}"
            )
    elif expected_ms is None:
        result.add_warning(
            f"Unknown timeframe '{timeframe}' — skipping gap detection"
        )

    return result
