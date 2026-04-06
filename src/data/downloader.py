"""
downloader.py
─────────────
Downloads and incrementally updates BTC/USDT OHLCV candles via ccxt.

Features
────────
- Auto-detects all timeframes supported by the exchange
- Saves each timeframe to a separate Parquet file
- Deduplicates and sorts candles
- Validates each download
- Saves/updates metadata.json
- Handles API rate-limits and pagination safely
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd

from src.data.validator import validate_ohlcv, ValidationResult
from src.utils.config_loader import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_utc_ms(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' string to UTC millisecond timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_to_dt(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def _raw_to_df(raw: list) -> pd.DataFrame:
    """Convert ccxt raw OHLCV list to a DataFrame."""
    df = pd.DataFrame(raw, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _parquet_path(cfg: Config, timeframe: str) -> Path:
    return cfg.raw_dir / f"{timeframe}.parquet"


def _load_existing(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        df = pd.read_parquet(path)
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        elif "timestamp" in df.columns:
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        return df
    return None


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df_save = df.copy()
    # Store timestamp as UTC-aware; pyarrow handles tz-aware correctly
    df_save.to_parquet(path, index=False, engine="pyarrow")


def _get_exchange_timeframes(cfg: Config) -> List[str]:
    """
    Return the list of supported timeframes for the configured exchange.
    If cfg.enabled_timeframes is set, intersect with supported ones.
    """
    exchange_cls = getattr(ccxt, cfg.exchange)
    exchange: ccxt.Exchange = exchange_cls({"enableRateLimit": True})

    # Fetch market meta to populate timeframes
    try:
        exchange.load_markets()
    except Exception as exc:
        logger.warning(f"Could not load markets ({exc}); using hardcoded fallback timeframes")
        # Fallback list — broad practical set
        all_tf = [
            "1m","3m","5m","15m","30m",
            "1h","2h","4h","6h","8h","12h",
            "1d","3d","1w","1M",
        ]
        if cfg.enabled_timeframes:
            return [t for t in cfg.enabled_timeframes if t in all_tf]
        return all_tf

    supported = list(exchange.timeframes.keys()) if exchange.timeframes else []

    if not supported:
        logger.warning("Exchange returned no timeframes; using fallback list")
        supported = [
            "1m","3m","5m","15m","30m",
            "1h","2h","4h","6h","8h","12h",
            "1d","3d","1w","1M",
        ]

    if cfg.enabled_timeframes:
        # Keep only user-requested TFs that the exchange actually supports
        result = [t for t in cfg.enabled_timeframes if t in supported]
        unsupported = [t for t in cfg.enabled_timeframes if t not in supported]
        if unsupported:
            logger.warning(f"Requested timeframes not supported by exchange: {unsupported}")
        return result

    logger.info(f"Exchange supports {len(supported)} timeframes: {supported}")
    return supported


# ── Core download logic ───────────────────────────────────────────────────────

class OHLCVDownloader:
    """
    Downloads and maintains BTC/USDT OHLCV data for all supported timeframes.
    """

    # Batch size for API requests (candles per request)
    BATCH_SIZE = 1000

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.exchange: ccxt.Exchange = self._init_exchange()

    def _init_exchange(self) -> ccxt.Exchange:
        exchange_cls = getattr(ccxt, self.cfg.exchange)
        ex = exchange_cls({"enableRateLimit": True})
        ex.load_markets()
        logger.info(
            f"Connected to {self.cfg.exchange} | "
            f"symbol: {self.cfg.symbol} | "
            f"rateLimit: {ex.rateLimit} ms"
        )
        return ex

    def _fetch_batch(
        self,
        timeframe: str,
        since_ms: int,
        limit: int = BATCH_SIZE,
    ) -> list:
        """Fetch one batch of candles, respecting rate limits."""
        retries = 5
        for attempt in range(retries):
            try:
                raw = self.exchange.fetch_ohlcv(
                    self.cfg.symbol,
                    timeframe=timeframe,
                    since=since_ms,
                    limit=limit,
                )
                return raw
            except ccxt.RateLimitExceeded:
                wait = 2 ** attempt
                logger.warning(f"Rate limit exceeded; sleeping {wait}s")
                time.sleep(wait)
            except ccxt.NetworkError as exc:
                wait = 2 ** attempt
                logger.warning(f"Network error ({exc}); retrying in {wait}s")
                time.sleep(wait)
            except Exception as exc:
                logger.error(f"Unrecoverable error fetching {timeframe}: {exc}")
                raise
        raise RuntimeError(f"Failed to fetch {timeframe} after {retries} retries")

    def _download_full(self, timeframe: str) -> pd.DataFrame:
        """Download full history for a given timeframe."""
        since_ms = _to_utc_ms(self.cfg.start_date)
        end_ms   = _to_utc_ms(self.cfg.end_date) if self.cfg.end_date else _now_utc_ms()

        all_rows: list = []
        logger.info(f"[{timeframe}] Full download from {self.cfg.start_date}")

        while since_ms < end_ms:
            batch = self._fetch_batch(timeframe, since_ms=since_ms)
            if not batch:
                break

            # Filter to requested date range
            batch = [r for r in batch if r[0] <= end_ms]
            all_rows.extend(batch)

            last_ts = batch[-1][0]
            logger.debug(f"[{timeframe}] fetched up to {_ms_to_dt(last_ts)}")

            if len(batch) < self.BATCH_SIZE:
                break  # reached end

            since_ms = last_ts + 1  # next candle after the last

            # Respect rate limit
            time.sleep(self.exchange.rateLimit / 1000.0)

        if not all_rows:
            logger.warning(f"[{timeframe}] No data returned")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = _raw_to_df(all_rows)
        return self._clean(df)

    def _update_incremental(self, timeframe: str, existing: pd.DataFrame) -> pd.DataFrame:
        """Fetch only candles newer than the latest stored timestamp."""
        last_ts = existing["timestamp"].max()
        since_ms = int(last_ts.timestamp() * 1000) + 1
        end_ms   = _to_utc_ms(self.cfg.end_date) if self.cfg.end_date else _now_utc_ms()

        if since_ms >= end_ms:
            logger.info(f"[{timeframe}] Already up to date ({last_ts})")
            return existing

        logger.info(f"[{timeframe}] Incremental update from {last_ts}")
        new_rows: list = []

        while since_ms < end_ms:
            batch = self._fetch_batch(timeframe, since_ms=since_ms)
            if not batch:
                break

            batch = [r for r in batch if r[0] <= end_ms]
            new_rows.extend(batch)

            last_ts_batch = batch[-1][0]
            logger.debug(f"[{timeframe}] fetched up to {_ms_to_dt(last_ts_batch)}")

            if len(batch) < self.BATCH_SIZE:
                break

            since_ms = last_ts_batch + 1
            time.sleep(self.exchange.rateLimit / 1000.0)

        if not new_rows:
            logger.info(f"[{timeframe}] No new candles")
            return existing

        new_df  = _raw_to_df(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        return self._clean(combined)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate, sort, and reset index."""
        df = df.drop_duplicates(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def download_timeframe(self, timeframe: str, force_full: bool = False) -> ValidationResult:
        """
        Download or update one timeframe.

        Parameters
        ----------
        timeframe  : ccxt timeframe string
        force_full : if True, re-download everything even if local file exists

        Returns
        -------
        ValidationResult from post-download validation
        """
        path     = _parquet_path(self.cfg, timeframe)
        existing = _load_existing(path)

        if existing is None or force_full:
            df = self._download_full(timeframe)
        else:
            df = self._update_incremental(timeframe, existing)

        if df.empty:
            logger.warning(f"[{timeframe}] Empty dataset — skipping save")
            return ValidationResult(ok=False, errors=["Empty dataset"])

        # ── Validate ───────────────────────────────────────────────────────────
        vr = validate_ohlcv(df, timeframe)
        if not vr.ok:
            logger.error(f"[{timeframe}] Validation errors:\n{vr}")
        else:
            if vr.warnings:
                for w in vr.warnings:
                    logger.warning(f"[{timeframe}] {w}")
            logger.info(
                f"[{timeframe}] OK | rows={len(df):,} | "
                f"{df['timestamp'].min()} → {df['timestamp'].max()} | "
                f"gaps={len(vr.gaps)}"
            )

        _save_parquet(df, path)
        return vr

    def download_all(
        self,
        timeframes: Optional[List[str]] = None,
        force_full: bool = False,
    ) -> Dict[str, ValidationResult]:
        """
        Download/update all timeframes.

        Parameters
        ----------
        timeframes : explicit list; if None, use exchange-detected list
        force_full : re-download everything

        Returns
        -------
        dict mapping timeframe → ValidationResult
        """
        if timeframes is None:
            timeframes = _get_exchange_timeframes(self.cfg)

        results: Dict[str, ValidationResult] = {}
        for tf in timeframes:
            logger.info(f"──── Timeframe: {tf} ────")
            try:
                vr = self.download_timeframe(tf, force_full=force_full)
                results[tf] = vr
            except Exception as exc:
                logger.error(f"[{tf}] Fatal error: {exc}")
                results[tf] = ValidationResult(ok=False, errors=[str(exc)])

        self._save_metadata(results)
        return results

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _save_metadata(self, results: Dict[str, ValidationResult]) -> None:
        meta: dict = {}
        for tf, vr in results.items():
            path = _parquet_path(self.cfg, tf)
            row: dict = {
                "symbol":      self.cfg.symbol,
                "timeframe":   tf,
                "ok":          vr.ok,
                "errors":      vr.errors,
                "warnings":    vr.warnings,
                "gap_count":   len(vr.gaps),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
            if path.exists():
                df = pd.read_parquet(path)
                if not df.empty and "timestamp" in df.columns:
                    ts = pd.to_datetime(df["timestamp"])
                    row["first_timestamp"] = str(ts.min())
                    row["last_timestamp"]  = str(ts.max())
                    row["row_count"]       = len(df)
                else:
                    row["row_count"] = 0
            meta[tf] = row

        meta_path = self.cfg.metadata_dir / f"{self.cfg.symbol_file}_metadata.json"
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        logger.info(f"Metadata saved → {meta_path}")
