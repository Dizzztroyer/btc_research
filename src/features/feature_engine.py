"""
feature_engine.py
─────────────────
Reads raw local OHLCV parquet files and computes a comprehensive set of
technical, structural, and calendar features.  Results are cached to
data/features/<SYMBOL>/<timeframe>_features.parquet.

All computation is purely forward-looking-free (no lookahead bias).
Every column is named explicitly — no anonymous numeric suffixes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.config_loader import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level indicator helpers (no external TA library dependency required)
# ═══════════════════════════════════════════════════════════════════════════════

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=length - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=length - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    length: int,
) -> pd.Series:
    """Compute ADX (Average Directional Index)."""
    tr  = _atr(high, low, close, 1)   # raw TR, not smoothed
    # Use true range rolling sum for DM smoothing
    raw_tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    dm_plus  = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)

    # Zero out where the other DM is larger
    mask = dm_plus >= dm_minus
    dm_plus  = dm_plus.where(mask, 0)
    dm_minus = dm_minus.where(~mask, 0)

    atr_s    = raw_tr.ewm(span=length, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=length, adjust=False).mean()  / atr_s.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=length, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    return dx.ewm(span=length, adjust=False).mean()


def _bollinger(
    close: pd.Series,
    length: int,
    n_std: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, mid, lower) Bollinger Bands."""
    mid   = _sma(close, length)
    sigma = close.rolling(length).std(ddof=0)
    return mid + n_std * sigma, mid, mid - n_std * sigma


def _donchian(
    high: pd.Series,
    low:  pd.Series,
    length: int,
) -> tuple[pd.Series, pd.Series]:
    """Returns (donchian_high, donchian_low) — excludes current bar."""
    return high.shift(1).rolling(length).max(), low.shift(1).rolling(length).min()


def _volume_zscore(volume: pd.Series, length: int) -> pd.Series:
    mu    = volume.rolling(length).mean()
    sigma = volume.rolling(length).std(ddof=0)
    return (volume - mu) / sigma.replace(0, np.nan)


def _rolling_high_break(high: pd.Series, length: int) -> pd.Series:
    """1 if current high exceeds the rolling high of previous `length` bars."""
    prev_max = high.shift(1).rolling(length).max()
    return (high > prev_max).astype(int)


def _rolling_low_break(low: pd.Series, length: int) -> pd.Series:
    """1 if current low is below the rolling low of previous `length` bars."""
    prev_min = low.shift(1).rolling(length).min()
    return (low < prev_min).astype(int)


def _squeeze(
    close: pd.Series,
    high:  pd.Series,
    low:   pd.Series,
    length: int,
    bb_std: float = 2.0,
    kc_mult: float = 1.5,
) -> pd.Series:
    """
    Bollinger / Keltner squeeze indicator.
    Returns 1 where BB is inside KC (squeeze on), 0 otherwise.
    """
    bb_upper, bb_mid, bb_lower = _bollinger(close, length, bb_std)
    atr_val = _atr(high, low, close, length)
    kc_upper = bb_mid + kc_mult * atr_val
    kc_lower = bb_mid - kc_mult * atr_val
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    return squeeze.astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# Main feature computation
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngine:
    """
    Computes and caches all features for one or all timeframes.
    Reads from data/raw/<SYMBOL>/<tf>.parquet
    Writes to data/features/<SYMBOL>/<tf>_features.parquet
    """

    # EMA and SMA lengths to compute (creates named columns)
    EMA_LENGTHS = [5, 8, 13, 21, 34, 55, 89, 144, 200]
    SMA_LENGTHS = [10, 20, 50, 100, 200]
    RSI_LENGTHS = [7, 14, 21]
    ATR_LENGTHS = [7, 14, 21]
    ADX_LENGTHS = [10, 14]
    BB_PARAMS   = [(20, 2.0), (20, 1.5), (14, 2.0)]   # (length, std)
    DONCHIAN_LENGTHS = [10, 20, 40, 55]
    VOL_LOOKBACKS = [10, 20, 40]
    SQUEEZE_LENGTHS = [10, 20]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def _raw_path(self, timeframe: str) -> Path:
        return self.cfg.raw_dir / f"{timeframe}.parquet"

    def _feat_path(self, timeframe: str) -> Path:
        return self.cfg.features_dir / f"{timeframe}_features.parquet"

    def _available_raw_timeframes(self) -> List[str]:
        paths = sorted(self.cfg.raw_dir.glob("*.parquet"))
        return [p.stem for p in paths]

    def build(self, timeframe: str) -> pd.DataFrame:
        """
        Build features for one timeframe.

        Reads raw parquet, computes all features, saves feature parquet,
        and returns the feature DataFrame.
        """
        raw_path = self._raw_path(timeframe)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data not found for {timeframe}: {raw_path}\n"
                "Run download_all_timeframes.py first."
            )

        logger.info(f"[{timeframe}] Loading raw data …")
        df = pd.read_parquet(raw_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"[{timeframe}] Computing features ({len(df):,} rows) …")
        feat = self._compute_all(df)

        feat_path = self._feat_path(timeframe)
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(feat_path, index=False, engine="pyarrow")
        logger.info(
            f"[{timeframe}] Features saved → {feat_path} "
            f"({feat.shape[1]} columns)"
        )
        return feat

    def build_all(self, timeframes: Optional[List[str]] = None) -> dict[str, pd.DataFrame]:
        """Build features for all (or selected) timeframes."""
        if timeframes is None:
            timeframes = self._available_raw_timeframes()

        results = {}
        for tf in timeframes:
            try:
                results[tf] = self.build(tf)
            except FileNotFoundError as exc:
                logger.warning(str(exc))
            except Exception as exc:
                logger.error(f"[{tf}] Feature build failed: {exc}")
        return results

    def load(self, timeframe: str) -> pd.DataFrame:
        """Load pre-built feature parquet. Build if missing."""
        path = self._feat_path(timeframe)
        if not path.exists():
            logger.info(f"[{timeframe}] Feature file not found — building now …")
            return self.build(timeframe)
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    # ── Core computation ───────────────────────────────────────────────────────

    def _compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the full feature set and return a combined DataFrame."""
        out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        c = df["close"]
        h = df["high"]
        l = df["low"]
        o = df["open"]
        v = df["volume"]

        # ── Returns ────────────────────────────────────────────────────────────
        out["ret_1"]      = c.pct_change(1)
        out["ret_2"]      = c.pct_change(2)
        out["ret_5"]      = c.pct_change(5)
        out["ret_10"]     = c.pct_change(10)
        out["log_ret_1"]  = np.log(c / c.shift(1))
        out["log_ret_5"]  = np.log(c / c.shift(5))

        # ── Candle anatomy ─────────────────────────────────────────────────────
        body = (c - o).abs()
        rng  = h - l
        out["body_size"]    = body
        out["body_pct"]     = body / rng.replace(0, np.nan)
        out["upper_wick"]   = h - pd.concat([c, o], axis=1).max(axis=1)
        out["lower_wick"]   = pd.concat([c, o], axis=1).min(axis=1) - l
        out["candle_range"] = rng
        out["is_bullish"]   = (c >= o).astype(int)

        # ── ATR ────────────────────────────────────────────────────────────────
        for n in self.ATR_LENGTHS:
            out[f"atr_{n}"]    = _atr(h, l, c, n)
            out[f"atr_{n}_pct"] = out[f"atr_{n}"] / c  # normalised

        # ── Rolling volatility (std of log returns) ────────────────────────────
        for n in self.VOL_LOOKBACKS:
            out[f"vol_{n}"] = out["log_ret_1"].rolling(n).std()

        # ── EMA ────────────────────────────────────────────────────────────────
        for n in self.EMA_LENGTHS:
            out[f"ema_{n}"] = _ema(c, n)
            out[f"dist_ema_{n}"] = (c - out[f"ema_{n}"]) / out[f"ema_{n}"]

        # ── SMA ────────────────────────────────────────────────────────────────
        for n in self.SMA_LENGTHS:
            out[f"sma_{n}"] = _sma(c, n)
            out[f"dist_sma_{n}"] = (c - out[f"sma_{n}"]) / out[f"sma_{n}"]

        # ── RSI ────────────────────────────────────────────────────────────────
        for n in self.RSI_LENGTHS:
            out[f"rsi_{n}"] = _rsi(c, n)

        # ── ADX ────────────────────────────────────────────────────────────────
        for n in self.ADX_LENGTHS:
            out[f"adx_{n}"] = _adx(h, l, c, n)

        # ── Bollinger Bands ────────────────────────────────────────────────────
        for bb_len, bb_std in self.BB_PARAMS:
            tag  = f"bb{bb_len}s{str(bb_std).replace('.','')}"
            bbu, bbm, bbl = _bollinger(c, bb_len, bb_std)
            out[f"{tag}_upper"] = bbu
            out[f"{tag}_mid"]   = bbm
            out[f"{tag}_lower"] = bbl
            bw = (bbu - bbl) / bbm.replace(0, np.nan)
            out[f"{tag}_width"]  = bw
            out[f"{tag}_pctb"]   = (c - bbl) / (bbu - bbl).replace(0, np.nan)

        # ── Donchian channels ──────────────────────────────────────────────────
        for n in self.DONCHIAN_LENGTHS:
            dh, dl   = _donchian(h, l, n)
            dmid     = (dh + dl) / 2
            out[f"don_{n}_high"] = dh
            out[f"don_{n}_low"]  = dl
            out[f"don_{n}_mid"]  = dmid
            out[f"don_{n}_dist_high"] = (c - dh) / dh.replace(0, np.nan)
            out[f"don_{n}_dist_low"]  = (c - dl) / dl.replace(0, np.nan)

        # ── Breakout distance ──────────────────────────────────────────────────
        for n in self.DONCHIAN_LENGTHS:
            dh, dl = _donchian(h, l, n)
            out[f"breakout_up_{n}"]   = (c - dh) / dh.replace(0, np.nan)
            out[f"breakout_down_{n}"] = (dl - c) / dl.replace(0, np.nan)

        # ── Rolling high / low breaks ──────────────────────────────────────────
        for n in [10, 20, 40]:
            out[f"high_break_{n}"] = _rolling_high_break(h, n)
            out[f"low_break_{n}"]  = _rolling_low_break(l, n)

        # ── Volume z-score ─────────────────────────────────────────────────────
        for n in self.VOL_LOOKBACKS:
            out[f"vol_zscore_{n}"] = _volume_zscore(v, n)

        # ── Squeeze indicator ──────────────────────────────────────────────────
        for n in self.SQUEEZE_LENGTHS:
            out[f"squeeze_{n}"] = _squeeze(c, h, l, n)

        # ── Range compression / expansion ──────────────────────────────────────
        for n in [10, 20]:
            avg_rng = rng.rolling(n).mean()
            out[f"range_ratio_{n}"] = rng / avg_rng.replace(0, np.nan)

        # ── Calendar features ──────────────────────────────────────────────────
        ts                   = df["timestamp"].dt
        out["hour"]          = ts.hour
        out["day_of_week"]   = ts.dayofweek     # Monday=0, Sunday=6
        out["week_of_year"]  = ts.isocalendar().week.astype(int)
        out["month"]         = ts.month
        out["quarter"]       = ts.quarter
        out["year"]          = ts.year

        # Cyclical encoding of hour and day-of-week (helps ML models)
        out["hour_sin"]      = np.sin(2 * np.pi * out["hour"]        / 24)
        out["hour_cos"]      = np.cos(2 * np.pi * out["hour"]        / 24)
        out["dow_sin"]       = np.sin(2 * np.pi * out["day_of_week"] / 7)
        out["dow_cos"]       = np.cos(2 * np.pi * out["day_of_week"] / 7)
        out["month_sin"]     = np.sin(2 * np.pi * out["month"]       / 12)
        out["month_cos"]     = np.cos(2 * np.pi * out["month"]       / 12)

        # ── Trading session markers (UTC) ──────────────────────────────────────
        h_val = out["hour"]
        out["session_asia"]   = ((h_val >= 0)  & (h_val < 8)).astype(int)
        out["session_europe"] = ((h_val >= 7)  & (h_val < 16)).astype(int)
        out["session_us"]     = ((h_val >= 13) & (h_val < 22)).astype(int)
        out["session_overlap"]= ((h_val >= 13) & (h_val < 16)).astype(int)

        # ── EMA cross signals ──────────────────────────────────────────────────
        out["ema_8_21_cross_up"]  = (
            (_ema(c, 8) > _ema(c, 21)) &
            (_ema(c, 8).shift(1) <= _ema(c, 21).shift(1))
        ).astype(int)
        out["ema_8_21_cross_dn"]  = (
            (_ema(c, 8) < _ema(c, 21)) &
            (_ema(c, 8).shift(1) >= _ema(c, 21).shift(1))
        ).astype(int)
        out["ema_21_55_above"] = (_ema(c, 21) > _ema(c, 55)).astype(int)
        out["ema_55_200_above"] = (_ema(c, 55) > _ema(c, 200)).astype(int)

        # ── Market regime labels ───────────────────────────────────────────────
        out = self._add_regime_labels(out)

        return out

    def _add_regime_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add simple rule-based regime labels.

        regime_trend  : 1 = bullish trend, -1 = bearish trend, 0 = neutral
        regime_range  : 1 = low ADX ranging, 0 = not
        regime_vol    : 1 = high volatility, -1 = low volatility, 0 = normal
        regime_master : combined categorical label
        """
        c = df["close"]

        # ── Trend regime via EMA alignment ────────────────────────────────────
        ema21  = df.get("ema_21",  _ema(c, 21))
        ema55  = df.get("ema_55",  _ema(c, 55))
        ema200 = df.get("ema_200", _ema(c, 200))

        trend_bull = (c > ema21) & (ema21 > ema55) & (ema55 > ema200)
        trend_bear = (c < ema21) & (ema21 < ema55) & (ema55 < ema200)
        df["regime_trend"] = np.where(trend_bull, 1, np.where(trend_bear, -1, 0))

        # ── Range regime via ADX ───────────────────────────────────────────────
        adx_col = "adx_14" if "adx_14" in df.columns else None
        if adx_col:
            df["regime_range"] = (df[adx_col] < 25).astype(int)
        else:
            df["regime_range"] = 0

        # ── Volatility regime ─────────────────────────────────────────────────
        if "vol_20" in df.columns:
            vol_series  = df["vol_20"]
            vol_50_pct  = vol_series.rolling(200, min_periods=50).quantile(0.75)
            vol_25_pct  = vol_series.rolling(200, min_periods=50).quantile(0.25)
            high_vol    = vol_series > vol_50_pct
            low_vol     = vol_series < vol_25_pct
            df["regime_vol"] = np.where(high_vol, 1, np.where(low_vol, -1, 0))
        else:
            df["regime_vol"] = 0

        # ── Master regime label (string for reporting) ─────────────────────────
        conditions = [
            (df["regime_trend"] == 1)  & (df["regime_range"] == 0),
            (df["regime_trend"] == -1) & (df["regime_range"] == 0),
            (df["regime_range"] == 1),
            (df["regime_vol"]   == 1),
        ]
        choices = ["bull_trend", "bear_trend", "range", "high_vol"]
        df["regime_master"] = np.select(conditions, choices, default="neutral")

        return df
