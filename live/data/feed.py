"""
live/data/feed.py
──────────────────
Live OHLCV data feed using ccxt (Binance).

Fetches the last N candles for each timeframe,
builds features, and returns the latest completed bar.

Usage:
    feed = LiveDataFeed()
    df   = feed.get_latest("8h", n_candles=500)
    # df is a feature-rich DataFrame, last row = most recently CLOSED bar
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

try:
    import ccxt
    CCXT_OK = True
except ImportError:
    CCXT_OK = False


class LiveDataFeed:
    """Fetches live OHLCV from Binance and builds features."""

    SYMBOL   = "BTC/USDT:USDT"  # USDT-M perpetual futures
    EXCHANGE = "binanceusdm"

    # Map our TF labels to ccxt timeframes
    TF_MAP = {
        "6h":  "6h",
        "8h":  "8h",
        "12h": "12h",
        "1d":  "1d",
    }

    def __init__(self, logger=None):
        if not CCXT_OK:
            raise ImportError("pip install ccxt")
        self._exchange = getattr(ccxt, self.EXCHANGE)({"enableRateLimit": True})
        self._logger   = logger
        self._cache:  dict = {}
        self._cache_ts: dict = {}

    def _log(self, msg: str) -> None:
        if self._logger:
            self._logger.info(msg)
        else:
            print(f"[feed] {msg}")

    def get_latest(
        self,
        tf:        str,
        n_candles: int = 500,
        cache_sec: int = 60,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch last n_candles of OHLCV, build features, return DataFrame.
        The last row is the most recently CLOSED candle (not the current open one).
        Uses a simple time-based cache to avoid hammering the API.
        """
        now = time.time()
        if tf in self._cache and (now - self._cache_ts.get(tf, 0)) < cache_sec:
            return self._cache[tf]

        ccxt_tf = self.TF_MAP.get(tf)
        if ccxt_tf is None:
            raise ValueError(f"Unsupported TF: {tf}. Available: {list(self.TF_MAP)}")

        try:
            self._log(f"Fetching {n_candles} × {tf} candles for {self.SYMBOL}...")
            ohlcv = self._exchange.fetch_ohlcv(
                self.SYMBOL, ccxt_tf, limit=n_candles + 1
            )
        except Exception as exc:
            self._log(f"Fetch failed: {exc}")
            return None

        if not ohlcv or len(ohlcv) < 10:
            self._log("Not enough data returned")
            return None

        df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns=["ts"]).sort_values("timestamp").reset_index(drop=True)

        # Drop the last (currently forming) candle
        # The last complete candle is df.iloc[-2] when exchange includes current
        # Detect: if last bar timestamp + tf_ms > now → it's the current open bar
        tf_ms = self._tf_to_ms(tf)
        now_ms = int(time.time() * 1000)
        last_ts_ms = int(df["timestamp"].iloc[-1].timestamp() * 1000)
        if last_ts_ms + tf_ms > now_ms:
            df = df.iloc[:-1]  # drop current open bar

        self._log(f"  Got {len(df)} closed bars | last: {df['timestamp'].iloc[-1]}")

        # Build features
        df = self._build_features(df)

        self._cache[tf]    = df
        self._cache_ts[tf] = now
        return df

    @staticmethod
    def _tf_to_ms(tf: str) -> int:
        """Convert timeframe string to milliseconds."""
        mapping = {"1m":60000,"3m":180000,"5m":300000,"15m":900000,"30m":1800000,
                   "1h":3600000,"2h":7200000,"3h":10800000,"4h":14400000,
                   "6h":21600000,"8h":28800000,"12h":43200000,
                   "1d":86400000,"3d":259200000,"1w":604800000}
        return mapping.get(tf, 86400000)

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Build minimal feature set needed for strategy signal generation and ML sizing.
        Mirrors src/features/feature_engine.py but runs on live data without file I/O.
        """
        import numpy as np

        df = df.copy()
        c = df["close"]; h = df["high"]; l = df["low"]; o = df["open"]

        # ATR
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"]     = tr.ewm(span=14, adjust=False).mean()
        df["atr_7"]      = tr.ewm(span=7,  adjust=False).mean()
        df["atr_14_pct"] = df["atr_14"] / c

        # Returns
        for n in [1, 2, 5, 10]:
            df[f"ret_{n}"]     = c.pct_change(n)
            df[f"log_ret_{n}"] = np.log(c / c.shift(n))

        # EMA
        for n in [5, 8, 13, 21, 34, 55, 89, 144, 200]:
            df[f"ema_{n}"] = c.ewm(span=n, adjust=False).mean()

        df["dist_ema_21"]  = (c - df["ema_21"])  / c
        df["dist_ema_55"]  = (c - df["ema_55"])  / c
        df["dist_ema_200"] = (c - df["ema_200"]) / c
        df["ema_21_55_above"]  = (df["ema_21"] > df["ema_55"]).astype(int)
        df["ema_55_200_above"] = (df["ema_55"] > df["ema_200"]).astype(int)

        # RSI
        for n in [7, 14, 21]:
            delta = c.diff()
            gain  = delta.clip(lower=0).ewm(span=n, adjust=False).mean()
            loss  = (-delta.clip(upper=0)).ewm(span=n, adjust=False).mean()
            df[f"rsi_{n}"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

        # ADX
        for n in [10, 14]:
            plus_dm  = (h.diff()).clip(lower=0)
            minus_dm = (-l.diff()).clip(lower=0)
            mask     = plus_dm < minus_dm
            plus_dm[mask] = 0
            minus_dm[~mask] = 0
            tr_s     = tr.ewm(span=n, adjust=False).mean()
            plus_di  = 100 * plus_dm.ewm(span=n, adjust=False).mean() / tr_s.replace(0, 1e-9)
            minus_di = 100 * minus_dm.ewm(span=n, adjust=False).mean() / tr_s.replace(0, 1e-9)
            dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
            df[f"adx_{n}"] = dx.ewm(span=n, adjust=False).mean()

        # Donchian channels
        for n in [10, 20, 40, 55]:
            don_h = h.rolling(n).max()
            don_l = l.rolling(n).min()
            df[f"don_{n}_dist_high"] = (don_h - c) / c
            df[f"don_{n}_dist_low"]  = (c - don_l)  / c
            df[f"high_break_{n}"]    = (c > don_h.shift(1)).astype(int)
            df[f"low_break_{n}"]     = (c < don_l.shift(1)).astype(int)
            df[f"breakout_up_{n}"]   = (c > don_h.shift(1)).astype(int)
            df[f"breakout_down_{n}"] = (c < don_l.shift(1)).astype(int)

        # Volume z-score
        for n in [10, 20, 40]:
            vol_ma    = df["volume"].rolling(n).mean()
            vol_std   = df["volume"].rolling(n).std()
            df[f"vol_zscore_{n}"] = (df["volume"] - vol_ma) / vol_std.replace(0, 1e-9)

        # Volatility
        for n in [10, 20, 40]:
            df[f"vol_{n}"] = c.pct_change().rolling(n).std()

        # Bollinger bands (20, 2.0)
        bb_ma  = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        df["bb20s20_width"] = (bb_std * 4) / bb_ma.replace(0, 1e-9)
        df["bb20s20_pctb"]  = (c - (bb_ma - 2*bb_std)) / (bb_std * 4).replace(0, 1e-9)

        # Squeeze
        for n in [10, 20]:
            don_r   = h.rolling(n).max() - l.rolling(n).min()
            bb_r    = c.rolling(n).std() * 4
            df[f"squeeze_{n}"] = (bb_r < don_r).astype(int)

        # Range ratio
        rng = h - l
        for n in [10, 20]:
            df[f"range_ratio_{n}"] = rng / rng.rolling(n).mean().replace(0, 1e-9)

        # Candle shape
        df["body_pct"]    = (c - o).abs() / (h - l).replace(0, 1e-9)
        df["upper_wick"]  = (h - df[["open","close"]].max(axis=1)) / (h - l).replace(0, 1e-9)
        df["lower_wick"]  = (df[["open","close"]].min(axis=1) - l) / (h - l).replace(0, 1e-9)
        df["candle_range"]= (h - l) / c
        df["is_bullish"]  = (c > o).astype(int)

        # Session (UTC hour)
        hour = df["timestamp"].dt.hour
        df["session_asia"]    = ((hour >= 0)  & (hour < 8)).astype(int)
        df["session_europe"]  = ((hour >= 8)  & (hour < 16)).astype(int)
        df["session_us"]      = ((hour >= 13) & (hour < 21)).astype(int)
        df["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)

        # Calendar
        ts = df["timestamp"]
        df["hour"]       = ts.dt.hour
        df["day_of_week"]= ts.dt.dayofweek
        df["month"]      = ts.dt.month
        df["year"]       = ts.dt.year

        # ATR percentile rank (rolling 200)
        df["atr_pct_rank"] = (
            df["atr_14_pct"].rolling(200, min_periods=50).rank(pct=True)
        )

        # Regime
        trend_bull = (df["ema_21"] > df["ema_55"]) & (df["ema_55"] > df["ema_200"])
        trend_bear = (df["ema_21"] < df["ema_55"]) & (df["ema_55"] < df["ema_200"])
        df["regime_trend"] = np.where(trend_bull, 1, np.where(trend_bear, -1, 0))
        df["regime_range"] = (df["adx_14"] < 25).astype(int)

        hi_vol = df["atr_14_pct"] > df["atr_14_pct"].rolling(100, min_periods=20).quantile(0.75)
        lo_vol = df["atr_14_pct"] < df["atr_14_pct"].rolling(100, min_periods=20).quantile(0.25)
        df["regime_vol"]   = np.where(hi_vol, 1, np.where(lo_vol, -1, 0))

        return df

    def get_current_price(self) -> Optional[float]:
        """Fetch current mid-price (for signal enrichment)."""
        try:
            ticker = self._exchange.fetch_ticker("BTC/USDT:USDT")
            return float(ticker["last"])
        except Exception:
            return None