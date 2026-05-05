"""
regime_filter.py
────────────────
Market regime classifier with two backends:

1. Rule-based (default, fast, no ML)
   Uses ADX + volatility + EMA alignment to classify:
       TREND    — high ADX, clear EMA direction
       RANGE    — low ADX, compressed volatility
       VOLATILE — high vol, no clear direction

2. ML-based (optional, XGBoost)
   Trains on labelled regime windows derived from future returns.
   More adaptive but requires more data.

Regime filtering:
   - TREND    → trade normally
   - RANGE    → reduce position size (configurable, default 0.0 = skip)
   - VOLATILE → trade normally OR reduce (configurable)

No lookahead: regime is computed from bar N features for bar N signal.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Regime labels ─────────────────────────────────────────────────────────────

class Regime(IntEnum):
    TREND    = 1
    RANGE    = 0
    VOLATILE = 2


REGIME_NAMES = {
    Regime.TREND:    "trend",
    Regime.RANGE:    "range",
    Regime.VOLATILE: "volatile",
}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RegimeConfig:
    # Rule-based thresholds
    adx_trend_thresh:   float = 22.0   # ADX above → trending
    adx_range_thresh:   float = 18.0   # ADX below → ranging
    adx_length:         int   = 14

    vol_high_pct:       float = 0.75   # vol above 75th pct of history → volatile
    vol_lookback:       int   = 200    # rolling window for vol percentile

    ema_align_fast:     int   = 21
    ema_align_slow:     int   = 55
    ema_align_required: bool  = True   # require EMA alignment for TREND

    # How to handle each regime
    # 1.0 = trade normally, 0.0 = skip, 0.5 = half size
    trend_size_mult:    float = 1.0
    range_size_mult:    float = 0.0    # default: skip in ranging markets
    volatile_size_mult: float = 0.7    # reduce in volatile markets

    # Smoothing: require N consecutive bars in regime before switching
    min_regime_bars:    int   = 3

    # ML-based regime (optional)
    use_ml_regime:      bool  = False
    ml_regime_lookforward: int = 20    # bars ahead to compute return label
    ml_regime_threshold:   float = 0.003  # annualised daily return threshold


# ── Rule-based classifier ─────────────────────────────────────────────────────

def classify_regime_rules(
    df:  pd.DataFrame,
    cfg: RegimeConfig,
) -> pd.Series:
    """
    Classify each bar into Regime.TREND / RANGE / VOLATILE.

    Uses only lagged indicators (shift(1)) — no lookahead.
    Returns a Series of Regime values, index aligned with df.
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # ── ADX ───────────────────────────────────────────────────────────────────
    adx_col = f"adx_{cfg.adx_length}"
    if adx_col in df.columns:
        adx = df[adx_col].shift(1)
    else:
        # Compute inline
        n      = cfg.adx_length
        raw_tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        dm_p   = (h - h.shift(1)).clip(lower=0)
        dm_m   = (l.shift(1) - l).clip(lower=0)
        mask   = dm_p >= dm_m
        dm_p   = dm_p.where(mask, 0)
        dm_m   = dm_m.where(~mask, 0)
        atr_s  = raw_tr.ewm(span=n, adjust=False).mean()
        dip    = 100 * dm_p.ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
        dim    = 100 * dm_m.ewm(span=n, adjust=False).mean() / atr_s.replace(0, np.nan)
        dx     = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
        adx    = dx.ewm(span=n, adjust=False).mean().shift(1)

    # ── Volatility percentile ─────────────────────────────────────────────────
    vol_col = "vol_20"
    if vol_col in df.columns:
        vol = df[vol_col].shift(1)
    else:
        log_ret = np.log(c / c.shift(1))
        vol     = log_ret.rolling(20).std().shift(1)

    vol_pct = vol.rolling(cfg.vol_lookback, min_periods=50).quantile(cfg.vol_high_pct)
    high_vol = vol > vol_pct

    # ── EMA alignment ─────────────────────────────────────────────────────────
    if cfg.ema_align_required:
        ema_fast_col = f"ema_{cfg.ema_align_fast}"
        ema_slow_col = f"ema_{cfg.ema_align_slow}"
        ema_fast = df[ema_fast_col].shift(1) if ema_fast_col in df.columns else \
                   c.ewm(span=cfg.ema_align_fast, adjust=False).mean().shift(1)
        ema_slow = df[ema_slow_col].shift(1) if ema_slow_col in df.columns else \
                   c.ewm(span=cfg.ema_align_slow, adjust=False).mean().shift(1)
        ema_aligned = (
            ((c.shift(1) > ema_fast) & (ema_fast > ema_slow)) |   # bullish
            ((c.shift(1) < ema_fast) & (ema_fast < ema_slow))     # bearish
        )
    else:
        ema_aligned = pd.Series(True, index=df.index)

    # ── Classify ───────────────────────────────────────────────────────────────
    raw_regime = pd.Series(Regime.RANGE, index=df.index, dtype=int)

    is_trend = (adx >= cfg.adx_trend_thresh) & ema_aligned & (~high_vol)
    is_volatile = high_vol & (adx < cfg.adx_trend_thresh)

    raw_regime = raw_regime.where(~is_trend,    Regime.TREND)
    raw_regime = raw_regime.where(~is_volatile,  Regime.VOLATILE)

    # ── Smoothing: require min_regime_bars consecutive before switching ────────
    if cfg.min_regime_bars > 1:
        smoothed = raw_regime.copy()
        for i in range(cfg.min_regime_bars, len(raw_regime)):
            window = raw_regime.iloc[i - cfg.min_regime_bars + 1: i + 1]
            if window.nunique() == 1:
                smoothed.iloc[i] = window.iloc[0]
            else:
                smoothed.iloc[i] = smoothed.iloc[i - 1]
        raw_regime = smoothed

    return raw_regime


# ── ML-based regime classifier ────────────────────────────────────────────────

def build_ml_regime_classifier(
    df:  pd.DataFrame,
    cfg: RegimeConfig,
) -> Optional[object]:
    """
    Train a simple XGBoost regime classifier.

    Labels derived from N-bar forward return:
        |forward_return| > threshold → TREND (clear direction)
        otherwise                   → RANGE (no direction)

    WARNING: labels computed from future returns.
    Use only on IS data. Never apply label computation to OOS data.
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.warning("xgboost not installed — falling back to rule-based regime")
        return None

    from src.ml.ml_position_sizer import SIZER_FEATURES, _avail

    # ── Compute future-return labels (only valid for training) ────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))
    fwd_ret = log_ret.shift(-cfg.ml_regime_lookforward).rolling(cfg.ml_regime_lookforward).sum()
    # Annualise
    bars_per_year = 365.0 * 24 * 3600 / max(
        (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / len(df), 1
    )
    fwd_ann = fwd_ret * np.sqrt(bars_per_year / cfg.ml_regime_lookforward)
    label   = (fwd_ann.abs() > cfg.ml_regime_threshold).astype(int)  # 1=trend, 0=range

    # ── Features: current bar context ─────────────────────────────────────────
    cols = _avail(df, SIZER_FEATURES)
    X    = df[cols].copy()
    y    = label

    # Drop NaN rows from both ends (lookforward creates NaN at end)
    valid = X.notna().all(axis=1) & y.notna()
    X, y  = X[valid], y[valid]

    if len(X) < 100 or len(y.unique()) < 2:
        logger.warning("Not enough data for ML regime classifier")
        return None

    split   = int(len(X) * 0.75)
    X_tr, X_vl = X.iloc[:split], X.iloc[split:]
    y_tr, y_vl = y.iloc[:split], y.iloc[split:]

    medians = X_tr.median()
    Xtr_c   = X_tr.fillna(medians)
    Xvl_c   = X_vl.fillna(medians)

    pos = y_tr.sum(); neg = len(y_tr) - pos
    spw = max(0.5, neg / pos) if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        early_stopping_rounds=20, eval_metric="auc",
        verbosity=0, random_state=42,
    )
    model.fit(Xtr_c, y_tr, eval_set=[(Xvl_c, y_vl)], verbose=False)

    auc = roc_auc_score(y_vl, model.predict_proba(Xvl_c)[:, 1]) \
          if len(y_vl.unique()) > 1 else 0.5
    logger.info(f"  ML regime classifier: val AUC={auc:.3f}")

    return {"model": model, "medians": medians, "cols": cols, "auc": auc}


def predict_ml_regime(
    df:         pd.DataFrame,
    classifier: dict,
    threshold:  float = 0.5,
) -> pd.Series:
    """Apply ML regime classifier to df. Returns Regime series."""
    model   = classifier["model"]
    medians = classifier["medians"]
    cols    = classifier["cols"]

    avail_cols = [c for c in cols if c in df.columns]
    X          = df[avail_cols].copy()
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[cols].fillna(medians)

    proba   = model.predict_proba(X)[:, 1]
    regime  = np.where(proba >= threshold, Regime.TREND, Regime.RANGE)
    return pd.Series(regime, index=df.index, dtype=int)


# ── Main RegimeFilter class ────────────────────────────────────────────────────

class RegimeFilter:
    """
    Classifies market regime and returns a size multiplier for each bar.

    Usage
    ─────
        rf = RegimeFilter(cfg_regime)
        # Optionally train ML classifier on IS
        rf.fit_ml(df_is)
        # Apply to full df
        size_mults = rf.get_size_multipliers(df)
        # size_mults is a Series [0.0, 0.7, 1.0] aligned with df.index
    """

    def __init__(self, cfg: RegimeConfig = None) -> None:
        self.cfg = cfg or RegimeConfig()
        self._ml_classifier: Optional[dict] = None

    def fit_ml(self, df_is: pd.DataFrame) -> bool:
        """Train ML regime classifier on IS data."""
        if not self.cfg.use_ml_regime:
            return False
        self._ml_classifier = build_ml_regime_classifier(df_is, self.cfg)
        return self._ml_classifier is not None

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """
        Return regime Series for each bar.
        Uses ML classifier if trained and cfg.use_ml_regime=True,
        otherwise rule-based.
        """
        if self.cfg.use_ml_regime and self._ml_classifier is not None:
            return predict_ml_regime(df, self._ml_classifier)
        return classify_regime_rules(df, self.cfg)

    def get_size_multipliers(self, df: pd.DataFrame) -> pd.Series:
        """
        Return size multiplier [0.0 – 1.0] for each bar.
        Applied AFTER the ML position sizer multiplier.
        """
        regime  = self.classify(df)
        mult    = pd.Series(self.cfg.trend_size_mult, index=df.index)
        mult[regime == Regime.RANGE]    = self.cfg.range_size_mult
        mult[regime == Regime.VOLATILE] = self.cfg.volatile_size_mult
        return mult

    def regime_stats(self, df: pd.DataFrame) -> dict:
        """Return regime distribution statistics."""
        regime = self.classify(df)
        total  = len(regime)
        counts = regime.value_counts()
        return {
            "trend_pct":    counts.get(Regime.TREND,    0) / total,
            "range_pct":    counts.get(Regime.RANGE,    0) / total,
            "volatile_pct": counts.get(Regime.VOLATILE, 0) / total,
            "total_bars":   total,
        }