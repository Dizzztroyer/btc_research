"""
ml_position_sizer.py
─────────────────────
XGBoost-based dynamic position sizer.

Instead of binary enter/skip, maps ML score → position size multiplier:
    score 0.0 → size_min  (0.5x)
    score 0.5 → size_base (1.0x)
    score 1.0 → size_max  (1.5x)

No trades are skipped. Low-confidence signals trade smaller.
High-confidence signals trade larger.

This preserves trade count while improving risk-adjusted returns.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from src.utils.config_loader import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Feature set ───────────────────────────────────────────────────────────────

SIZER_FEATURES = [
    # Momentum
    "ret_1", "ret_2", "ret_5", "ret_10",
    "log_ret_1", "log_ret_5",
    "rsi_7", "rsi_14", "rsi_21",
    # Trend strength
    "adx_14", "adx_10",
    "dist_ema_21", "dist_ema_55", "dist_ema_200",
    "ema_21_55_above", "ema_55_200_above",
    # Volatility
    "atr_14_pct", "atr_7_pct",
    "vol_20", "vol_40",
    "bb20s20_width", "bb20s20_pctb",
    "range_ratio_10", "range_ratio_20",
    "squeeze_10", "squeeze_20",
    # Structure
    "don_20_dist_high", "don_20_dist_low",
    "don_40_dist_high", "don_40_dist_low",
    "breakout_up_20", "breakout_down_20",
    "high_break_20", "low_break_20",
    # Volume
    "vol_zscore_20", "vol_zscore_40",
    # Candle
    "body_pct", "upper_wick", "lower_wick",
    "candle_range", "is_bullish",
    # Regime
    "regime_trend", "regime_range", "regime_vol",
]


def _avail(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _norm_ts(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    return ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")


def _ts_lookup(df: pd.DataFrame) -> dict:
    ts = _norm_ts(df["timestamp"])
    return {t: i for i, t in zip(df.index, ts)}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SizerConfig:
    # Position size range
    size_min:   float = 0.5    # multiplier for lowest-confidence signal
    size_base:  float = 1.0    # multiplier at score = 0.5 (neutral)
    size_max:   float = 1.5    # multiplier for highest-confidence signal

    # XGBoost
    n_estimators:    int   = 400
    max_depth:       int   = 4
    learning_rate:   float = 0.04
    subsample:       float = 0.75
    colsample:       float = 0.75
    min_child_weight:int   = 8
    early_stopping:  int   = 40

    # Sample weighting (emphasise large wins/losses)
    weight_by_magnitude: bool  = True
    weight_scale:        float = 2.0
    weight_floor:        float = 0.4

    # Training
    is_train_ratio:   float = 0.75
    min_train_trades: int   = 50


# ── Score → size mapping ──────────────────────────────────────────────────────

def score_to_size(scores: np.ndarray, cfg: SizerConfig) -> np.ndarray:
    """
    Map XGBoost probability scores to position size multipliers.

    Linear interpolation:
        score = 0.0  →  size_min
        score = 0.5  →  size_base
        score = 1.0  →  size_max

    Clipped to [size_min, size_max].
    """
    sizes = np.where(
        scores >= 0.5,
        cfg.size_base + (scores - 0.5) * 2 * (cfg.size_max - cfg.size_base),
        cfg.size_min  + scores          * 2 * (cfg.size_base - cfg.size_min),
    )
    return np.clip(sizes, cfg.size_min, cfg.size_max)


# ── Sample weights ────────────────────────────────────────────────────────────

def _sample_weights(pnls: np.ndarray, labels: np.ndarray, cfg: SizerConfig) -> np.ndarray:
    """Higher weight for large magnitude trades (clear signal)."""
    if not cfg.weight_by_magnitude:
        return np.ones(len(pnls))
    abs_pnl = np.abs(pnls)
    if abs_pnl.max() == 0:
        return np.ones(len(pnls))
    norm = abs_pnl / abs_pnl.max()
    w = cfg.weight_floor + (1.0 - cfg.weight_floor) * norm
    # Extra weight for large losers ("don't do this strongly")
    w[labels == 0] *= cfg.weight_scale
    return w / w.mean()


# ── Main class ────────────────────────────────────────────────────────────────

class MLPositionSizer:
    """
    Trains XGBoost on IS trade outcomes, then assigns a size multiplier
    to each signal bar on OOS/live data.

    Usage
    ─────
        sizer = MLPositionSizer(cfg, sizer_cfg)
        sizer.fit(strategy, params, df_is)
        size_col = sizer.predict_sizes(df_oos, strategy, params)
        # size_col is a Series aligned with df_oos index
    """

    def __init__(self, cfg: Config, sizer_cfg: Optional[SizerConfig] = None) -> None:
        self.cfg   = cfg
        self.scfg  = sizer_cfg or SizerConfig()
        self.model: Optional[xgb.XGBClassifier] = None
        self.medians: Optional[pd.Series]        = None
        self.feature_cols: List[str]             = []
        self.auc:   float = 0.5
        self.is_fitted = False

        if not ML_AVAILABLE:
            raise ImportError("pip install xgboost scikit-learn")

    def _get_trade_data(
        self,
        strategy,
        params: dict,
        df:     pd.DataFrame,
        direction: str = "both",
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Run backtest on df, return (X_features, y_labels, pnls)."""
        from src.backtest.engine import BacktestEngine, SimConfig
        sim    = SimConfig(
            fees=self.cfg.fees, slippage=self.cfg.slippage,
            leverage=self.cfg.leverage, risk_per_trade=self.cfg.risk_per_trade,
            direction=direction,
        )
        df_sig = strategy.generate_signals(df.copy(), params)
        result = BacktestEngine(sim).run(df_sig, strategy.name, "", params)

        if result.trades.empty:
            return pd.DataFrame(), pd.Series(dtype=int), np.array([])

        trades         = result.trades.copy()
        trades["label"]= (trades["pnl"] > 0).astype(int)

        entry_times = _norm_ts(trades["entry_time"])
        lookup      = _ts_lookup(df_sig)
        entry_idx   = [lookup.get(et) for et in entry_times]

        valid = [(i, l, p) for i, l, p in
                 zip(entry_idx, trades["label"], trades["pnl"])
                 if i is not None]
        if not valid:
            return pd.DataFrame(), pd.Series(dtype=int), np.array([])

        indices, labels, pnls = zip(*valid)
        indices = list(indices)
        labels  = np.array(list(labels))
        pnls    = np.array(list(pnls))

        cols = _avail(df_sig, SIZER_FEATURES)
        X    = df_sig[cols].shift(1).loc[indices].copy()
        y    = pd.Series(labels, index=indices, name="label")

        common = X.index.intersection(y.index)
        X, y   = X.loc[common], y.loc[common]

        idx_map  = {orig: i for i, orig in enumerate(indices)}
        pnl_arr  = np.array([pnls[idx_map[i]] for i in common])

        return X, y, pnl_arr

    def fit(
        self,
        strategy,
        params:    dict,
        df_is:     pd.DataFrame,
        direction: str = "both",
    ) -> float:
        """
        Train on IS period. Returns validation AUC.
        """
        logger.info(f"  Sizer: fitting on IS ({len(df_is)} bars)...")
        X, y, pnls = self._get_trade_data(strategy, params, df_is, direction)

        if len(X) < self.scfg.min_train_trades:
            logger.warning(f"  Only {len(X)} IS trades — sizer not trained")
            return 0.5

        if len(y.unique()) < 2:
            logger.warning("  Single class in IS — sizer not trained")
            return 0.5

        split    = int(len(X) * self.scfg.is_train_ratio)
        X_tr, X_vl = X.iloc[:split], X.iloc[split:]
        y_tr, y_vl = y.iloc[:split], y.iloc[split:]
        pnl_tr     = pnls[:split]

        if len(X_tr) < 10 or len(X_vl) < 5:
            logger.warning("  Train/val too small")
            return 0.5

        if len(y_tr.unique()) < 2 or len(y_vl.unique()) < 2:
            logger.warning("  Not enough class diversity")
            return 0.5

        self.medians      = X_tr.median()
        self.feature_cols = list(X.columns)

        Xtr_c = X_tr.fillna(self.medians)
        Xvl_c = X_vl.fillna(self.medians)

        pos = y_tr.sum(); neg = len(y_tr) - pos
        spw = max(0.5, neg / pos) if pos > 0 else 1.0

        weights = _sample_weights(pnl_tr, y_tr.values, self.scfg)

        self.model = xgb.XGBClassifier(
            n_estimators          = self.scfg.n_estimators,
            max_depth             = self.scfg.max_depth,
            learning_rate         = self.scfg.learning_rate,
            subsample             = self.scfg.subsample,
            colsample_bytree      = self.scfg.colsample,
            min_child_weight      = self.scfg.min_child_weight,
            scale_pos_weight      = spw,
            early_stopping_rounds = self.scfg.early_stopping,
            eval_metric           = "auc",
            verbosity             = 0,
            random_state          = 42,
        )
        self.model.fit(
            Xtr_c, y_tr,
            eval_set         = [(Xvl_c, y_vl)],
            sample_weight    = weights,
            verbose          = False,
        )

        y_pred    = self.model.predict_proba(Xvl_c)[:, 1]
        self.auc  = roc_auc_score(y_vl, y_pred) if len(y_vl.unique()) > 1 else 0.5
        self.is_fitted = True

        logger.info(
            f"  Sizer fitted | IS trades={len(X)} | "
            f"win_rate={y.mean():.1%} | AUC={self.auc:.3f}"
        )
        return self.auc

    def predict_sizes(
        self,
        df:        pd.DataFrame,
        strategy,
        params:    dict,
    ) -> pd.Series:
        """
        For every signal bar in df, return a size multiplier Series.
        Non-signal bars get multiplier 1.0 (neutral, won't be used).
        """
        if not self.is_fitted or self.model is None:
            return pd.Series(1.0, index=df.index)

        df_sig      = strategy.generate_signals(df.copy(), params)
        signal_mask = df_sig["signal"] != 0

        sizes = pd.Series(1.0, index=df.index)
        if not signal_mask.any():
            return sizes

        sig_idx = df_sig.index[signal_mask]
        cols    = _avail(df_sig, self.feature_cols)
        X       = df_sig[cols].shift(1).loc[sig_idx].copy()

        # Add any missing cols as NaN
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.feature_cols]

        X_clean = X.fillna(self.medians)
        try:
            scores      = self.model.predict_proba(X_clean)[:, 1]
            multipliers = score_to_size(scores, self.scfg)
            sizes.loc[sig_idx] = multipliers
        except Exception as exc:
            logger.warning(f"  Sizer prediction failed: {exc}")

        return sizes

    @property
    def feature_importances(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)