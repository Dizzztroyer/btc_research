"""
ml_filter.py  v2
────────────────
XGBoost signal filter — improved version.

Key improvements over v1
────────────────────────
1. Auto-threshold calibration
   - Sweeps thresholds 0.40-0.65 on IS validation set
   - Picks the threshold that maximises Sharpe on IS val
   - Applies that threshold to OOS (no lookahead)

2. Weighted negative examples ("teach what NOT to do")
   - Trades with large losses get higher sample weight
   - Trades with large wins get higher sample weight
   - Near-zero PnL trades get lower weight (ambiguous signal)
   - This focuses the model on confident cases

3. Better feature set
   - Adds regime context at signal bar
   - Adds momentum confirmation features
   - Adds volatility regime features
   - Removes low-signal calendar features from primary set

4. Better IS size handling
   - Warns and skips if IS trades < min_train_trades
   - 1d TF needs 200+ bars minimum — handled gracefully

5. Cleaner OOS comparison
   - Both base and filtered run on identical OOS slice
   - Metrics are comparable apples-to-apples
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.backtest.engine import BacktestEngine, BacktestResult, SimConfig
from src.backtest.metrics import compute_metrics
from src.utils.config_loader import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Feature groups ────────────────────────────────────────────────────────────

# Primary features — most predictive for trade outcome
PRIMARY_FEATURES = [
    # Momentum at entry
    "ret_1", "ret_2", "ret_5", "ret_10",
    "log_ret_1", "log_ret_5",
    "rsi_7", "rsi_14", "rsi_21",
    # Trend strength
    "adx_14", "adx_10",
    "dist_ema_21", "dist_ema_55", "dist_ema_200",
    "ema_21_55_above", "ema_55_200_above",
    # Volatility context
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
    # Candle anatomy
    "body_pct", "upper_wick", "lower_wick",
    "candle_range", "is_bullish",
    # Regime
    "regime_trend", "regime_range", "regime_vol",
]

# Calendar — lower priority, add if enough data
CALENDAR_FEATURES = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "session_asia", "session_europe", "session_us",
]


def _available(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class MLFilterConfig:
    # Threshold — set to None for auto-calibration (recommended)
    threshold:            Optional[float] = None
    threshold_search:     List[float]     = field(
        default_factory=lambda: [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
    )
    # XGBoost params
    n_estimators:         int   = 400
    max_depth:            int   = 4
    learning_rate:        float = 0.04
    subsample:            float = 0.75
    colsample:            float = 0.75
    min_child_weight:     int   = 8
    early_stopping:       int   = 40
    # Sample weighting ("teach what NOT to do")
    use_sample_weights:   bool  = True
    weight_by_pnl:        bool  = True    # large loss/win → higher weight
    pnl_weight_scale:     float = 2.0     # scale factor for PnL-based weights
    min_pnl_weight:       float = 0.3     # floor weight for near-zero PnL trades
    # Training
    is_train_ratio:       float = 0.75
    min_train_trades:     int   = 60      # skip if IS has fewer trades
    min_val_trades:       int   = 8       # skip if val set has fewer trades
    # Feature selection
    use_calendar_features: bool = True
    importance_threshold: float = 0.005   # drop features below this importance


# ── Sample weight computation ─────────────────────────────────────────────────

def _compute_sample_weights(
    pnls:  np.ndarray,
    labels: np.ndarray,
    cfg:   MLFilterConfig,
) -> np.ndarray:
    """
    Compute per-trade sample weights.

    Logic:
    - Base weight = 1.0 for all trades
    - Trades with large absolute PnL get higher weight (clear signal)
    - Trades with near-zero PnL get lower weight (ambiguous)
    - This is the 'teach what NOT to do' principle:
      a trade that lost -5% is a much stronger 'do NOT do this'
      signal than a trade that lost -0.1%
    """
    if not cfg.use_sample_weights or not cfg.weight_by_pnl:
        return np.ones(len(pnls))

    abs_pnl = np.abs(pnls)
    if abs_pnl.max() == 0:
        return np.ones(len(pnls))

    # Normalise to [0, 1]
    norm_pnl = abs_pnl / abs_pnl.max()

    # Weight: large |PnL| → high weight, near-zero → low weight
    raw_weights = cfg.min_pnl_weight + (1.0 - cfg.min_pnl_weight) * norm_pnl

    # Scale winners and losers separately
    # Losers get extra weight (stronger "don't do this" signal)
    loser_mask = labels == 0
    raw_weights[loser_mask] *= cfg.pnl_weight_scale

    return raw_weights / raw_weights.mean()  # normalise to mean=1


# ── Timestamp matching ────────────────────────────────────────────────────────

def _build_ts_lookup(df: pd.DataFrame) -> dict:
    """Build UTC timestamp → df_index lookup."""
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    return {t: i for i, t in zip(df.index, ts)}


def _normalise_ts(ts_series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts_series)
    if ts.dt.tz is None:
        return ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("UTC")


# ── Feature matrix ────────────────────────────────────────────────────────────

def _build_feature_matrix(
    df:          pd.DataFrame,
    signal_idx:  list,
    use_calendar: bool = True,
) -> pd.DataFrame:
    """
    Build feature matrix for given bar indices.
    Shifts features by 1 bar to avoid lookahead.
    """
    cols = _available(df, PRIMARY_FEATURES)
    if use_calendar:
        cols += _available(df, CALENDAR_FEATURES)

    if not cols:
        raise ValueError("No ML features in DataFrame. Run build_features.py first.")

    feat_df = df[cols].shift(1)
    return feat_df.loc[signal_idx].copy()


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class MLFilterResult:
    timeframe:     str
    strategy_name: str
    threshold:     float
    auto_threshold: bool

    base_metrics:  dict
    base_trades:   int
    filt_metrics:  dict
    filt_trades:   int

    auc_score:     float
    feature_importances:   pd.DataFrame
    threshold_search_df:   pd.DataFrame  # IS val performance at each threshold
    trade_scores:          pd.DataFrame

    def improvement_summary(self) -> str:
        auto_str = " (auto-calibrated)" if self.auto_threshold else ""
        lines = [
            f"\n{'='*60}",
            f"ML Filter v2 — {self.strategy_name} / {self.timeframe}",
            f"{'='*60}",
            f"Threshold : {self.threshold:.3f}{auto_str}",
            f"AUC score : {self.auc_score:.3f}",
            f"",
            f"{'Metric':<24} {'Base':>10} {'Filtered':>10} {'Delta':>10}",
            f"{'-'*60}",
        ]
        metrics_to_show = [
            ("profit_factor", "Profit Factor",  False),
            ("sharpe",        "Sharpe Ratio",   False),
            ("total_return",  "Total Return",   True),
            ("max_drawdown",  "Max Drawdown",   True),
            ("win_rate",      "Win Rate",       True),
            ("trade_count",   "Trade Count",    False),
            ("expectancy",    "Expectancy ($)", False),
            ("calmar",        "Calmar Ratio",   False),
        ]
        for key, label, is_pct in metrics_to_show:
            b = self.base_metrics.get(key, np.nan)
            f = self.filt_metrics.get(key, np.nan)
            if pd.isna(b) or pd.isna(f):
                lines.append(f"{label:<24} {'—':>10} {'—':>10} {'—':>10}")
                continue
            delta = f - b
            sign  = "+" if delta > 0 else ""
            def _v(v):
                return f"{v:.2%}" if is_pct else f"{v:.3f}"
            lines.append(f"{label:<24} {_v(b):>10} {_v(f):>10} {sign}{_v(delta):>10}")
        lines.append(f"{'='*60}")

        if not self.threshold_search_df.empty:
            lines.append("\nIS Val threshold search (top 5 by Sharpe):")
            top5 = self.threshold_search_df.sort_values("val_sharpe", ascending=False).head(5)
            for _, r in top5.iterrows():
                marker = " ← chosen" if abs(r["threshold"] - self.threshold) < 0.001 else ""
                lines.append(
                    f"  thresh={r['threshold']:.2f}  "
                    f"Sh={r['val_sharpe']:+.3f}  "
                    f"PF={r['val_pf']:.3f}  "
                    f"trades={int(r['val_trades'])}{marker}"
                )

        return "\n".join(lines)


# ── Main class ────────────────────────────────────────────────────────────────

class MLFilter:
    """
    XGBoost signal filter v2.

    Key behaviours:
    - Automatically calibrates threshold on IS validation data
    - Weights negative examples by magnitude (large losses = stronger signal)
    - Uses expanding IS window in walk-forward mode
    - Provides clear IS val threshold sweep for interpretation
    """

    def __init__(self, cfg: Config, ml_cfg: Optional[MLFilterConfig] = None) -> None:
        self.cfg    = cfg
        self.ml_cfg = ml_cfg or MLFilterConfig()
        if not XGB_AVAILABLE:
            raise ImportError("pip install xgboost")
        if not SKLEARN_AVAILABLE:
            raise ImportError("pip install scikit-learn")

    def _sim(self, direction: str = "both") -> SimConfig:
        return SimConfig(
            fees=self.cfg.fees, slippage=self.cfg.slippage,
            leverage=self.cfg.leverage, risk_per_trade=self.cfg.risk_per_trade,
            direction=direction,
        )

    def _split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        a = int(n * self.cfg.validation.is_ratio)
        b = int(n * (self.cfg.validation.is_ratio + self.cfg.validation.oos_ratio))
        return df.iloc[:a], df.iloc[a:b], df.iloc[b:]

    # ── Get trade data with labels ─────────────────────────────────────────────

    def _get_labeled_trades(
        self,
        strategy,
        params:    dict,
        df:        pd.DataFrame,
        direction: str = "both",
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Returns:
            X      : feature matrix at entry bars
            y      : binary labels (1=profit, 0=loss)
            weights: sample weights
        """
        df_sig = strategy.generate_signals(df.copy(), params)
        engine = BacktestEngine(self._sim(direction))
        result = engine.run(df_sig, strategy.name, "", params)

        if result.trades.empty:
            return pd.DataFrame(), pd.Series(dtype=int), np.array([])

        trades        = result.trades.copy()
        trades["label"] = (trades["pnl"] > 0).astype(int)

        # Timestamp matching with tz normalisation
        entry_times = _normalise_ts(trades["entry_time"])
        ts_lookup   = _build_ts_lookup(df_sig)

        entry_indices = [ts_lookup.get(et) for et in entry_times]
        valid = [(idx, lbl, pnl)
                 for idx, lbl, pnl in zip(entry_indices, trades["label"], trades["pnl"])
                 if idx is not None]

        if not valid:
            return pd.DataFrame(), pd.Series(dtype=int), np.array([])

        indices, labels, pnls = zip(*valid)
        indices = list(indices)
        labels  = np.array(list(labels))
        pnls    = np.array(list(pnls))

        X = _build_feature_matrix(df_sig, indices, self.ml_cfg.use_calendar_features)
        y = pd.Series(labels, index=indices, name="label")

        common  = X.index.intersection(y.index)
        X, y    = X.loc[common], y.loc[common]

        # Align pnls to common index
        idx_map = {orig: i for i, orig in enumerate(indices)}
        pnl_arr = np.array([pnls[idx_map[i]] for i in common])

        weights = _compute_sample_weights(pnl_arr, y.values, self.ml_cfg)
        return X, y, weights

    # ── Train model ────────────────────────────────────────────────────────────

    def _train(
        self,
        X_train: pd.DataFrame, y_train: pd.Series, w_train: np.ndarray,
        X_val:   pd.DataFrame, y_val:   pd.Series,
    ) -> Tuple[xgb.XGBClassifier, float, pd.Series]:
        """Train XGBoost. Returns (model, val_auc, fill_medians)."""
        medians     = X_train.median()
        Xtr_clean   = X_train.fillna(medians)
        Xvl_clean   = X_val.fillna(medians)

        # Class imbalance
        pos = y_train.sum()
        neg = len(y_train) - pos
        spw = max(0.5, neg / pos) if pos > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators          = self.ml_cfg.n_estimators,
            max_depth             = self.ml_cfg.max_depth,
            learning_rate         = self.ml_cfg.learning_rate,
            subsample             = self.ml_cfg.subsample,
            colsample_bytree      = self.ml_cfg.colsample,
            min_child_weight      = self.ml_cfg.min_child_weight,
            scale_pos_weight      = spw,
            early_stopping_rounds = self.ml_cfg.early_stopping,
            eval_metric           = "auc",
            verbosity             = 0,
            random_state          = 42,
        )

        fit_kwargs = {"eval_set": [(Xvl_clean, y_val)], "verbose": False}
        if self.ml_cfg.use_sample_weights and w_train is not None and len(w_train) == len(Xtr_clean):
            fit_kwargs["sample_weight"] = w_train

        model.fit(Xtr_clean, y_train, **fit_kwargs)

        y_pred  = model.predict_proba(Xvl_clean)[:, 1]
        auc     = roc_auc_score(y_val, y_pred) if len(y_val.unique()) > 1 else 0.5
        return model, auc, medians

    # ── Auto-threshold calibration ─────────────────────────────────────────────

    def _calibrate_threshold(
        self,
        model:   xgb.XGBClassifier,
        medians: pd.Series,
        X_val:   pd.DataFrame,
        y_val:   pd.Series,
        pnls_val: np.ndarray,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Find optimal threshold on IS validation data.

        For each candidate threshold:
        - Compute which trades would be taken
        - Compute Sharpe-like score on those trades
        - Pick threshold with best Sharpe, minimum trades

        Returns (best_threshold, search_df)
        """
        Xvl_clean = X_val.fillna(medians)
        scores    = model.predict_proba(Xvl_clean)[:, 1]

        # Use a lower minimum for the sweep itself (we already validated IS size earlier)
        sweep_min = max(3, self.ml_cfg.min_val_trades // 3)
        rows = []
        for t in self.ml_cfg.threshold_search:
            mask = scores >= t
            n    = mask.sum()
            if n < sweep_min:
                rows.append({"threshold": t, "val_trades": n, "val_pf": np.nan,
                             "val_sharpe": np.nan, "val_winrate": np.nan})
                continue

            taken_pnl = pnls_val[mask]
            wins      = (taken_pnl > 0).sum()
            gp        = taken_pnl[taken_pnl > 0].sum()
            gl        = abs(taken_pnl[taken_pnl < 0].sum())
            pf        = gp / gl if gl > 0 else np.nan
            wr        = wins / n

            # Sharpe-like: mean/std of trade PnLs
            sh = taken_pnl.mean() / taken_pnl.std() * np.sqrt(252) if taken_pnl.std() > 0 else np.nan
            # Penalise if fewer than 60% of base trades remain (too aggressive)
            trade_pct = n / max(len(scores), 1)
            if trade_pct < 0.20:
                sh = sh * 0.5 if not np.isnan(sh) else np.nan

            rows.append({
                "threshold":   t,
                "val_trades":  n,
                "val_pf":      pf,
                "val_sharpe":  sh,
                "val_winrate": wr,
                "pct_taken":   trade_pct,
            })

        search_df = pd.DataFrame(rows)

        # Best threshold: highest val_sharpe with enough trades
        valid = search_df[search_df["val_sharpe"].notna() & (search_df["val_trades"] >= self.ml_cfg.min_val_trades)]
        if valid.empty:
            # No threshold yielded enough trades — model too uncertain, use permissive default
            logger.warning("    No threshold found with enough val trades — defaulting to 0.48")
            best_thresh = 0.48  # slightly permissive fallback
        else:
            best_thresh = float(valid.loc[valid["val_sharpe"].idxmax(), "threshold"])

        if not valid.empty:
            best_sh_val = valid.loc[valid['threshold']==best_thresh,'val_sharpe'].values
            sh_str = f"{best_sh_val[0]:.3f}" if len(best_sh_val)>0 else 'n/a'
        else:
            sh_str = 'n/a'
        logger.info(f"    Auto-threshold: {best_thresh:.3f} (val_Sharpe={sh_str})")
        return best_thresh, search_df

    # ── Apply filter ───────────────────────────────────────────────────────────

    def _apply(
        self,
        df:        pd.DataFrame,
        strategy,
        params:    dict,
        model:     xgb.XGBClassifier,
        medians:   pd.Series,
        threshold: float,
        direction: str = "both",
    ) -> pd.DataFrame:
        df_sig      = strategy.generate_signals(df.copy(), params)
        signal_mask = df_sig["signal"] != 0
        if not signal_mask.any():
            return df_sig

        sig_idx  = df_sig.index[signal_mask]
        X        = _build_feature_matrix(df_sig, list(sig_idx), self.ml_cfg.use_calendar_features)
        X_clean  = X.fillna(medians)

        try:
            scores = model.predict_proba(X_clean)[:, 1]
        except Exception:
            return df_sig

        suppress = sig_idx[scores < threshold]
        df_sig.loc[suppress, "signal"]   = 0
        df_sig.loc[suppress, "sl_price"] = np.nan
        df_sig.loc[suppress, "tp_price"] = np.nan
        return df_sig

    # ── Public: single run ─────────────────────────────────────────────────────

    def run(
        self,
        strategy,
        params:    dict,
        df:        pd.DataFrame,
        timeframe: str,
        direction: str = "both",
        threshold: Optional[float] = None,
    ) -> MLFilterResult:
        """Full OOS evaluation: train on IS, evaluate on OOS."""
        use_auto  = (threshold is None and self.ml_cfg.threshold is None)
        threshold = threshold or self.ml_cfg.threshold  # may still be None → auto

        logger.info(f"ML Filter v2: {strategy.name}/{timeframe} | "
                    f"threshold={'auto' if use_auto else threshold}")

        df_is, df_oos, _ = self._split(df)
        engine = BacktestEngine(self._sim(direction))

        # ── IS: get labels ─────────────────────────────────────────────────────
        logger.info(f"  IS: {len(df_is)} bars")
        X_is, y_is, w_is = self._get_labeled_trades(strategy, params, df_is, direction)

        if len(X_is) < self.ml_cfg.min_train_trades:
            logger.warning(f"  Too few IS trades ({len(X_is)} < {self.ml_cfg.min_train_trades})")
            return self._empty(strategy.name, timeframe, threshold or 0.50, use_auto)

        logger.info(f"  IS trades: {len(X_is)} | win rate: {y_is.mean():.1%}")

        if len(y_is.unique()) < 2:
            logger.warning("  IS has only one class — cannot train")
            return self._empty(strategy.name, timeframe, threshold or 0.50, use_auto)

        # ── Train / val split within IS ────────────────────────────────────────
        split     = int(len(X_is) * self.ml_cfg.is_train_ratio)
        X_tr, X_vl = X_is.iloc[:split], X_is.iloc[split:]
        y_tr, y_vl = y_is.iloc[:split], y_is.iloc[split:]
        w_tr       = w_is[:split] if w_is is not None else None

        if len(X_tr) < 20 or len(X_vl) < self.ml_cfg.min_val_trades:
            logger.warning(f"  Train ({len(X_tr)}) or val ({len(X_vl)}) too small")
            return self._empty(strategy.name, timeframe, threshold or 0.50, use_auto)

        if len(y_tr.unique()) < 2 or len(y_vl.unique()) < 2:
            logger.warning("  Not enough class diversity in train/val")
            return self._empty(strategy.name, timeframe, threshold or 0.50, use_auto)

        # ── Train XGBoost ──────────────────────────────────────────────────────
        logger.info(f"  Training XGBoost (n_estimators={self.ml_cfg.n_estimators})...")
        model, auc, medians = self._train(X_tr, y_tr, w_tr, X_vl, y_vl)
        logger.info(f"  Val AUC: {auc:.3f}")

        # ── Feature importances ────────────────────────────────────────────────
        importances = pd.DataFrame({
            "feature":    X_is.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        # ── Get val PnLs for threshold calibration ─────────────────────────────
        # Re-run backtest on IS val period to get PnLs
        df_is_val = df_is.iloc[split:]
        _, _, w_vl2 = self._get_labeled_trades(strategy, params, df_is_val, direction)
        # Get raw pnls from val trades
        df_is_val_sig = strategy.generate_signals(df_is_val.copy(), params)
        res_val = engine.run(df_is_val_sig, strategy.name, timeframe, params)
        pnls_val = res_val.trades["pnl"].values if not res_val.trades.empty else np.array([])

        # Align scores to val trades
        Xvl_clean = X_vl.fillna(medians)
        val_scores = model.predict_proba(Xvl_clean)[:, 1]
        n_align = min(len(val_scores), len(pnls_val))
        pnls_aligned = pnls_val[:n_align] if n_align > 0 else np.zeros(len(val_scores))
        if len(pnls_aligned) < len(val_scores):
            pnls_aligned = np.concatenate([pnls_aligned,
                                           np.zeros(len(val_scores)-len(pnls_aligned))])

        # ── Auto-threshold calibration ─────────────────────────────────────────
        search_df = pd.DataFrame()
        if use_auto:
            threshold, search_df = self._calibrate_threshold(
                model, medians, X_vl, y_vl, pnls_aligned
            )
        elif threshold is None:
            threshold = 0.50

        # ── OOS: base vs filtered ──────────────────────────────────────────────
        logger.info(f"  OOS evaluation (threshold={threshold:.3f})...")
        df_oos_base = strategy.generate_signals(df_oos.copy(), params)
        res_base    = engine.run(df_oos_base, strategy.name, timeframe, params)

        df_oos_filt = self._apply(df_oos, strategy, params, model, medians, threshold, direction)
        res_filt    = engine.run(df_oos_filt, strategy.name, timeframe, params)

        logger.info(
            f"  Base  : trades={res_base.metrics.get('trade_count',0)} | "
            f"PF={res_base.metrics.get('profit_factor',0):.3f} | "
            f"Sh={res_base.metrics.get('sharpe',0):.3f}"
        )
        logger.info(
            f"  Filt  : trades={res_filt.metrics.get('trade_count',0)} | "
            f"PF={res_filt.metrics.get('profit_factor',0):.3f} | "
            f"Sh={res_filt.metrics.get('sharpe',0):.3f}"
        )

        # ── OOS trade scores ───────────────────────────────────────────────────
        oos_sig_mask = df_oos_base["signal"] != 0
        trade_scores = pd.DataFrame()
        if oos_sig_mask.any():
            oos_X = _build_feature_matrix(
                df_oos_base, list(df_oos_base.index[oos_sig_mask]),
                self.ml_cfg.use_calendar_features
            )
            try:
                oos_sc = model.predict_proba(oos_X.fillna(medians))[:, 1]
                trade_scores = pd.DataFrame({
                    "score":  oos_sc,
                    "signal": df_oos_base.loc[oos_sig_mask, "signal"].values,
                    "taken":  (oos_sc >= threshold).astype(int),
                })
            except Exception:
                pass

        return MLFilterResult(
            timeframe             = timeframe,
            strategy_name         = strategy.name,
            threshold             = threshold,
            auto_threshold        = use_auto,
            base_metrics          = res_base.metrics,
            base_trades           = res_base.metrics.get("trade_count", 0),
            filt_metrics          = res_filt.metrics,
            filt_trades           = res_filt.metrics.get("trade_count", 0),
            auc_score             = auc,
            feature_importances   = importances,
            threshold_search_df   = search_df,
            trade_scores          = trade_scores,
        )

    # ── Public: walk-forward ───────────────────────────────────────────────────

    def run_walk_forward(
        self,
        strategy,
        params:    dict,
        df:        pd.DataFrame,
        timeframe: str,
        direction: str = "both",
        threshold: Optional[float] = None,
        n_windows: int = 5,
    ) -> Dict[str, Any]:
        """
        Walk-forward ML evaluation with expanding IS window.

        IS grows with each window → model always trains on maximum available data.
        Threshold is auto-calibrated per window on IS val set.
        """
        use_auto  = (threshold is None and self.ml_cfg.threshold is None)
        n         = len(df)
        is_pool   = int(n * self.cfg.validation.is_ratio)
        oos_total = n - is_pool
        oos_win   = max(50, oos_total // n_windows)

        engine   = BacktestEngine(self._sim(direction))
        wf_rows  = []

        for i in range(n_windows):
            oos_start = is_pool + i * oos_win
            oos_end   = oos_start + oos_win if i < n_windows - 1 else n
            if oos_start >= n or oos_end > n:
                break

            # Expanding IS: all data before this OOS window
            df_is  = df.iloc[:oos_start]
            df_oos = df.iloc[oos_start:oos_end]

            logger.info(
                f"  WF {i+1}/{n_windows} | "
                f"IS: {df_is['timestamp'].iloc[0].date()} → {df_is['timestamp'].iloc[-1].date()} | "
                f"OOS: {df_oos['timestamp'].iloc[0].date()} → {df_oos['timestamp'].iloc[-1].date()}"
            )

            if len(df_is) < 300 or len(df_oos) < 20:
                logger.warning(f"  Window {i+1}: too small, skipping")
                continue

            # Get IS labels
            X_is, y_is, w_is = self._get_labeled_trades(strategy, params, df_is, direction)
            if len(X_is) < self.ml_cfg.min_train_trades:
                logger.warning(f"  Window {i+1}: only {len(X_is)} IS trades, skipping")
                continue
            if len(y_is.unique()) < 2:
                continue

            split    = int(len(X_is) * self.ml_cfg.is_train_ratio)
            if split < 20 or (len(X_is) - split) < self.ml_cfg.min_val_trades:
                continue

            X_tr, X_vl = X_is.iloc[:split], X_is.iloc[split:]
            y_tr, y_vl = y_is.iloc[:split], y_is.iloc[split:]
            w_tr       = w_is[:split] if w_is is not None and len(w_is) >= split else None

            if len(y_tr.unique()) < 2 or len(y_vl.unique()) < 2:
                continue

            try:
                model, auc, medians = self._train(X_tr, y_tr, w_tr, X_vl, y_vl)
            except Exception as exc:
                logger.warning(f"  Window {i+1}: training failed: {exc}")
                continue

            # Threshold for this window
            win_thresh = threshold
            if use_auto or win_thresh is None:
                # Get val PnLs
                df_val_sig  = strategy.generate_signals(df_is.iloc[split:].copy(), params)
                res_val_bt  = engine.run(df_val_sig, strategy.name, timeframe, params)
                pnls_val    = res_val_bt.trades["pnl"].values if not res_val_bt.trades.empty else np.array([])

                Xvl_sc      = model.predict_proba(X_vl.fillna(medians))[:, 1]
                n_al        = min(len(Xvl_sc), len(pnls_val))
                pnls_al     = np.concatenate([pnls_val[:n_al], np.zeros(max(0,len(Xvl_sc)-n_al))])

                win_thresh, _ = self._calibrate_threshold(model, medians, X_vl, y_vl, pnls_al)
            else:
                win_thresh = float(win_thresh)

            # OOS evaluation
            df_oos_sig  = strategy.generate_signals(df_oos.copy(), params)
            res_base    = engine.run(df_oos_sig, strategy.name, timeframe, params)
            df_oos_filt = self._apply(df_oos, strategy, params, model, medians, win_thresh, direction)
            res_filt    = engine.run(df_oos_filt, strategy.name, timeframe, params)

            bm = res_base.metrics
            fm = res_filt.metrics

            wf_rows.append({
                "window":        i + 1,
                "auc":           auc,
                "threshold":     win_thresh,
                "base_pf":       bm.get("profit_factor", np.nan),
                "base_sharpe":   bm.get("sharpe",        np.nan),
                "base_return":   bm.get("total_return",  np.nan),
                "base_mdd":      bm.get("max_drawdown",  np.nan),
                "base_trades":   bm.get("trade_count",   0),
                "filt_pf":       fm.get("profit_factor", np.nan),
                "filt_sharpe":   fm.get("sharpe",        np.nan),
                "filt_return":   fm.get("total_return",  np.nan),
                "filt_mdd":      fm.get("max_drawdown",  np.nan),
                "filt_trades":   fm.get("trade_count",   0),
                "pf_delta":      (fm.get("profit_factor",1) - bm.get("profit_factor",1)),
                "sh_delta":      (fm.get("sharpe",0)        - bm.get("sharpe",0)),
            })

            logger.info(
                f"  → thresh={win_thresh:.2f} AUC={auc:.3f} | "
                f"PF {bm.get('profit_factor',0):.2f}→{fm.get('profit_factor',0):.2f} | "
                f"Sh {bm.get('sharpe',0):.2f}→{fm.get('sharpe',0):.2f} | "
                f"trades {bm.get('trade_count',0)}→{fm.get('trade_count',0)}"
            )

        if not wf_rows:
            return {"error": "No valid WF windows — try larger dataset or fewer windows"}

        wf_df = pd.DataFrame(wf_rows)

        return {
            "windows":          wf_df,
            "avg_auc":          wf_df["auc"].mean(),
            "avg_threshold":    wf_df["threshold"].mean(),
            "avg_pf_base":      wf_df["base_pf"].mean(),
            "avg_pf_filt":      wf_df["filt_pf"].mean(),
            "avg_sh_base":      wf_df["base_sharpe"].mean(),
            "avg_sh_filt":      wf_df["filt_sharpe"].mean(),
            "avg_pf_delta":     wf_df["pf_delta"].mean(),
            "avg_sh_delta":     wf_df["sh_delta"].mean(),
            "pf_improved_pct":  (wf_df["pf_delta"] > 0).mean(),
            "sh_improved_pct":  (wf_df["sh_delta"] > 0).mean(),
            "avg_base_mdd":     wf_df["base_mdd"].mean(),
            "avg_filt_mdd":     wf_df["filt_mdd"].mean(),
            "avg_base_trades":  wf_df["base_trades"].mean(),
            "avg_filt_trades":  wf_df["filt_trades"].mean(),
        }

    def _empty(self, name, tf, thresh, auto) -> MLFilterResult:
        from src.backtest.metrics import _empty_metrics
        return MLFilterResult(
            timeframe=tf, strategy_name=name,
            threshold=thresh, auto_threshold=auto,
            base_metrics=_empty_metrics(), base_trades=0,
            filt_metrics=_empty_metrics(), filt_trades=0,
            auc_score=0.5,
            feature_importances=pd.DataFrame(),
            threshold_search_df=pd.DataFrame(),
            trade_scores=pd.DataFrame(),
        )