"""
ensemble.py
───────────
Ensemble / portfolio strategy that combines signals from multiple sub-strategies.

Approach:
    Each sub-strategy votes (+1 long / -1 short / 0 flat).
    A trade is entered only when votes exceed a consensus threshold.
    Equal-weight voting; no ML required.

Three ensemble variants:
1. MajorityVote      — enter when majority of models agree
2. UnanimousVote     — enter only when all models agree
3. WeightedEnsemble  — weight by sub-strategy type (trend gets more weight in trending regime)
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.trend import EMACrossStrategy, DonchianBreakoutStrategy
from src.strategies.mean_reversion import RSIReversionStrategy, BollingerReversionStrategy
from src.strategies.breakout import SqueezeBreakoutStrategy


# ── Pre-defined default parameter sets for each sub-strategy ─────────────────

_DEFAULT_TREND_PARAMS = {
    "fast": 13, "slow": 55, "atr_len": 14, "sl_mult": 2.0, "tp_mult": 3.0
}
_DEFAULT_DON_PARAMS = {
    "length": 20, "atr_len": 14, "sl_mult": 2.0, "tp_mult": 3.0
}
_DEFAULT_RSI_PARAMS = {
    "rsi_len": 14, "rsi_os": 30, "rsi_ob": 70,
    "atr_len": 14, "sl_mult": 1.5, "tp_mult": 2.0, "adx_thresh": 0
}
_DEFAULT_BB_PARAMS = {
    "bb_len": 20, "bb_std": 2.0, "atr_len": 14, "sl_mult": 1.0
}
_DEFAULT_SQZ_PARAMS = {
    "length": 20, "bb_std": 2.0, "kc_mult": 1.5,
    "atr_len": 14, "tp_mult": 3.0, "sl_mult": 1.5
}


def _compute_vote_signal(votes: pd.DataFrame, threshold: float) -> pd.Series:
    """
    Combine vote columns into a consensus signal.

    Parameters
    ----------
    votes     : DataFrame with one column per strategy (values -1, 0, 1)
    threshold : fraction of strategies that must agree (e.g. 0.6 = 60%)

    Returns
    -------
    Series with values -1, 0, 1
    """
    n = votes.shape[1]
    min_agree = max(1, int(np.ceil(n * threshold)))

    long_votes  = (votes == 1).sum(axis=1)
    short_votes = (votes == -1).sum(axis=1)

    signal = np.where(
        long_votes  >= min_agree, 1,
        np.where(short_votes >= min_agree, -1, 0)
    )
    return pd.Series(signal, index=votes.index)


class MajorityVoteEnsemble(BaseStrategy):
    """
    Enter when ≥ vote_threshold fraction of sub-strategies agree on direction.
    SL/TP: average of agreeing sub-strategies' SL/TP levels.
    """

    name = "majority_vote_ensemble"

    def __init__(self) -> None:
        self._strategies = {
            "ema_cross":  EMACrossStrategy(),
            "donchian":   DonchianBreakoutStrategy(),
            "rsi_rev":    RSIReversionStrategy(),
            "bb_rev":     BollingerReversionStrategy(),
            "squeeze":    SqueezeBreakoutStrategy(),
        }
        self._default_params = {
            "ema_cross": _DEFAULT_TREND_PARAMS,
            "donchian":  _DEFAULT_DON_PARAMS,
            "rsi_rev":   _DEFAULT_RSI_PARAMS,
            "bb_rev":    _DEFAULT_BB_PARAMS,
            "squeeze":   _DEFAULT_SQZ_PARAMS,
        }

    def param_grid(self) -> List[Dict[str, Any]]:
        thresholds = [0.4, 0.5, 0.6, 0.8, 1.0]
        atr_lens   = [7, 14]
        sl_mults   = [1.5, 2.0]
        tp_mults   = [2.0, 3.0]
        return [
            {"vote_threshold": vt, "atr_len": a, "sl_mult": sl, "tp_mult": tp}
            for vt, a, sl, tp in itertools.product(thresholds, atr_lens, sl_mults, tp_mults)
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        atr_col = f"atr_{params['atr_len']}"
        atr     = df[atr_col] if atr_col in df.columns else (df["high"] - df["low"])

        vote_cols: Dict[str, pd.Series] = {}
        sl_cols:   Dict[str, pd.Series] = {}
        tp_cols:   Dict[str, pd.Series] = {}

        for name, strat in self._strategies.items():
            try:
                sub_df = strat.generate_signals(df.copy(), self._default_params[name])
                vote_cols[name] = sub_df["signal"]
                sl_cols[name]   = sub_df["sl_price"]
                tp_cols[name]   = sub_df["tp_price"]
            except Exception:
                vote_cols[name] = pd.Series(0, index=df.index)
                sl_cols[name]   = pd.Series(np.nan, index=df.index)
                tp_cols[name]   = pd.Series(np.nan, index=df.index)

        votes_df = pd.DataFrame(vote_cols)
        signal   = _compute_vote_signal(votes_df, params["vote_threshold"])

        df["signal"] = signal.values

        # Average SL/TP of agreeing strategies
        sl_arr = np.full(len(df), np.nan)
        tp_arr = np.full(len(df), np.nan)
        sl_df  = pd.DataFrame(sl_cols)
        tp_df  = pd.DataFrame(tp_cols)

        for i in range(len(df)):
            sig = df["signal"].iloc[i]
            if sig == 0:
                continue
            # Find which strategies agree with this signal
            agreeing = [k for k in vote_cols if vote_cols[k].iloc[i] == sig]
            if agreeing:
                sl_vals = [sl_df[k].iloc[i] for k in agreeing if not pd.isna(sl_df[k].iloc[i])]
                tp_vals = [tp_df[k].iloc[i] for k in agreeing if not pd.isna(tp_df[k].iloc[i])]
                if sl_vals:
                    sl_arr[i] = np.mean(sl_vals)
                if tp_vals:
                    tp_arr[i] = np.mean(tp_vals)

            # Fallback if no SL/TP from sub-strategies
            if np.isnan(sl_arr[i]):
                sl_arr[i] = df["close"].iloc[i] - sig * params["sl_mult"] * atr.iloc[i]
            if np.isnan(tp_arr[i]):
                tp_arr[i] = df["close"].iloc[i] + sig * params["tp_mult"] * atr.iloc[i]

        df["sl_price"] = sl_arr
        df["tp_price"] = tp_arr

        return df


class WeightedEnsemble(BaseStrategy):
    """
    Weighted voting ensemble.

    In trending regimes  : trend strategies get weight 2, MR strategies get weight 1
    In ranging regimes   : MR strategies get weight 2, trend strategies get weight 1

    Regime determined by ADX (prior bar, no lookahead).
    """

    name = "weighted_ensemble"

    def __init__(self) -> None:
        self._trend_strats = {
            "ema_cross": (EMACrossStrategy(), _DEFAULT_TREND_PARAMS),
            "donchian":  (DonchianBreakoutStrategy(), _DEFAULT_DON_PARAMS),
            "squeeze":   (SqueezeBreakoutStrategy(), _DEFAULT_SQZ_PARAMS),
        }
        self._mr_strats = {
            "rsi_rev": (RSIReversionStrategy(), _DEFAULT_RSI_PARAMS),
            "bb_rev":  (BollingerReversionStrategy(), _DEFAULT_BB_PARAMS),
        }

    def param_grid(self) -> List[Dict[str, Any]]:
        adx_thresholds = [20, 25, 30]
        adx_lengths    = [10, 14]
        vote_thresholds= [0.4, 0.6]
        atr_lens       = [7, 14]
        sl_mults       = [1.5, 2.0]
        tp_mults       = [2.0, 3.0]
        return [
            {"adx_thresh": at, "adx_len": al, "vote_thresh": vt,
             "atr_len": a, "sl_mult": sl, "tp_mult": tp}
            for at, al, vt, a, sl, tp in itertools.product(
                adx_thresholds, adx_lengths, vote_thresholds, atr_lens, sl_mults, tp_mults
            )
        ]

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        self._add_signal_columns(df)

        adx_col = f"adx_{params['adx_len']}"
        atr_col = f"atr_{params['atr_len']}"
        atr     = df[atr_col] if atr_col in df.columns else (df["high"] - df["low"])

        adx = df[adx_col] if adx_col in df.columns else pd.Series(20, index=df.index)
        is_trending = adx.shift(1) > params["adx_thresh"]

        # Generate all sub-signals
        all_signals: Dict[str, pd.Series] = {}
        all_sl:      Dict[str, pd.Series] = {}
        all_tp:      Dict[str, pd.Series] = {}

        for name, (strat, p) in {**self._trend_strats, **self._mr_strats}.items():
            try:
                sub = strat.generate_signals(df.copy(), p)
                all_signals[name] = sub["signal"]
                all_sl[name]      = sub["sl_price"]
                all_tp[name]      = sub["tp_price"]
            except Exception:
                all_signals[name] = pd.Series(0, index=df.index)
                all_sl[name]      = pd.Series(np.nan, index=df.index)
                all_tp[name]      = pd.Series(np.nan, index=df.index)

        trend_names = list(self._trend_strats.keys())
        mr_names    = list(self._mr_strats.keys())

        weighted_signal = np.zeros(len(df))
        total_weight    = np.zeros(len(df))

        for name in trend_names:
            w = np.where(is_trending, 2.0, 1.0)
            weighted_signal += w * all_signals[name].values
            total_weight    += w

        for name in mr_names:
            w = np.where(is_trending, 1.0, 2.0)
            weighted_signal += w * all_signals[name].values
            total_weight    += w

        norm_signal = weighted_signal / np.where(total_weight > 0, total_weight, 1)

        threshold   = params["vote_thresh"]
        final_sig   = np.where(norm_signal >=  threshold,  1,
                      np.where(norm_signal <= -threshold, -1, 0))

        df["signal"] = final_sig

        # SL/TP: simple ATR-based fallback
        sl_dist = params["sl_mult"] * atr
        tp_dist = params["tp_mult"] * atr

        df["sl_price"] = np.where(
            df["signal"] == 1,  df["close"] - sl_dist,
            np.where(df["signal"] == -1, df["close"] + sl_dist, np.nan)
        )
        df["tp_price"] = np.where(
            df["signal"] == 1,  df["close"] + tp_dist,
            np.where(df["signal"] == -1, df["close"] - tp_dist, np.nan)
        )

        return df
