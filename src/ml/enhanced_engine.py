"""
enhanced_engine.py
──────────────────
BacktestEngine extension that supports dynamic position sizing.

Integrates:
1. MLPositionSizer  → score-based size multiplier per signal bar
2. RegimeFilter     → regime-based size multiplier per bar

Combined multiplier = ml_mult × regime_mult × base_risk_per_trade

The engine itself is unchanged — we just modify risk_per_trade
on a per-signal basis by injecting a 'size_mult' column into the DataFrame
and reading it in the simulation loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult, SimConfig
from src.backtest.metrics import compute_metrics, yearly_breakdown
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_enhanced_backtest(
    df:              pd.DataFrame,       # feature + signal DataFrame
    strategy_name:   str,
    timeframe:       str,
    sim_cfg:         SimConfig,
    size_multipliers: Optional[pd.Series] = None,  # per-bar multipliers
    params:          Optional[dict] = None,
) -> BacktestResult:
    """
    Run backtest with per-bar position size multipliers.

    The multiplier modifies risk_per_trade for each individual trade.
    All other simulation logic (SL/TP/trailing/fees) is unchanged.

    Parameters
    ----------
    df               : DataFrame with signal, sl_price, tp_price columns
    size_multipliers : Series indexed like df, values in [0.0, 2.0]
                       If None, falls back to standard backtest
    """
    if size_multipliers is None:
        engine = BacktestEngine(sim_cfg)
        return engine.run(df, strategy_name, timeframe, params)

    # Validate alignment
    if not size_multipliers.index.equals(df.index):
        size_multipliers = size_multipliers.reindex(df.index, fill_value=1.0)

    # Clip multipliers to safe range
    size_multipliers = size_multipliers.clip(0.0, 3.0)

    # Run bar-by-bar with per-trade risk adjustment
    return _run_with_dynamic_sizing(
        df=df,
        strategy_name=strategy_name,
        timeframe=timeframe,
        sim_cfg=sim_cfg,
        size_mults=size_multipliers,
        params=params or {},
    )


def _run_with_dynamic_sizing(
    df:            pd.DataFrame,
    strategy_name: str,
    timeframe:     str,
    sim_cfg:       SimConfig,
    size_mults:    pd.Series,
    params:        dict,
) -> BacktestResult:
    """
    Full bar-by-bar simulation with dynamic position sizing.
    Same logic as BacktestEngine.run() but risk_per_trade is scaled
    by size_mults at each entry.
    """
    from src.backtest.fast_engine import NUMBA_AVAILABLE

    df = df.sort_values("timestamp").reset_index(drop=True)
    size_mults = size_mults.reset_index(drop=True)
    n  = len(df)

    # Convert direction
    dir_map = {"both": 0, "long": 1, "short": -1}
    direction_int = dir_map.get(sim_cfg.direction, 0)

    opens   = df["open"].to_numpy(dtype=np.float64)
    highs   = df["high"].to_numpy(dtype=np.float64)
    lows    = df["low"].to_numpy(dtype=np.float64)
    closes  = df["close"].to_numpy(dtype=np.float64)
    signals = df["signal"].fillna(0).to_numpy(dtype=np.int64)
    sl_arr  = df["sl_price"].to_numpy(dtype=np.float64) if "sl_price" in df.columns else np.full(n, np.nan)
    tp_arr  = df["tp_price"].to_numpy(dtype=np.float64) if "tp_price" in df.columns else np.full(n, np.nan)
    atr_col = next((c for c in ["atr_14","atr_7","atr_21"] if c in df.columns), None)
    atrs    = df[atr_col].to_numpy(dtype=np.float64) if atr_col else (highs - lows)
    mults   = size_mults.to_numpy(dtype=np.float64)

    EXIT_REASONS = {0:"sl", 1:"tp", 2:"trail", 3:"time", 4:"signal", 5:"end_of_data"}

    # ── Simulation loop ────────────────────────────────────────────────────────
    equity      = np.full(n, sim_cfg.initial_capital, dtype=np.float64)
    capital     = float(sim_cfg.initial_capital)
    trade_rows: List[dict] = []

    in_pos      = False
    pos_side    = 0
    entry_price = 0.0
    entry_bar   = 0
    sl_price    = 0.0
    tp_price    = 0.0
    pos_size    = 0.0
    trail_price = 0.0
    max_bh      = sim_cfg.max_bars_held or -1

    for i in range(1, n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        equity[i]  = equity[i-1]

        atr_val = atrs[i]
        if np.isnan(atr_val) or atr_val <= 0:
            atr_val = max(h - l, c * 0.01)

        # ── Manage open position ───────────────────────────────────────────────
        if in_pos:
            bars_held   = i - entry_bar
            exit_price  = -1.0
            exit_reason = -1

            if sim_cfg.trailing_stop and atr_val > 0:
                dist = sim_cfg.trail_atr_mult * atr_val
                if pos_side == 1:
                    trail_price = max(trail_price, o - dist)
                else:
                    trail_price = min(trail_price, o + dist)

            if pos_side == 1 and l <= sl_price:
                exit_price, exit_reason = sl_price, 0
            elif pos_side == -1 and h >= sl_price:
                exit_price, exit_reason = sl_price, 0

            if exit_price < 0:
                if pos_side == 1 and h >= tp_price:
                    exit_price, exit_reason = tp_price, 1
                elif pos_side == -1 and l <= tp_price:
                    exit_price, exit_reason = tp_price, 1

            if exit_price < 0 and sim_cfg.trailing_stop:
                if pos_side == 1 and l <= trail_price:
                    exit_price, exit_reason = trail_price, 2
                elif pos_side == -1 and h >= trail_price:
                    exit_price, exit_reason = trail_price, 2

            if exit_price < 0 and max_bh > 0 and bars_held >= max_bh:
                exit_price, exit_reason = o, 3

            if exit_price < 0:
                sig = int(signals[i-1])
                if sig == 0 or (pos_side == 1 and sig == -1) or (pos_side == -1 and sig == 1):
                    exit_price, exit_reason = o, 4

            if exit_price >= 0:
                fill    = exit_price * (1.0 + (-pos_side) * sim_cfg.slippage)
                fee     = pos_size * sim_cfg.fees
                raw_pnl = pos_side * (fill - entry_price) / entry_price * pos_size
                net_pnl = raw_pnl - fee
                capital += net_pnl
                equity[i] = capital
                risk_amt  = abs(entry_price - sl_price) / entry_price * pos_size if sl_price > 0 else pos_size * 0.02
                r_mult    = net_pnl / risk_amt if risk_amt > 0 else 0.0
                trade_rows.append({
                    "entry_time":  df["timestamp"].iloc[entry_bar],
                    "exit_time":   df["timestamp"].iloc[i],
                    "side":        pos_side,
                    "entry_price": entry_price,
                    "exit_price":  fill,
                    "size":        pos_size,
                    "pnl":         net_pnl,
                    "pnl_pct":     net_pnl / pos_size if pos_size > 0 else 0,
                    "r_multiple":  r_mult,
                    "bars_held":   bars_held,
                    "exit_reason": EXIT_REASONS.get(exit_reason, "unknown"),
                    "size_mult":   mults[entry_bar],
                })
                in_pos   = False
                pos_side = 0

        # ── Open position ──────────────────────────────────────────────────────
        if not in_pos:
            sig = int(signals[i-1])
            if sig == 0: continue
            if direction_int == 1  and sig != 1:  continue
            if direction_int == -1 and sig != -1: continue

            # Skip if multiplier is zero (regime filter disabled trading)
            mult = float(mults[i-1])
            if mult <= 0.0:
                continue

            side   = sig
            raw_sl = float(sl_arr[i-1])
            sl     = raw_sl if not np.isnan(raw_sl) and raw_sl > 0 else o * (1.0 - side * sim_cfg.default_sl_pct)
            raw_tp = float(tp_arr[i-1])
            tp     = raw_tp if not np.isnan(raw_tp) and raw_tp > 0 else o * (1.0 + side * sim_cfg.default_tp_pct)

            if side == 1 and sl >= o: continue
            if side == -1 and sl <= o: continue

            fill     = o * (1.0 + side * sim_cfg.slippage)
            risk_pct = abs(fill - sl) / fill
            if risk_pct <= 0: continue

            # Apply size multiplier to risk_per_trade
            effective_risk = sim_cfg.risk_per_trade * mult
            size = min((capital * effective_risk) / risk_pct, capital * sim_cfg.leverage)

            capital   -= size * sim_cfg.fees
            equity[i]  = capital

            in_pos      = True
            pos_side    = side
            entry_price = fill
            entry_bar   = i
            sl_price    = sl
            tp_price    = tp
            pos_size    = size
            trail_price = sl if sim_cfg.trailing_stop else 0.0

    # Close open trade at end
    if in_pos:
        fill    = closes[n-1] * (1.0 - pos_side * sim_cfg.slippage)
        fee     = pos_size * sim_cfg.fees
        net_pnl = pos_side * (fill - entry_price) / entry_price * pos_size - fee
        capital += net_pnl
        equity[n-1] = capital
        risk_amt    = abs(entry_price - sl_price) / entry_price * pos_size if sl_price > 0 else pos_size * 0.02
        trade_rows.append({
            "entry_time": df["timestamp"].iloc[entry_bar],
            "exit_time":  df["timestamp"].iloc[-1],
            "side": pos_side, "entry_price": entry_price, "exit_price": fill,
            "size": pos_size, "pnl": net_pnl,
            "pnl_pct": net_pnl/pos_size if pos_size>0 else 0,
            "r_multiple": net_pnl/risk_amt if risk_amt>0 else 0,
            "bars_held": n-1-entry_bar, "exit_reason": "end_of_data",
            "size_mult": mults[entry_bar],
        })

    eq_series  = pd.Series(equity, index=df.index)
    dd_series  = (eq_series - eq_series.cummax()) / eq_series.cummax().replace(0, np.nan)
    trades_df  = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()

    # bpy
    ts    = df["timestamp"]
    total = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    bpy   = 365.25 * 86400 / (total / max(len(ts)-1, 1)) if total > 0 else 365.0

    metrics = compute_metrics(trades_df, eq_series, bpy, sim_cfg.initial_capital)
    yearly  = yearly_breakdown(trades_df, eq_series, ts, bpy, sim_cfg.initial_capital)

    return BacktestResult(
        metrics=metrics, trades=trades_df,
        equity=eq_series, drawdown=dd_series,
        yearly=yearly, timeframe=timeframe,
        strategy_name=strategy_name, params=params,
    )