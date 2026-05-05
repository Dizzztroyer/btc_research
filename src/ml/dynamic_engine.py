"""
dynamic_engine.py
─────────────────
Backtest engine with per-bar dynamic position sizing.

Design
──────
- Every signal gets a trade (no skipping)
- Position size = base_risk × size_multiplier
- size_multiplier comes from ML sizer (or 1.0 if disabled)
- All other logic (SL/TP/fees/slippage) identical to base engine

This is the ONLY change vs the base engine — nothing else is modified.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult, SimConfig
from src.backtest.metrics import compute_metrics, yearly_breakdown
from src.utils.logger import get_logger

logger = get_logger(__name__)

_EXIT = {0:"sl", 1:"tp", 2:"trail", 3:"time", 4:"signal", 5:"end_of_data"}
_DIR  = {"both": 0, "long": 1, "short": -1}


def run_dynamic(
    df:           pd.DataFrame,
    strategy_name: str,
    timeframe:    str,
    sim_cfg:      SimConfig,
    size_mults:   Optional[pd.Series] = None,
    params:       Optional[dict] = None,
) -> BacktestResult:
    """
    Run backtest with optional per-bar size multipliers.

    Parameters
    ----------
    size_mults : Series aligned with df.index
                 Values in [0.5, 1.5] — multiplied against risk_per_trade
                 If None or all 1.0, identical to base engine
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n  = len(df)

    if size_mults is None:
        mults = np.ones(n, dtype=np.float64)
    else:
        mults = size_mults.reset_index(drop=True).to_numpy(dtype=np.float64)
        if len(mults) != n:
            mults = np.ones(n, dtype=np.float64)

    # Numpy arrays for speed
    opens   = df["open"].to_numpy(np.float64)
    highs   = df["high"].to_numpy(np.float64)
    lows    = df["low"].to_numpy(np.float64)
    closes  = df["close"].to_numpy(np.float64)
    signals = df["signal"].fillna(0).to_numpy(np.int64)
    sl_arr  = df["sl_price"].to_numpy(np.float64) if "sl_price" in df.columns else np.full(n, np.nan)
    tp_arr  = df["tp_price"].to_numpy(np.float64) if "tp_price" in df.columns else np.full(n, np.nan)
    atr_col = next((c for c in ["atr_14","atr_7","atr_21"] if c in df.columns), None)
    atrs    = df[atr_col].to_numpy(np.float64) if atr_col else highs - lows

    dir_int = _DIR.get(sim_cfg.direction, 0)
    max_bh  = sim_cfg.max_bars_held or -1
    fees    = sim_cfg.fees
    slip    = sim_cfg.slippage
    base_rpt= sim_cfg.risk_per_trade
    lev     = sim_cfg.leverage
    init_cap= sim_cfg.initial_capital
    sl_pct  = sim_cfg.default_sl_pct
    tp_pct  = sim_cfg.default_tp_pct
    trail   = sim_cfg.trailing_stop
    trail_m = sim_cfg.trail_atr_mult

    equity      = np.full(n, init_cap, dtype=np.float64)
    capital     = float(init_cap)
    trade_rows: List[dict] = []

    in_pos      = False
    pos_side    = 0
    entry_price = 0.0
    entry_bar   = 0
    sl_price    = 0.0
    tp_price    = 0.0
    pos_size    = 0.0
    trail_price = 0.0
    entry_mult  = 1.0

    for i in range(1, n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        equity[i] = equity[i-1]

        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            atr = max(h - l, c * 0.005)

        # ── Manage open position ───────────────────────────────────────────────
        if in_pos:
            bh = i - entry_bar
            ep = -1.0
            er = -1

            if trail and atr > 0:
                dist = trail_m * atr
                trail_price = max(trail_price, o - dist) if pos_side == 1 \
                              else min(trail_price, o + dist)

            if pos_side == 1 and l <= sl_price:   ep, er = sl_price, 0
            elif pos_side == -1 and h >= sl_price: ep, er = sl_price, 0

            if ep < 0:
                if pos_side == 1 and h >= tp_price:   ep, er = tp_price, 1
                elif pos_side == -1 and l <= tp_price: ep, er = tp_price, 1

            if ep < 0 and trail:
                if pos_side == 1 and l <= trail_price:   ep, er = trail_price, 2
                elif pos_side == -1 and h >= trail_price: ep, er = trail_price, 2

            if ep < 0 and max_bh > 0 and bh >= max_bh:
                ep, er = o, 3

            if ep < 0:
                sig = int(signals[i-1])
                if sig == 0 or (pos_side==1 and sig==-1) or (pos_side==-1 and sig==1):
                    ep, er = o, 4

            if ep >= 0:
                fill    = ep * (1.0 + (-pos_side) * slip)
                fee     = pos_size * fees
                raw     = pos_side * (fill - entry_price) / entry_price * pos_size
                net     = raw - fee
                capital += net
                equity[i] = capital

                ra    = abs(entry_price - sl_price) / entry_price * pos_size
                rmult = net / ra if ra > 0 else 0.0

                trade_rows.append({
                    "entry_time":  df["timestamp"].iloc[entry_bar],
                    "exit_time":   df["timestamp"].iloc[i],
                    "side":        pos_side,
                    "entry_price": entry_price,
                    "exit_price":  fill,
                    "size":        pos_size,
                    "pnl":         net,
                    "pnl_pct":     net / pos_size if pos_size > 0 else 0.0,
                    "r_multiple":  rmult,
                    "bars_held":   bh,
                    "exit_reason": _EXIT.get(er, "?"),
                    "size_mult":   entry_mult,
                })
                in_pos = False; pos_side = 0

        # ── Open position ──────────────────────────────────────────────────────
        if not in_pos:
            sig = int(signals[i-1])
            if sig == 0: continue
            if dir_int == 1  and sig != 1:  continue
            if dir_int == -1 and sig != -1: continue

            mult = float(mults[i-1])
            # Safety clamp — never zero, never over 2x
            mult = max(0.5, min(2.0, mult))

            side  = sig
            raw_sl = float(sl_arr[i-1])
            sl = raw_sl if (not np.isnan(raw_sl) and raw_sl > 0) else o*(1.0 - side*sl_pct)
            raw_tp = float(tp_arr[i-1])
            tp = raw_tp if (not np.isnan(raw_tp) and raw_tp > 0) else o*(1.0 + side*tp_pct)

            if side == 1  and sl >= o: continue
            if side == -1 and sl <= o: continue

            fill     = o * (1.0 + side * slip)
            risk_pct = abs(fill - sl) / fill
            if risk_pct <= 0: continue

            eff_risk = base_rpt * mult
            size     = min((capital * eff_risk) / risk_pct, capital * lev)

            capital   -= size * fees
            equity[i]  = capital

            in_pos      = True
            pos_side    = side
            entry_price = fill
            entry_bar   = i
            sl_price    = sl
            tp_price    = tp
            pos_size    = size
            entry_mult  = mult
            trail_price = sl if trail else 0.0

    # Close remaining at end
    if in_pos:
        fill    = closes[n-1] * (1.0 - pos_side * slip)
        net     = pos_side * (fill - entry_price) / entry_price * pos_size - pos_size * fees
        capital += net
        equity[n-1] = capital
        ra    = abs(entry_price - sl_price) / entry_price * pos_size
        trade_rows.append({
            "entry_time": df["timestamp"].iloc[entry_bar],
            "exit_time":  df["timestamp"].iloc[-1],
            "side": pos_side, "entry_price": entry_price, "exit_price": fill,
            "size": pos_size, "pnl": net,
            "pnl_pct": net/pos_size if pos_size>0 else 0,
            "r_multiple": net/ra if ra>0 else 0,
            "bars_held": n-1-entry_bar, "exit_reason": "end_of_data",
            "size_mult": entry_mult,
        })

    eq   = pd.Series(equity, index=df.index)
    dd   = (eq - eq.cummax()) / eq.cummax().replace(0, np.nan)
    tdf  = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()
    ts   = df["timestamp"]
    tot  = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    bpy  = 365.25*86400 / (tot/max(len(ts)-1,1)) if tot>0 else 365.0

    metrics = compute_metrics(tdf, eq, bpy, init_cap)
    yearly  = yearly_breakdown(tdf, eq, ts, bpy, init_cap)

    return BacktestResult(
        metrics=metrics, trades=tdf, equity=eq, drawdown=dd,
        yearly=yearly, timeframe=timeframe,
        strategy_name=strategy_name, params=params or {},
    )