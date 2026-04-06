"""
engine.py
─────────
Bar-based backtesting engine.

Design principles
─────────────────
- Operates bar-by-bar in a strict time-ordered loop
- No lookahead: signals generated on bar N are acted on bar N+1 open
- Supports long, short, and long-short strategies
- Position sizing: fixed-fractional (risk % of equity per trade)
- Stop-loss, take-profit, trailing stop, time-based exit
- Realistic fees and slippage applied on every fill
- No impossible fills: cannot open and close on the same bar signal
- Returns: trade log, equity curve, drawdown curve

Usage
─────
    from src.backtest.engine import BacktestEngine, SimConfig

    sim_cfg = SimConfig(fees=0.00075, slippage=0.0003, ...)
    engine  = BacktestEngine(sim_cfg)
    result  = engine.run(df_features, signals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_metrics, yearly_breakdown
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """Simulation parameters."""
    fees:          float = 0.00075   # fraction per side
    slippage:      float = 0.0003    # fraction per side
    leverage:      float = 1.0
    risk_per_trade: float = 0.01     # fraction of equity risked per trade
    initial_capital: float = 10_000.0
    direction: str = "long"          # "long" | "short" | "both"
    # Stop / target defaults (strategy may override via signal columns)
    default_sl_pct: float = 0.02     # stop-loss as fraction of entry price
    default_tp_pct: float = 0.04     # take-profit as fraction of entry price
    trailing_stop: bool = False
    trail_atr_mult: float = 2.0      # trailing stop = trail_atr_mult * ATR
    max_bars_held:  Optional[int] = None  # time-based exit
    partial_tp_pct: float = 0.0      # fraction of position to exit at TP1 (0 = disabled)
    partial_tp_level: float = 0.0    # TP1 level as fraction of price


@dataclass
class Trade:
    """Represents a single closed trade."""
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    side:        int            # 1 = long, -1 = short
    entry_price: float
    exit_price:  float
    size:        float          # position size in quote currency
    pnl:         float
    pnl_pct:     float
    r_multiple:  float
    exit_reason: str            # "sl", "tp", "trail", "time", "signal"
    bars_held:   int

    def to_dict(self) -> dict:
        return {
            "entry_time":  self.entry_time,
            "exit_time":   self.exit_time,
            "side":        self.side,
            "entry_price": self.entry_price,
            "exit_price":  self.exit_price,
            "size":        self.size,
            "pnl":         self.pnl,
            "pnl_pct":     self.pnl_pct,
            "r_multiple":  self.r_multiple,
            "exit_reason": self.exit_reason,
            "bars_held":   self.bars_held,
        }


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    metrics:       dict
    trades:        pd.DataFrame
    equity:        pd.Series
    drawdown:      pd.Series
    yearly:        pd.DataFrame
    timeframe:     str = ""
    strategy_name: str = ""
    params:        dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.metrics.get("trade_count", 0) > 0


# ── Simulation helpers ─────────────────────────────────────────────────────────

def _fill_price(price: float, side: int, slippage: float) -> float:
    """Apply slippage to a fill price."""
    # Long buys at a slightly higher price; short sells at a slightly lower price
    return price * (1 + side * slippage)


def _bars_per_year(timestamps: pd.Series) -> float:
    """Estimate bars per year from timestamp spacing."""
    if len(timestamps) < 2:
        return 365.0
    total_seconds = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    n_bars        = len(timestamps) - 1
    if total_seconds <= 0:
        return 365.0
    seconds_per_bar = total_seconds / n_bars
    return 365.25 * 86_400 / seconds_per_bar


# ── Main engine ────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Bar-based backtest engine.

    Signals DataFrame must contain at minimum:
        timestamp : UTC bar open time
        open      : bar open price
        high      : bar high
        low       : bar low
        close     : bar close
        signal    : integer — 1 = go long, -1 = go short, 0 = flat / exit

    Optional signal columns (override SimConfig defaults):
        sl_price  : explicit stop-loss price (per bar)
        tp_price  : explicit take-profit price (per bar)
        atr       : current ATR (used for trailing stop)
    """

    def __init__(self, sim_cfg: SimConfig) -> None:
        self.cfg = sim_cfg

    def run(
        self,
        df: pd.DataFrame,
        strategy_name: str = "unnamed",
        timeframe: str = "",
        params: Optional[dict] = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Parameters
        ----------
        df            : feature+signal DataFrame with required columns
        strategy_name : label for reporting
        timeframe     : label for reporting
        params        : parameter dict (stored in result for reference)

        Returns
        -------
        BacktestResult
        """
        required = {"timestamp", "open", "high", "low", "close", "signal"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        df = df.sort_values("timestamp").reset_index(drop=True)
        n  = len(df)

        equity   = np.full(n, self.cfg.initial_capital, dtype=float)
        capital  = self.cfg.initial_capital
        trades: List[Trade] = []

        # ── Open position state ────────────────────────────────────────────────
        in_position     = False
        position_side   = 0       # 1 or -1
        entry_price     = 0.0
        entry_bar       = 0
        sl_price        = 0.0
        tp_price        = 0.0
        position_size   = 0.0    # in quote currency (dollars)
        trail_price     = 0.0    # trailing stop reference
        partial_done    = False

        bpy = _bars_per_year(df["timestamp"])

        for i in range(1, n):
            bar   = df.iloc[i]
            prev  = df.iloc[i - 1]
            o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
            ts    = bar["timestamp"]

            # Carry equity forward
            equity[i] = equity[i - 1]

            # ── ATR for trailing stop ──────────────────────────────────────────
            atr_val = bar.get("atr_14", bar.get("atr_7", None))
            if atr_val is None or pd.isna(atr_val):
                atr_val = (h - l)  # fallback

            # ── Manage open position ───────────────────────────────────────────
            if in_position:
                bars_held = i - entry_bar
                exit_price  = None
                exit_reason = None

                # Update trailing stop
                if self.cfg.trailing_stop and atr_val > 0:
                    trail_dist = self.cfg.trail_atr_mult * atr_val
                    if position_side == 1:
                        trail_price = max(trail_price, o - trail_dist)
                    else:
                        trail_price = min(trail_price, o + trail_dist)

                # ── Check stop-loss ────────────────────────────────────────────
                if position_side == 1 and l <= sl_price:
                    exit_price  = sl_price
                    exit_reason = "sl"
                elif position_side == -1 and h >= sl_price:
                    exit_price  = sl_price
                    exit_reason = "sl"

                # ── Check take-profit ──────────────────────────────────────────
                if exit_price is None:
                    if position_side == 1 and h >= tp_price:
                        exit_price  = tp_price
                        exit_reason = "tp"
                    elif position_side == -1 and l <= tp_price:
                        exit_price  = tp_price
                        exit_reason = "tp"

                # ── Check trailing stop ────────────────────────────────────────
                if exit_price is None and self.cfg.trailing_stop:
                    if position_side == 1 and l <= trail_price:
                        exit_price  = trail_price
                        exit_reason = "trail"
                    elif position_side == -1 and h >= trail_price:
                        exit_price  = trail_price
                        exit_reason = "trail"

                # ── Time-based exit ────────────────────────────────────────────
                if exit_price is None and self.cfg.max_bars_held:
                    if bars_held >= self.cfg.max_bars_held:
                        exit_price  = o
                        exit_reason = "time"

                # ── Signal-based exit ──────────────────────────────────────────
                if exit_price is None:
                    sig = prev["signal"]  # signal generated on prev bar
                    # Exit on opposite or zero signal
                    if sig == 0 or (position_side == 1 and sig == -1) or \
                                   (position_side == -1 and sig == 1):
                        exit_price  = o
                        exit_reason = "signal"

                # ── Close the position ─────────────────────────────────────────
                if exit_price is not None:
                    fill = _fill_price(exit_price, -position_side, self.cfg.slippage)
                    fee  = position_size * self.cfg.fees

                    raw_pnl = position_side * (fill - entry_price) / entry_price * position_size
                    net_pnl = raw_pnl - fee

                    capital += net_pnl
                    equity[i] = capital

                    risk_amount = abs(entry_price - sl_price) / entry_price * position_size
                    r_mult      = net_pnl / risk_amount if risk_amount > 0 else 0.0

                    trades.append(Trade(
                        entry_time  = df.iloc[entry_bar]["timestamp"],
                        exit_time   = ts,
                        side        = position_side,
                        entry_price = entry_price,
                        exit_price  = fill,
                        size        = position_size,
                        pnl         = net_pnl,
                        pnl_pct     = net_pnl / position_size,
                        r_multiple  = r_mult,
                        exit_reason = exit_reason,
                        bars_held   = bars_held,
                    ))

                    in_position = False
                    position_side = 0

            # ── Open a new position ────────────────────────────────────────────
            if not in_position:
                sig = prev["signal"]  # signal on prev bar → enter on this bar open

                if sig == 0:
                    continue

                # Direction filter
                if self.cfg.direction == "long"  and sig != 1:
                    continue
                if self.cfg.direction == "short" and sig != -1:
                    continue

                side = sig  # 1 or -1

                # Determine SL price
                if "sl_price" in df.columns and not pd.isna(prev.get("sl_price")):
                    sl = float(prev["sl_price"])
                else:
                    sl = o * (1 - side * self.cfg.default_sl_pct)

                # Determine TP price
                if "tp_price" in df.columns and not pd.isna(prev.get("tp_price")):
                    tp = float(prev["tp_price"])
                else:
                    tp = o * (1 + side * self.cfg.default_tp_pct)

                # Skip if SL makes no sense
                if side == 1  and sl >= o: continue
                if side == -1 and sl <= o: continue

                fill = _fill_price(o, side, self.cfg.slippage)
                fee  = capital * self.cfg.risk_per_trade * self.cfg.fees

                # Position size: risk a fixed fraction of equity
                risk_pct  = abs(fill - sl) / fill
                if risk_pct <= 0:
                    continue
                pos_size  = (capital * self.cfg.risk_per_trade) / risk_pct
                pos_size  = min(pos_size, capital * self.cfg.leverage)

                # Deduct entry fee from equity
                entry_cost = pos_size * self.cfg.fees
                capital   -= entry_cost
                equity[i]  = capital

                in_position   = True
                position_side = side
                entry_price   = fill
                entry_bar     = i
                sl_price      = sl
                tp_price      = tp
                position_size = pos_size
                partial_done  = False

                # Init trailing stop
                trail_price = sl if self.cfg.trailing_stop else 0.0

        # Close any open trade at end
        if in_position and len(df) > 0:
            last     = df.iloc[-1]
            fill     = _fill_price(last["close"], -position_side, self.cfg.slippage)
            fee      = position_size * self.cfg.fees
            raw_pnl  = position_side * (fill - entry_price) / entry_price * position_size
            net_pnl  = raw_pnl - fee
            capital += net_pnl
            equity[-1] = capital
            risk_amount = abs(entry_price - sl_price) / entry_price * position_size
            r_mult = net_pnl / risk_amount if risk_amount > 0 else 0.0
            trades.append(Trade(
                entry_time  = df.iloc[entry_bar]["timestamp"],
                exit_time   = last["timestamp"],
                side        = position_side,
                entry_price = entry_price,
                exit_price  = fill,
                size        = position_size,
                pnl         = net_pnl,
                pnl_pct     = net_pnl / position_size,
                r_multiple  = r_mult,
                exit_reason = "end_of_data",
                bars_held   = n - 1 - entry_bar,
            ))

        # ── Assemble results ───────────────────────────────────────────────────
        eq_series = pd.Series(equity, index=df.index)
        dd_series = (eq_series - eq_series.cummax()) / eq_series.cummax()

        trade_df = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()

        metrics = compute_metrics(
            trades          = trade_df,
            equity          = eq_series,
            bars_per_year   = bpy,
            initial_capital = self.cfg.initial_capital,
        )

        yearly = yearly_breakdown(
            trades         = trade_df,
            equity         = eq_series,
            timestamps     = df["timestamp"],
            bars_per_year  = bpy,
            initial_capital= self.cfg.initial_capital,
        )

        return BacktestResult(
            metrics       = metrics,
            trades        = trade_df,
            equity        = eq_series,
            drawdown      = dd_series,
            yearly        = yearly,
            timeframe     = timeframe,
            strategy_name = strategy_name,
            params        = params or {},
        )
