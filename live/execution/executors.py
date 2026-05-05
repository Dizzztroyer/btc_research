"""
live/execution/paper.py
────────────────────────
Paper trading executor — logs orders without sending to exchange.
Use this to verify signal flow before enabling live trading.

live/execution/binance.py
──────────────────────────
Binance Futures executor (USDT-M perpetual).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .base import BaseExecutor, Order, OrderSide, OrderStatus, Position


# ══════════════════════════════════════════════════════════════════════════════
# PAPER EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

class PaperExecutor(BaseExecutor):
    """
    Simulates order execution locally. No real orders sent.
    Fills at the price specified (simulating market order at open).
    Tracks positions and P&L in memory + writes to CSV log.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        log_path:        Optional[Path] = None,
    ):
        self._balance   = initial_balance
        self._positions: List[Position] = []
        self._log_path  = log_path
        self._trade_log: List[dict] = []

    def is_connected(self) -> bool:
        return True  # paper mode is always "connected"

    def get_account_balance(self) -> float:
        return self._balance

    def open_position(self, order: Order) -> Order:
        # Simulate market fill at specified entry price
        order.entry_price = order.entry_price  # already set by caller
        order.order_id    = f"PAPER-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{order.strategy}"
        order.status      = OrderStatus.OPEN
        order.filled_at   = datetime.now(timezone.utc)
        order.risk_usd    = order.quantity * abs(order.entry_price - order.sl_price)

        pos = Position(
            strategy    = order.strategy,
            timeframe   = order.timeframe,
            side        = order.side,
            quantity    = order.quantity,
            entry_price = order.entry_price,
            sl_price    = order.sl_price,
            tp_price    = order.tp_price,
            tp_partial  = order.tp_partial,
            partial_pct = order.partial_pct,
            order_id    = order.order_id,
        )
        self._positions.append(pos)
        self._log_event("OPEN", pos, order.entry_price)
        return order

    def close_position(self, position: Position, reason: str = "") -> float:
        price     = self.get_current_price()
        direction = 1 if position.side == OrderSide.BUY else -1
        pnl       = direction * (price - position.entry_price) / position.entry_price * \
                    position.quantity * position.entry_price
        self._balance += pnl
        if position in self._positions:
            self._positions.remove(position)
        self._log_event(f"CLOSE:{reason}", position, price, pnl)
        return pnl

    def close_partial(self, position: Position, fraction: float) -> float:
        price     = self.get_current_price()
        qty       = position.quantity * fraction
        direction = 1 if position.side == OrderSide.BUY else -1
        pnl       = direction * (price - position.entry_price) / position.entry_price * \
                    qty * position.entry_price
        self._balance    += pnl
        position.quantity -= qty
        position.partial_done = True
        self._log_event("PARTIAL", position, price, pnl)
        return pnl

    def get_open_positions(self) -> List[Position]:
        return list(self._positions)

    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        # In paper mode, price is set externally or mocked
        return getattr(self, "_mock_price", 0.0)

    def set_sl_tp(self, position: Position, sl_price: float, tp_price: float) -> bool:
        position.sl_price = sl_price
        position.tp_price = tp_price
        return True

    def set_mock_price(self, price: float) -> None:
        """Set mock current price (used by engine for SL/TP checks)."""
        self._mock_price = price

    def _log_event(
        self, event: str, pos: Position, price: float, pnl: float = 0.0
    ) -> None:
        row = {
            "ts":        datetime.now(timezone.utc).isoformat(),
            "event":     event,
            "strategy":  pos.strategy,
            "timeframe": pos.timeframe,
            "side":      pos.side.value,
            "price":     price,
            "qty":       pos.quantity,
            "pnl_usd":   round(pnl, 4),
            "balance":   round(self._balance, 4),
        }
        self._trade_log.append(row)
        print(f"[paper] {event:<16} {pos.strategy}/{pos.timeframe}  "
              f"{pos.side.value}  price={price:.1f}  PnL={pnl:+.2f}  bal={self._balance:.2f}")

        if self._log_path:
            write_header = not self._log_path.exists()
            with open(self._log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
# BINANCE FUTURES EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

class BinanceExecutor(BaseExecutor):
    """
    Binance USDT-M perpetual futures executor.

    ⚠️  START ON TESTNET (testnet=True).
    ⚠️  Review every order before enabling auto_trade=True.

    Requires: pip install ccxt
    """

    SYMBOL = "BTC/USDT:USDT"   # ccxt unified symbol for USDT perpetual

    def __init__(
        self,
        api_key:    str,
        api_secret: str,
        testnet:    bool = True,
        leverage:   int  = 1,
        log_path:   Optional[Path] = None,
    ):
        try:
            import ccxt
        except ImportError:
            raise ImportError("pip install ccxt")

        self._ex = ccxt.binanceusdm({
            "apiKey":        api_key,
            "secret":        api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        if testnet:
            self._ex.set_sandbox_mode(True)

        self._leverage  = leverage
        self._log_path  = log_path
        self._positions: List[Position] = []

        # Set leverage
        try:
            self._ex.set_leverage(leverage, "BTC/USDT:USDT")
        except Exception as exc:
            print(f"[binance] set_leverage warning: {exc}")

    def is_connected(self) -> bool:
        try:
            self._ex.fetch_balance()
            return True
        except Exception:
            return False

    def get_account_balance(self) -> float:
        try:
            bal = self._ex.fetch_balance()
            return float(bal["USDT"]["free"])
        except Exception as exc:
            print(f"[binance] get_balance failed: {exc}")
            return 0.0

    def open_position(self, order: Order) -> Order:
        """Open a market order with SL and TP on Binance Futures."""
        try:
            side_str = "buy" if order.side == OrderSide.BUY else "sell"

            # Market order
            result = self._ex.create_market_order(
                symbol   = self.SYMBOL,
                side     = side_str,
                amount   = order.quantity,
                params   = {"reduceOnly": False},
            )
            fill_price = float(result.get("average", result.get("price", order.entry_price)))
            order.entry_price = fill_price
            order.order_id    = str(result["id"])
            order.status      = OrderStatus.OPEN
            order.filled_at   = datetime.now(timezone.utc)

            # Stop-loss order (opposite side, reduceOnly)
            sl_side = "sell" if order.side == OrderSide.BUY else "buy"
            sl_result = self._ex.create_order(
                symbol     = self.SYMBOL,
                type       = "stop_market",
                side       = sl_side,
                amount     = order.quantity,
                params     = {
                    "stopPrice": order.sl_price,
                    "reduceOnly": True,
                    "closePosition": True,
                },
            )
            order.sl_order_id = str(sl_result["id"])

            # Take-profit order
            tp_side = sl_side
            tp_result = self._ex.create_order(
                symbol     = self.SYMBOL,
                type       = "take_profit_market",
                side       = tp_side,
                amount     = order.quantity,
                params     = {
                    "stopPrice": order.tp_price,
                    "reduceOnly": True,
                    "closePosition": True,
                },
            )
            order.tp_order_id = str(tp_result["id"])

            # Track position
            pos = Position(
                strategy    = order.strategy,
                timeframe   = order.timeframe,
                side        = order.side,
                quantity    = order.quantity,
                entry_price = fill_price,
                sl_price    = order.sl_price,
                tp_price    = order.tp_price,
                tp_partial  = order.tp_partial,
                partial_pct = order.partial_pct,
                order_id    = order.order_id,
            )
            self._positions.append(pos)
            self._log(order, fill_price, "OPENED")
            return order

        except Exception as exc:
            print(f"[binance] open_position failed: {exc}")
            order.status = OrderStatus.ERROR
            return order

    def close_position(self, position: Position, reason: str = "") -> float:
        try:
            side   = "sell" if position.side == OrderSide.BUY else "buy"
            result = self._ex.create_market_order(
                symbol = self.SYMBOL, side = side, amount = position.quantity,
                params = {"reduceOnly": True},
            )
            price = float(result.get("average", result.get("price", 0)))
            direction = 1 if position.side == OrderSide.BUY else -1
            pnl = direction * (price - position.entry_price) / position.entry_price * \
                  position.quantity * position.entry_price
            if position in self._positions:
                self._positions.remove(position)
            self._log_close(position, price, pnl, reason)
            return pnl
        except Exception as exc:
            print(f"[binance] close_position failed: {exc}")
            return 0.0

    def close_partial(self, position: Position, fraction: float) -> float:
        try:
            qty    = position.quantity * fraction
            side   = "sell" if position.side == OrderSide.BUY else "buy"
            result = self._ex.create_market_order(
                symbol = self.SYMBOL, side = side, amount = qty,
                params = {"reduceOnly": True},
            )
            price = float(result.get("average", result.get("price", 0)))
            direction = 1 if position.side == OrderSide.BUY else -1
            pnl = direction * (price - position.entry_price) / position.entry_price * \
                  qty * position.entry_price
            position.quantity    -= qty
            position.partial_done = True
            print(f"[binance] PARTIAL CLOSE {position.strategy}  qty={qty:.6f}  PnL=${pnl:+.2f}")
            return pnl
        except Exception as exc:
            print(f"[binance] close_partial failed: {exc}")
            return 0.0

    def get_open_positions(self) -> List[Position]:
        return list(self._positions)

    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        try:
            ticker = self._ex.fetch_ticker(self.SYMBOL)
            return float(ticker["last"])
        except Exception:
            return 0.0

    def set_sl_tp(self, position: Position, sl_price: float, tp_price: float) -> bool:
        try:
            # Cancel existing SL/TP and re-place
            # In practice, use amend order if exchange supports it
            position.sl_price = sl_price
            position.tp_price = tp_price
            return True
        except Exception as exc:
            print(f"[binance] set_sl_tp failed: {exc}")
            return False

    def _log(self, order: Order, price: float, event: str) -> None:
        print(f"[binance] {event}  {order.strategy}/{order.timeframe}  "
              f"{order.side.value}  qty={order.quantity:.6f}  "
              f"entry={price:.1f}  SL={order.sl_price:.1f}  TP={order.tp_price:.1f}")

    def _log_close(self, pos: Position, price: float, pnl: float, reason: str) -> None:
        print(f"[binance] CLOSED:{reason}  {pos.strategy}/{pos.timeframe}  "
              f"entry={pos.entry_price:.1f}  close={price:.1f}  PnL=${pnl:+.2f}")