"""
live/execution/base.py
───────────────────────
Abstract execution layer.

Defines the interface that all execution backends must implement.
Concrete implementations: BinanceExecutor, PaperExecutor.

To add a new exchange/prop platform:
    1. Subclass BaseExecutor
    2. Implement the abstract methods
    3. Register in live/execution/__init__.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    OPEN      = "OPEN"
    PARTIAL   = "PARTIAL"
    CLOSED    = "CLOSED"
    CANCELLED = "CANCELLED"
    ERROR     = "ERROR"


@dataclass
class Order:
    """Represents a trade order."""
    strategy:    str
    timeframe:   str
    side:        OrderSide
    quantity:    float      # in base currency (BTC)
    entry_price: float      # fill price (0 if not yet filled)
    sl_price:    float
    tp_price:    float
    tp_partial:  float = 0.0    # partial TP level (0 = disabled)
    partial_pct: float = 0.0    # fraction to close at partial TP

    # Filled by exchange
    order_id:    str       = ""
    sl_order_id: str       = ""
    tp_order_id: str       = ""
    status:      OrderStatus = OrderStatus.PENDING
    created_at:  datetime  = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at:   Optional[datetime] = None
    closed_at:   Optional[datetime] = None

    # P&L tracking
    close_price: float = 0.0
    realized_pnl:float = 0.0
    risk_usd:    float = 0.0


@dataclass
class Position:
    """Current open position for a strategy."""
    strategy:    str
    timeframe:   str
    side:        OrderSide
    quantity:    float
    entry_price: float
    sl_price:    float
    tp_price:    float
    tp_partial:  float = 0.0
    partial_done:bool  = False
    partial_pct: float = 0.0
    order_id:    str   = ""
    opened_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    unrealized_pnl: float = 0.0

    def update_pnl(self, current_price: float) -> None:
        direction = 1 if self.side == OrderSide.BUY else -1
        self.unrealized_pnl = direction * (current_price - self.entry_price) / self.entry_price


class BaseExecutor(ABC):
    """
    Abstract execution backend.

    All methods that interact with an exchange must be implemented here.
    The signal engine calls these methods without knowing which exchange is used.
    """

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the connection to the exchange is healthy."""
        ...

    @abstractmethod
    def get_account_balance(self) -> float:
        """Return available capital in USDT."""
        ...

    @abstractmethod
    def open_position(self, order: Order) -> Order:
        """
        Open a market position.

        Sets order.entry_price, order.order_id, order.status = OPEN.
        Also sets SL and TP orders on the exchange.
        Returns the updated Order.
        """
        ...

    @abstractmethod
    def close_position(self, position: Position, reason: str = "") -> float:
        """
        Close an open position at market.

        Returns the realized PnL in USDT.
        """
        ...

    @abstractmethod
    def close_partial(self, position: Position, fraction: float) -> float:
        """
        Close fraction of position (e.g. 0.5 = 50%).

        Returns realized PnL for the closed portion.
        Modifies position.quantity in place.
        """
        ...

    @abstractmethod
    def get_open_positions(self) -> List[Position]:
        """Return list of currently open positions."""
        ...

    @abstractmethod
    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """Return the current market price."""
        ...

    @abstractmethod
    def set_sl_tp(
        self,
        position: Position,
        sl_price: float,
        tp_price: float,
    ) -> bool:
        """Update SL and TP for an open position. Returns True on success."""
        ...

    def compute_quantity(
        self,
        capital:    float,
        risk_pct:   float,
        entry:      float,
        sl:         float,
    ) -> float:
        """
        Compute position size in BTC such that risk = risk_pct × capital.

        quantity = (capital × risk_pct) / |entry - sl|
        """
        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            return 0.0
        risk_usd = capital * risk_pct
        return risk_usd / risk_per_unit