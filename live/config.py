"""
live/config.py
──────────────
Live trading system configuration.

Edit this file to configure your trading system.
Never commit API keys — use environment variables or .env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ── Load .env if present ───────────────────────────────────────────────────────
def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ── Strategy definitions (final portfolio) ─────────────────────────────────────
@dataclass
class StrategyConfig:
    name:          str
    timeframe:     str         # "8h", "12h", "6h"
    risk_pct:      float       # fraction of capital at risk per trade
    direction:     str = "both"
    # ML + ATR flags
    use_ml:        bool = True
    use_atr:       bool = False  # only 8h uses ATR
    # Exit settings
    long_tp_mult:  float = 1.0
    short_tp_mult: float = 1.0
    long_partial:  Optional[float] = None   # fraction to close at 1R
    short_partial: Optional[float] = None
    partial_r:     float = 1.0
    # Display
    label:         str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"swing/{self.timeframe}"


PORTFOLIO: List[StrategyConfig] = [
    StrategyConfig(
        name="swing_breakout", timeframe="8h",  risk_pct=0.012,
        use_ml=True, use_atr=True,
        long_tp_mult=1.2, short_tp_mult=0.8,
        long_partial=0.5, short_partial=0.5, partial_r=1.0,
        label="swing/8h",
    ),
    StrategyConfig(
        name="swing_breakout", timeframe="12h", risk_pct=0.012,
        use_ml=True, use_atr=False,
        long_partial=0.5, short_partial=0.7, partial_r=1.0,
        label="swing/12h",
    ),
    StrategyConfig(
        name="swing_breakout", timeframe="6h",  risk_pct=0.006,
        use_ml=True, use_atr=False,
        label="swing/6h",
    ),
]


# ── System modes ───────────────────────────────────────────────────────────────
@dataclass
class ModeConfig:
    # Signal generation
    signals_only: bool = True    # True = notify only, no orders
    auto_trade:   bool = False   # True = send orders to exchange

    # Safety limits
    max_open_positions: int   = 3      # max simultaneous positions
    max_daily_loss_pct: float = 0.05   # halt trading if daily loss > 5%
    min_signal_interval_sec: int = 60  # minimum time between same-TF signals

    # Paper trading mode (log orders but don't execute)
    paper_mode: bool = True


# ── Telegram ───────────────────────────────────────────────────────────────────
@dataclass
class TelegramConfig:
    bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN"))
    chat_id:   str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID"))
    enabled:   bool = True
    # Advanced: different chat for alerts vs signals
    alert_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_ALERT_CHAT_ID", ""))

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


# ── Exchange: Binance ──────────────────────────────────────────────────────────
@dataclass
class BinanceConfig:
    api_key:    str = field(default_factory=lambda: _env("BINANCE_API_KEY"))
    api_secret: str = field(default_factory=lambda: _env("BINANCE_API_SECRET"))
    symbol:     str = "BTCUSDT"
    testnet:    bool = True    # ALWAYS start on testnet
    leverage:   int  = 1       # 1x = no leverage (safest)
    margin_type: str = "ISOLATED"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
STATE_DIR  = BASE_DIR / "live" / "state"
LOG_DIR    = BASE_DIR / "outputs" / "logs"
STATE_FILE = STATE_DIR / "positions.json"
TRADES_LOG = LOG_DIR   / "live_trades.csv"
SIGNALS_LOG= LOG_DIR   / "live_signals.csv"


# ── Feature and model paths ────────────────────────────────────────────────────
RESULTS_CSV  = BASE_DIR / "outputs" / "rankings" / "all_results.csv"
FEATURES_DIR = BASE_DIR / "data" / "features" / "BTCUSDT"
RAW_DIR      = BASE_DIR / "data" / "raw" / "BTCUSDT"


# ── Global instances ───────────────────────────────────────────────────────────
MODE     = ModeConfig()
TELEGRAM = TelegramConfig()
BINANCE  = BinanceConfig()


def print_config() -> None:
    """Print current configuration (without secrets)."""
    print("=== Live Trading Configuration ===")
    print(f"  Mode:         signals_only={MODE.signals_only}  auto_trade={MODE.auto_trade}  paper={MODE.paper_mode}")
    print(f"  Telegram:     {'✓ configured' if TELEGRAM.is_configured() else '✗ NOT configured'}")
    print(f"  Binance:      {'✓ configured' if BINANCE.is_configured() else '✗ NOT configured'}")
    print(f"  Symbol:       {BINANCE.symbol}")
    print(f"  Testnet:      {BINANCE.testnet}")
    print(f"  Portfolio:")
    for s in PORTFOLIO:
        print(f"    {s.label:<12}: risk={s.risk_pct:.1%}  ML={s.use_ml}  ATR={s.use_atr}")
    print(f"  Safety:")
    print(f"    max_positions={MODE.max_open_positions}")
    print(f"    max_daily_loss={MODE.max_daily_loss_pct:.1%}")