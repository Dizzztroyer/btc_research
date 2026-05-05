"""
live/notifications/telegram.py
────────────────────────────────
Telegram notification system for trading signals.

Sends formatted signal messages to a Telegram chat before entry.
Supports both signal-only and trade-confirmation modes.

Setup:
    1. Create a bot via @BotFather → get token
    2. Add bot to your channel/group → get chat_id
    3. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SignalMessage:
    """Structured trading signal for formatting."""
    strategy_label: str       # e.g. "swing/8h"
    direction:      str       # "LONG" or "SHORT"
    est_entry:      float     # estimated next candle open
    sl_price:       float     # stop-loss
    tp_price:       float     # take-profit (primary)
    tp_partial:     float     # partial TP level (if applicable)
    risk_pct:       float     # risk as fraction of capital
    timeframe:      str       # "8h", "12h", "6h"
    signal_time:    datetime  # when signal was generated (candle close time)
    # Optional enrichment
    r_ratio:        float = 0.0   # reward-to-risk ratio
    atr_regime:     str   = ""    # "LOW", "MEDIUM", "HIGH"
    ml_mult:        float = 1.0   # ML position size multiplier
    candle_close:   float = 0.0   # price at signal generation


@dataclass
class TradeUpdate:
    """Trade lifecycle update (open/partial/close)."""
    strategy_label: str
    event:          str    # "OPENED", "PARTIAL_CLOSE", "CLOSED", "SL_HIT", "TP_HIT"
    direction:      str
    entry_price:    float
    current_price:  float
    pnl_pct:        float
    pnl_usd:        float
    timeframe:      str


class TelegramNotifier:
    """
    Sends messages to Telegram via Bot API.
    Uses only stdlib urllib (no dependencies).
    """

    BASE_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str, chat_id: str, alert_chat_id: str = ""):
        self.token        = token
        self.chat_id      = chat_id
        self.alert_chat_id= alert_chat_id or chat_id
        self._last_msg_ts = 0.0

    def _request(self, method: str, payload: dict) -> bool:
        """Make Telegram API request."""
        url  = self.BASE_URL.format(token=self.token, method=method)
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                return result.get("ok", False)
        except urllib.error.URLError as exc:
            print(f"[telegram] Request failed: {exc}")
            return False
        except Exception as exc:
            print(f"[telegram] Unexpected error: {exc}")
            return False

    def send_message(
        self,
        text:        str,
        chat_id:     Optional[str] = None,
        parse_mode:  str = "HTML",
        silent:      bool = False,
    ) -> bool:
        """Send a text message."""
        if not self.token or not self.chat_id:
            print(f"[telegram] NOT configured — would send:\n{text}\n")
            return False

        payload = {
            "chat_id":              chat_id or self.chat_id,
            "text":                 text,
            "parse_mode":           parse_mode,
            "disable_notification": silent,
        }
        ok = self._request("sendMessage", payload)
        if ok:
            self._last_msg_ts = time.time()
        return ok

    # ── Signal formatting ──────────────────────────────────────────────────────

    def format_signal(self, sig: SignalMessage) -> str:
        """Format a trading signal for Telegram (HTML)."""
        icon  = "📈" if sig.direction == "LONG" else "📉"
        color = "🟢" if sig.direction == "LONG" else "🔴"

        sl_pct  = abs(sig.est_entry - sig.sl_price) / sig.est_entry * 100
        tp_pct  = abs(sig.tp_price - sig.est_entry) / sig.est_entry * 100
        rr      = tp_pct / sl_pct if sl_pct > 0 else 0

        regime_str = f"\nATR Regime:  <code>{sig.atr_regime}</code>" if sig.atr_regime else ""
        ml_str     = f"\nML mult:     <code>{sig.ml_mult:.2f}×</code>" if sig.ml_mult != 1.0 else ""
        partial_str= ""
        if sig.tp_partial > 0:
            partial_str = f"\nPartial TP:  <code>{sig.tp_partial:.0f}</code> (50% close @ 1R)"

        ts_str = sig.signal_time.strftime("%Y-%m-%d %H:%M UTC")

        return (
            f"{icon} <b>{sig.strategy_label.upper()} — {color} {sig.direction}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Strategy:    <code>{sig.strategy_label}</code>\n"
            f"Entry:       MARKET (next candle open)\n"
            f"Est. entry:  <code>{sig.est_entry:.1f}</code>\n"
            f"Stop-loss:   <code>{sig.sl_price:.1f}</code>  ({sl_pct:.1f}%)\n"
            f"Take-profit: <code>{sig.tp_price:.1f}</code>  ({tp_pct:.1f}%){partial_str}\n"
            f"R/R ratio:   <code>{rr:.2f}</code>\n"
            f"Risk:        <code>{sig.risk_pct*100:.1f}%</code> of capital"
            f"{regime_str}{ml_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>⏰ Signal at {ts_str}</i>\n"
            f"<i>BTC Research System</i>"
        )

    def format_trade_update(self, upd: TradeUpdate) -> str:
        """Format a trade lifecycle update."""
        pnl_icon = "✅" if upd.pnl_usd >= 0 else "❌"
        event_icons = {
            "OPENED":        "🔔",
            "PARTIAL_CLOSE": "📤",
            "CLOSED":        "🏁",
            "SL_HIT":        "🛑",
            "TP_HIT":        "🎯",
        }
        icon = event_icons.get(upd.event, "ℹ️")

        return (
            f"{icon} <b>{upd.strategy_label} — {upd.event}</b>\n"
            f"Direction: {upd.direction}\n"
            f"Entry:  <code>{upd.entry_price:.1f}</code>\n"
            f"Now:    <code>{upd.current_price:.1f}</code>\n"
            f"PnL:    {pnl_icon} <code>{upd.pnl_pct*100:+.2f}%  (${upd.pnl_usd:+.2f})</code>\n"
            f"<i>BTC Research System</i>"
        )

    def format_system_alert(self, level: str, message: str) -> str:
        """Format a system alert (WARNING, ERROR, INFO)."""
        icons = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "🚨", "HALT": "🛑"}
        icon  = icons.get(level, "ℹ️")
        ts    = datetime.now(timezone.utc).strftime("%H:%M UTC")
        return f"{icon} <b>[{level}] BTC Research System</b>\n{message}\n<i>{ts}</i>"

    # ── Convenience senders ────────────────────────────────────────────────────

    def send_signal(self, sig: SignalMessage) -> bool:
        text = self.format_signal(sig)
        return self.send_message(text)

    def send_trade_update(self, upd: TradeUpdate) -> bool:
        text = self.format_trade_update(upd)
        return self.send_message(text)

    def send_alert(self, level: str, message: str) -> bool:
        text = self.format_system_alert(level, message)
        return self.send_message(text, chat_id=self.alert_chat_id)

    def send_startup(self, mode: str, strategies: list) -> bool:
        strat_list = "\n".join(f"  • {s}" for s in strategies)
        text = (
            f"🚀 <b>BTC Research System — STARTED</b>\n"
            f"Mode: <code>{mode}</code>\n"
            f"Strategies:\n{strat_list}\n"
            f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>"
        )
        return self.send_message(text)

    def send_daily_summary(self, stats: dict) -> bool:
        lines = [f"📊 <b>Daily Summary — {stats.get('date','')}</b>"]
        lines.append(f"Signals sent:  <code>{stats.get('signals',0)}</code>")
        lines.append(f"Trades open:   <code>{stats.get('open_trades',0)}</code>")
        if "pnl_usd" in stats:
            pnl = stats["pnl_usd"]
            icon = "✅" if pnl >= 0 else "❌"
            lines.append(f"Daily PnL:     {icon} <code>${pnl:+.2f}</code>")
        text = "\n".join(lines) + "\n<i>BTC Research System</i>"
        return self.send_message(text)

    def test(self) -> bool:
        """Send a test message to verify configuration."""
        return self.send_message(
            "✅ <b>Telegram test OK</b>\nBTC Research System is connected."
        )