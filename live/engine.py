"""
live/engine.py
───────────────
Main signal engine.

Runs continuously, checks for new closed candles,
generates signals, sends Telegram notifications,
and optionally executes trades.

Architecture:
  1. For each strategy TF: check if new candle closed
  2. If yes: fetch OHLCV, build features, generate signal
  3. If signal exists: compute SL/TP, ML mult, ATR regime
  4. Notify via Telegram
  5. If auto_trade: send order to executor
  6. Track open positions, check partial TP and final exit

Entry timing (matching backtest):
  Signal is generated on BAR CLOSE.
  Entry is on NEXT BAR OPEN (market order).
  This matches the backtest methodology exactly.

Usage:
    from live.engine import SignalEngine
    engine = SignalEngine()
    engine.run()   # blocks, runs forever
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from live.config import (
    PORTFOLIO, MODE, TELEGRAM, BINANCE,
    STATE_FILE, TRADES_LOG, SIGNALS_LOG, RESULTS_CSV,
    StrategyConfig, print_config,
)
from live.data.feed import LiveDataFeed
from live.notifications.telegram import TelegramNotifier, SignalMessage, TradeUpdate
from live.execution.base import BaseExecutor, Order, OrderSide, OrderStatus, Position
from live.execution.executors import PaperExecutor, BinanceExecutor

from src.utils.logger import get_logger
from src.ml.ml_sizer_v2 import MLSizerV2, SizerConfig
from src.strategies.structure import SwingBreakoutStrategy

logger = get_logger("live_engine", log_file=Path("outputs/logs/live_engine.log"))


# ── ATR regime ────────────────────────────────────────────────────────────────

def _atr_regime(df, lookback=200, lp=33, hp=66):
    """Return ATR regime for the last bar: 'LOW', 'MEDIUM', 'HIGH'."""
    import pandas as pd, numpy as np
    atr_col = next((c for c in ["atr_14_pct","atr_14"] if c in df.columns), None)
    if atr_col is None:
        return "MEDIUM"
    atr = df[atr_col].iloc[-2]  # prior bar (no lookahead)
    roll = df[atr_col].rolling(lookback, min_periods=50)
    low  = roll.quantile(lp/100).iloc[-2]
    high = roll.quantile(hp/100).iloc[-2]
    if atr < low:  return "LOW"
    if atr > high: return "HIGH"
    return "MEDIUM"


def _atr_mult(regime: str, low_m=1.3, high_m=0.7) -> float:
    return {"LOW": low_m, "HIGH": high_m}.get(regime, 1.0)


def _compute_tp(entry, sl, side, tp_base, tp_mult=1.0):
    """Apply TP multiplier: extend/shrink TP distance."""
    dist = abs(tp_base - entry) * tp_mult
    return entry + side * dist


def _compute_partial_tp(entry, sl, side) -> float:
    """Partial TP at 1R."""
    risk = abs(entry - sl)
    return entry + side * risk


# ── State persistence ─────────────────────────────────────────────────────────

class StateManager:
    """Persists engine state (open positions, last signal times) to JSON."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except Exception:
                pass
        return {"positions": {}, "last_signal": {}, "daily_pnl": 0.0, "signals_today": 0}

    def save(self) -> None:
        self.path.write_text(json.dumps(self._state, indent=2, default=str))

    def get_last_signal_ts(self, tf: str) -> float:
        return self._state["last_signal"].get(tf, 0.0)

    def set_last_signal_ts(self, tf: str, ts: float) -> None:
        self._state["last_signal"][tf] = ts
        self.save()

    def add_daily_pnl(self, pnl: float) -> None:
        self._state["daily_pnl"] = self._state.get("daily_pnl", 0.0) + pnl
        self.save()

    def get_daily_pnl(self) -> float:
        return self._state.get("daily_pnl", 0.0)

    def reset_daily(self) -> None:
        self._state["daily_pnl"] = 0.0
        self._state["signals_today"] = 0
        self.save()

    def incr_signals(self) -> None:
        self._state["signals_today"] = self._state.get("signals_today", 0) + 1
        self.save()


# ── Signal logger ─────────────────────────────────────────────────────────────

class SignalLogger:
    """Logs all signals to CSV."""

    COLS = ["ts","strategy","timeframe","direction","entry","sl","tp",
            "tp_partial","risk_pct","ml_mult","atr_regime","notified","executed"]

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            import csv
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.COLS).writeheader()

    def log(self, row: dict) -> None:
        import csv
        row_out = {k: row.get(k, "") for k in self.COLS}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.COLS).writerow(row_out)


# ── Main engine ───────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Main live trading engine.

    Loops over all portfolio strategies, checks for new signals,
    sends notifications, and optionally executes trades.
    """

    # How often to check each TF (seconds before its candle close)
    CHECK_INTERVAL_SEC = 30

    def __init__(self):
        logger.info("Initializing Signal Engine...")
        print_config()

        self.feed     = LiveDataFeed(logger=logger)
        self.state    = StateManager(STATE_FILE)
        self.sig_log  = SignalLogger(SIGNALS_LOG)

        # Telegram
        self.tg = TelegramNotifier(
            token        = TELEGRAM.bot_token,
            chat_id      = TELEGRAM.chat_id,
            alert_chat_id= TELEGRAM.alert_chat_id,
        )

        # Executor (paper or live)
        self.executor: BaseExecutor = self._build_executor()

        # ML sizers per strategy (trained once on startup)
        self.sizers: Dict[str, Optional[MLSizerV2]] = {}
        self.strategy = SwingBreakoutStrategy()

        # Strategy params
        self.params: Dict[str, Optional[dict]] = {}

        # Candle close tracker (last seen bar timestamps)
        self._last_bar: Dict[str, Optional[datetime]] = {s.timeframe: None for s in PORTFOLIO}

        self._running = False

    def _build_executor(self) -> BaseExecutor:
        if MODE.paper_mode or not MODE.auto_trade:
            logger.info("Using PAPER executor (no real orders)")
            return PaperExecutor(log_path=TRADES_LOG)
        if BINANCE.is_configured():
            logger.info(f"Using BINANCE executor (testnet={BINANCE.testnet})")
            return BinanceExecutor(
                api_key    = BINANCE.api_key,
                api_secret = BINANCE.api_secret,
                testnet    = BINANCE.testnet,
                leverage   = BINANCE.leverage,
                log_path   = TRADES_LOG,
            )
        raise RuntimeError("No executor configured. Set BINANCE_API_KEY or use paper mode.")

    # ── Startup ────────────────────────────────────────────────────────────────

    def startup(self) -> None:
        """Load params and train ML sizers on startup."""
        logger.info("Loading strategy params and training ML sizers...")

        import pandas as pd
        from src.utils.config_loader import load_config
        from src.features.feature_engine import FeatureEngine

        cfg    = load_config()
        fe     = FeatureEngine(cfg)
        sc_cfg = SizerConfig(n_estimators=400, min_train_trades=150,
                             min_val_trades=30,  min_mult_std=0.05)

        for strat_cfg in PORTFOLIO:
            tf = strat_cfg.timeframe
            # Load best params from research results
            self.params[tf] = self._load_params(tf)
            if self.params[tf] is None:
                logger.warning(f"  {tf}: no params found in all_results.csv")
                self.sizers[tf] = None
                continue

            # Try to train ML sizer on historical data
            try:
                df    = fe.load(tf)
                is_n  = int(len(df) * cfg.validation.is_ratio)
                df_is = df.iloc[:is_n]

                sizer = MLSizerV2(cfg, sc_cfg)
                ok    = sizer.fit(self.strategy, self.params[tf], df_is,
                                   direction="both", label=f"live/{tf}")
                self.sizers[tf] = sizer if ok else None
                logger.info(f"  {tf}: ML {'enabled' if ok else 'disabled'} "
                            f"(AUC={sizer.auc:.3f})" if ok else f"  {tf}: ML disabled")
            except Exception as exc:
                logger.warning(f"  {tf}: ML training failed: {exc}")
                self.sizers[tf] = None

        # Send startup notification
        strat_labels = [s.label for s in PORTFOLIO]
        mode_str = "SIGNALS_ONLY" if MODE.signals_only else "AUTO_TRADE"
        if MODE.paper_mode:
            mode_str += " (PAPER)"
        self.tg.send_startup(mode_str, strat_labels)
        logger.info("Startup complete.")

    def _load_params(self, tf: str) -> Optional[dict]:
        """Load best strategy params from research results."""
        import pandas as pd
        if not RESULTS_CSV.exists():
            return None
        df   = pd.read_csv(RESULTS_CSV)
        mask = (df["strategy"]=="swing_breakout")&(df["timeframe"]==tf)
        sub  = df[mask]
        if sub.empty:
            return None
        best   = sub.sort_values("robustness", ascending=False).iloc[0]
        p_cols = [c for c in best.index if c.startswith("p_") and not pd.isna(best[c])]
        out    = {}
        for c in p_cols:
            v=best[c]; k=c[2:]
            out[k] = int(v) if isinstance(v,float) and v==int(v) else v
        return out

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main loop. Runs until interrupted."""
        self.startup()
        self._running = True
        logger.info("Engine running. Ctrl+C to stop.")

        try:
            while self._running:
                self._tick()
                time.sleep(self.CHECK_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            self.tg.send_alert("INFO", "Engine stopped (KeyboardInterrupt).")
        except Exception as exc:
            logger.error(f"Fatal error: {exc}")
            self.tg.send_alert("ERROR", f"Engine crashed: {exc}")
            raise

    def _tick(self) -> None:
        """One iteration: check all strategies for new signals."""
        now = datetime.now(timezone.utc)

        # Daily reset at 00:00 UTC
        if now.hour == 0 and now.minute < 1:
            self.state.reset_daily()

        # Safety: daily loss limit
        if self.state.get_daily_pnl() < -(self._get_balance() * MODE.max_daily_loss_pct):
            logger.warning("Daily loss limit hit — halting trading")
            self.tg.send_alert("HALT", f"Daily loss limit reached ({MODE.max_daily_loss_pct:.0%}). Trading halted.")
            MODE.auto_trade = False
            return

        for strat_cfg in PORTFOLIO:
            try:
                self._check_strategy(strat_cfg, now)
            except Exception as exc:
                logger.error(f"Error checking {strat_cfg.label}: {exc}")

        # Check open positions for partial TP / SL hit
        self._monitor_positions()

    def _check_strategy(self, strat_cfg: StrategyConfig, now: datetime) -> None:
        """Check one strategy for a new signal on its TF."""
        tf = strat_cfg.timeframe

        # Fetch latest data
        df = self.feed.get_latest(tf, n_candles=600, cache_sec=30)
        if df is None or len(df) < 200:
            return

        last_bar_ts = df["timestamp"].iloc[-1]

        # Check if we have a new closed bar
        if self._last_bar[tf] is not None and last_bar_ts <= self._last_bar[tf]:
            return  # no new bar

        prev_bar = self._last_bar[tf]
        self._last_bar[tf] = last_bar_ts

        if prev_bar is None:
            logger.info(f"  {tf}: first bar seen at {last_bar_ts}")
            return  # skip first tick to avoid stale signals

        # Throttle: don't repeat signal within min_interval
        ts_float = last_bar_ts.timestamp()
        if ts_float - self.state.get_last_signal_ts(tf) < MODE.min_signal_interval_sec:
            return

        logger.info(f"  New bar closed: {tf} at {last_bar_ts}")

        # Generate signals using strategy
        if self.params[tf] is None:
            return

        df_sig = self.strategy.generate_signals(df.copy(), self.params[tf])
        last   = df_sig.iloc[-1]  # the just-closed bar

        signal = int(last.get("signal", 0))
        if signal == 0:
            return

        # Respect direction config
        if strat_cfg.direction == "long"  and signal != 1:  return
        if strat_cfg.direction == "short" and signal != -1: return

        logger.info(f"  ✓ Signal: {tf}  {'LONG' if signal==1 else 'SHORT'}")
        self._handle_signal(strat_cfg, df_sig, last, signal, last_bar_ts)

    def _handle_signal(
        self,
        strat_cfg:   StrategyConfig,
        df_sig,
        last_bar,
        signal:      int,
        bar_ts:      datetime,
    ) -> None:
        """Process a confirmed signal: compute levels, notify, optionally trade."""
        import numpy as np

        side     = signal  # 1=long, -1=short
        side_str = "LONG" if side == 1 else "SHORT"

        # Current price = last close (approximate next open)
        est_entry = float(last_bar["close"])
        sl_raw    = float(last_bar.get("sl_price", np.nan))
        tp_raw    = float(last_bar.get("tp_price", np.nan))

        # Use signal SL/TP if valid, else fallback
        default_sl_pct = 0.025
        default_tp_pct = 0.06
        sl = sl_raw if not np.isnan(sl_raw) and sl_raw > 0 else est_entry * (1 - side * default_sl_pct)
        tp = tp_raw if not np.isnan(tp_raw) and tp_raw > 0 else est_entry * (1 + side * default_tp_pct)

        # Apply asymmetric TP multiplier
        tp_mult = strat_cfg.long_tp_mult if side == 1 else strat_cfg.short_tp_mult
        if tp_mult != 1.0:
            tp = _compute_tp(est_entry, sl, side, tp, tp_mult)

        # Partial TP level (1R from entry)
        partial_pct = strat_cfg.long_partial if side == 1 else strat_cfg.short_partial
        tp_partial  = _compute_partial_tp(est_entry, sl, side) if partial_pct else 0.0

        # ML size multiplier
        ml_mult = 1.0
        sizer   = self.sizers.get(strat_cfg.timeframe)
        if sizer and sizer.enabled:
            try:
                mults_series = sizer.predict(df_sig, self.strategy, self.params[strat_cfg.timeframe])
                ml_mult = float(mults_series.iloc[-1])
            except Exception:
                ml_mult = 1.0

        # ATR regime
        regime = "MEDIUM"
        atr_mult_val = 1.0
        if strat_cfg.use_atr:
            regime       = _atr_regime(df_sig)
            atr_mult_val = _atr_mult(regime)

        # Combined size multiplier
        size_mult = max(0.5, min(2.0, ml_mult * atr_mult_val))

        # Effective risk
        eff_risk = strat_cfg.risk_pct * size_mult

        # R/R ratio
        rr = abs(tp - est_entry) / abs(est_entry - sl) if abs(est_entry - sl) > 0 else 0.0

        logger.info(
            f"    {strat_cfg.label} {side_str}: entry≈{est_entry:.1f}  "
            f"SL={sl:.1f}  TP={tp:.1f}  R/R={rr:.2f}  "
            f"ML={ml_mult:.2f}  ATR={regime}  risk={eff_risk:.1%}"
        )

        # Build signal object for Telegram
        sig_msg = SignalMessage(
            strategy_label = strat_cfg.label,
            direction      = side_str,
            est_entry      = est_entry,
            sl_price       = sl,
            tp_price       = tp,
            tp_partial     = tp_partial,
            risk_pct       = eff_risk,
            timeframe      = strat_cfg.timeframe,
            signal_time    = bar_ts,
            r_ratio        = rr,
            atr_regime     = regime if strat_cfg.use_atr else "",
            ml_mult        = ml_mult,
            candle_close   = est_entry,
        )

        # Notify Telegram
        notified = False
        if TELEGRAM.enabled:
            notified = self.tg.send_signal(sig_msg)

        # Log signal
        self.state.set_last_signal_ts(strat_cfg.timeframe, bar_ts.timestamp())
        self.state.incr_signals()
        self.sig_log.log({
            "ts":         bar_ts.isoformat(),
            "strategy":   strat_cfg.name,
            "timeframe":  strat_cfg.timeframe,
            "direction":  side_str,
            "entry":      est_entry,
            "sl":         sl,
            "tp":         tp,
            "tp_partial": tp_partial,
            "risk_pct":   eff_risk,
            "ml_mult":    ml_mult,
            "atr_regime": regime,
            "notified":   notified,
            "executed":   False,
        })

        # Execute trade if enabled
        if MODE.auto_trade and not MODE.signals_only:
            self._execute_signal(strat_cfg, sig_msg, side, sl, tp, tp_partial,
                                 partial_pct or 0.0, eff_risk)

    def _execute_signal(
        self,
        strat_cfg:   StrategyConfig,
        sig:         SignalMessage,
        side:        int,
        sl:          float,
        tp:          float,
        tp_partial:  float,
        partial_pct: float,
        eff_risk:    float,
    ) -> None:
        """Execute a market order for the signal."""
        # Check position limit
        open_pos = self.executor.get_open_positions()
        if len(open_pos) >= MODE.max_open_positions:
            logger.warning(f"Max positions ({MODE.max_open_positions}) reached — skipping")
            return

        # Check for existing position on this strategy/TF
        existing = [p for p in open_pos if p.timeframe == strat_cfg.timeframe
                    and p.strategy == strat_cfg.name]
        if existing:
            logger.warning(f"Position already open for {strat_cfg.label} — skipping")
            return

        capital  = self.executor.get_account_balance()
        side_obj = OrderSide.BUY if side == 1 else OrderSide.SELL
        quantity = self.executor.compute_quantity(capital, eff_risk, sig.est_entry, sl)

        if quantity <= 0:
            logger.warning("Computed quantity=0, skipping")
            return

        order = Order(
            strategy    = strat_cfg.name,
            timeframe   = strat_cfg.timeframe,
            side        = side_obj,
            quantity    = quantity,
            entry_price = sig.est_entry,
            sl_price    = sl,
            tp_price    = tp,
            tp_partial  = tp_partial,
            partial_pct = partial_pct,
        )

        logger.info(f"  Executing: {strat_cfg.label} {side_obj.value}  "
                    f"qty={quantity:.6f}  entry≈{sig.est_entry:.1f}")
        filled = self.executor.open_position(order)

        if filled.status == OrderStatus.OPEN:
            upd = TradeUpdate(
                strategy_label = strat_cfg.label,
                event          = "OPENED",
                direction      = sig.direction,
                entry_price    = filled.entry_price,
                current_price  = filled.entry_price,
                pnl_pct        = 0.0,
                pnl_usd        = 0.0,
                timeframe      = strat_cfg.timeframe,
            )
            self.tg.send_trade_update(upd)
        else:
            logger.error(f"  Order failed: {filled.status}")
            self.tg.send_alert("ERROR", f"Order failed for {strat_cfg.label}: {filled.status}")

    def _monitor_positions(self) -> None:
        """Check open positions for partial TP."""
        if not MODE.auto_trade:
            return
        open_pos = self.executor.get_open_positions()
        if not open_pos:
            return

        price = self.executor.get_current_price()
        if price <= 0:
            return

        for pos in open_pos:
            pos.update_pnl(price)

            # Partial TP check
            if pos.tp_partial > 0 and not pos.partial_done:
                triggered = (pos.side == OrderSide.BUY  and price >= pos.tp_partial) or \
                            (pos.side == OrderSide.SELL and price <= pos.tp_partial)
                if triggered:
                    frac = pos.partial_pct or 0.5
                    pnl  = self.executor.close_partial(pos, frac)
                    self.state.add_daily_pnl(pnl)
                    upd  = TradeUpdate(
                        strategy_label = f"{pos.strategy}/{pos.timeframe}",
                        event          = "PARTIAL_CLOSE",
                        direction      = pos.side.value,
                        entry_price    = pos.entry_price,
                        current_price  = price,
                        pnl_pct        = pnl / (pos.entry_price * pos.quantity + 1e-9),
                        pnl_usd        = pnl,
                        timeframe      = pos.timeframe,
                    )
                    self.tg.send_trade_update(upd)

    def _get_balance(self) -> float:
        try:
            return self.executor.get_account_balance()
        except Exception:
            return 10_000.0