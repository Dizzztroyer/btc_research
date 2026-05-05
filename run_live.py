#!/usr/bin/env python3
"""
run_live.py
────────────
Entry point for the live trading signal system.

Modes:
    signals-only  — generate signals and send Telegram notifications (default)
    paper         — signals + paper trade execution (no real orders)
    live          — signals + real execution via Binance API

Setup:
    1. Copy .env.example to .env and fill in your tokens
    2. Set mode in live/config.py
    3. Run: python run_live.py

Safety:
    - Starts in signals-only mode by default
    - paper_mode=True by default (no real orders even if auto_trade=True)
    - Binance testnet=True by default

Usage:
    python run_live.py                    # signals only (default)
    python run_live.py --mode paper       # paper trading
    python run_live.py --mode live        # live trading (careful!)
    python run_live.py --test-telegram    # test Telegram connection
    python run_live.py --once             # run one tick and exit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BTC Research Live Trading System")
    p.add_argument("--mode", choices=["signals", "paper", "live"],
                   default="signals", help="Trading mode")
    p.add_argument("--test-telegram", action="store_true",
                   help="Test Telegram connection and exit")
    p.add_argument("--once",          action="store_true",
                   help="Run one tick and exit (for testing)")
    p.add_argument("--config",        default="config/config.yaml")
    return p.parse_args()


def load_dotenv() -> None:
    """Load .env file if present (minimal implementation, no dependency)."""
    env_file = Path(".env")
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
    print("Loaded .env")


def main() -> None:
    args = parse_args()
    load_dotenv()

    from live.config import MODE, TELEGRAM, print_config
    from live.notifications.telegram import TelegramNotifier

    # Apply mode
    if args.mode == "signals":
        MODE.signals_only = True
        MODE.auto_trade   = False
        MODE.paper_mode   = True
        print("Mode: SIGNALS ONLY (Telegram notifications, no orders)")

    elif args.mode == "paper":
        MODE.signals_only = False
        MODE.auto_trade   = True
        MODE.paper_mode   = True
        print("Mode: PAPER TRADING (simulate orders, no real execution)")

    elif args.mode == "live":
        MODE.signals_only = False
        MODE.auto_trade   = True
        MODE.paper_mode   = False
        print("⚠️  Mode: LIVE TRADING (real orders will be sent!)")
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm.strip() != "CONFIRM":
            print("Aborted.")
            sys.exit(0)

    # Test Telegram
    if args.test_telegram:
        tg = TelegramNotifier(
            token    = TELEGRAM.bot_token,
            chat_id  = TELEGRAM.chat_id,
        )
        ok = tg.test()
        print(f"Telegram test: {'✓ OK' if ok else '✗ FAILED'}")
        sys.exit(0 if ok else 1)

    # Run engine
    from live.engine import SignalEngine
    engine = SignalEngine()

    if args.once:
        # Single tick for testing
        from datetime import datetime, timezone
        engine.startup()
        engine._tick()
        print("Single tick complete.")
    else:
        engine.run()


if __name__ == "__main__":
    main()