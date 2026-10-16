#!/usr/bin/env python3
"""Run the trading engine standalone (systemd / CLI)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    from wtb.bot.engine import TradingEngine
    from wtb.core.config import get_settings

    engine = TradingEngine(get_settings())
    try:
        asyncio.run(engine.run_forever())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
