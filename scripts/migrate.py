#!/usr/bin/env python3
"""One-shot migration: TimescaleDB hypertables + PostgreSQL app schema."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


async def main() -> None:
    from wtb.core.config import get_settings
    from wtb.db.postgres import Engine
    from wtb.db.timescale import TimescaleStore

    s = get_settings()
    ts = TimescaleStore(s.database.timescale_dsn)
    await ts.connect()
    await ts.migrate()
    await ts.close()
    print("wtb_ts migrated (hypertables ready)")

    pg = Engine(s.database.app_dsn)
    await pg.migrate()
    await pg.close()
    print("wtb_app migrated (ORM tables ready)")


if __name__ == "__main__":
    asyncio.run(main())
