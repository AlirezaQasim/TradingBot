"""Celery background workers (Redis broker): retraining, backtests, reports.

Start:  celery -A wtb.workers.tasks worker --loglevel=INFO --concurrency=1
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from celery import Celery  # noqa: E402

from wtb.core.config import get_settings  # noqa: E402

_settings = get_settings()

celery_app = Celery(
    "webtraderbot",
    broker=_settings.database.redis_url,
    backend=_settings.database.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_max_tasks_per_child=20,     # Pi memory hygiene
    task_time_limit=3600,
    timezone="UTC",
)


@celery_app.task(name="wtb.retrain_model", bind=True, max_retries=2)
def retrain_model(self) -> dict:
    """Full model retrain on the latest collected history."""
    import asyncio

    from wtb.core.config import load_settings
    from wtb.db.timescale import TimescaleStore
    from wtb.exchanges.factory import get_exchange
    from wtb.data.collector import Collector
    from wtb.ml.service import ModelService

    s = load_settings()
    ex = get_exchange(s.active_exchange, s.exchange())

    async def build():
        ts = TimescaleStore(s.database.timescale_dsn)
        try:
            await ts.connect()
        except Exception:
            ts = None
        collector = Collector(ex, s.trading.symbol, s.trading.timeframe,
                              data_dir=s.app.data_dir, max_points=s.data.max_data_points,
                              timescale=ts)
        collector.load_history()
        await collector.fetch_new_data(limit=1000)
        return collector, ts

    collector, ts = asyncio.run(build())
    try:
        service = ModelService(s.ai, s.trading.symbol, s.trading.timeframe, s.app.model_dir)
        metrics = service.train(collector)
        result = {
            "status": "ok", "artifact": service.key,
            "loss": metrics.get("loss"), "val_acc": metrics.get("val_acc"),
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
    finally:
        asyncio.run(ex.close())
        if ts is not None:
            asyncio.run(ts.close())
    return result


@celery_app.task(name="wtb.daily_report")
def daily_report() -> dict:
    """Daily PnL / risk summary from the app database, pushed to Telegram."""
    import asyncio

    from sqlalchemy import select

    from wtb.db.postgres import Engine, Position

    async def run():
        s = get_settings()
        pg = Engine(s.database.app_dsn)
        out: dict = {"closed": 0, "pnl": 0.0, "wins": 0}
        async with pg.session_factory() as session:
            rows = await session.execute(
                select(Position).where(Position.state == "closed")
                .order_by(Position.closed_at.desc()).limit(200)
            )
            for pos in rows.scalars():
                if pos.pnl is not None:
                    out["closed"] += 1
                    out["pnl"] += pos.pnl
                    out["wins"] += 1 if pos.pnl > 0 else 0
        await pg.close()
        return out

    stats = asyncio.run(run())
    win_rate = (stats["wins"] / stats["closed"] * 100) if stats["closed"] else 0.0
    text = (f"📊 <b>Daily report</b>\nTrades closed: {stats['closed']}\n"
            f"Total PnL: {stats['pnl']:+.4f}\nWin rate: {win_rate:.1f}%")
    try:
        import aiohttp

        async def send():
            n = get_settings().notifications
            if not (n.telegram_bot_token and n.telegram_chat_id):
                return
            url = f"https://api.telegram.org/bot{n.telegram_bot_token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={"chat_id": n.telegram_chat_id,
                                              "text": text, "parse_mode": "HTML"},
                                   timeout=aiohttp.ClientTimeout(total=10))

        asyncio.run(send())
    except Exception:
        pass
    return stats
