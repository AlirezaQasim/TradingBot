"""TimescaleDB access via asyncpg: batch upsert of OHLCV, predictions, book snapshots."""
from __future__ import annotations

from datetime import datetime, timezone

import asyncpg

from wtb.core.logging import get_logger
from wtb.db.timescale_schema import TIMESCALE_DDL
from wtb.data.contracts import Candle

log = get_logger(__name__)


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class TimescaleStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=5)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def migrate(self) -> None:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            for stmt in TIMESCALE_DDL:
                await conn.execute(stmt)
        log.info("timescale_migrated")

    async def upsert_candles(self, candles: list[Candle]) -> int:
        """Idempotent insert (ON CONFLICT DO UPDATE on the hypertable identity)."""
        if not candles:
            return 0
        assert self._pool is not None
        rows = [
            (_ms_to_dt(c.ts), c.symbol, c.timeframe, c.open, c.high, c.low, c.close, c.volume)
            for c in candles
        ]
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT DO NOTHING
                """,
                rows,
            )
        return len(rows)

    async def insert_prediction(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
        model_version: str,
        prob: float | None,
        signal: str | None,
        features: dict | None,
    ) -> None:
        assert self._pool is not None
        import json
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO predictions (time, symbol, timeframe, model_name, model_version, prob, signal, features)
                VALUES (now(), $1, $2, $3, $4, $5, $6, $7)
                """,
                symbol, timeframe, model_name, model_version, prob, signal,
                json.dumps(features) if features else None,
            )

    async def insert_orderbook(self, symbol: str, bids: list[tuple], asks: list[tuple]) -> None:
        assert self._pool is not None
        import json
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO orderbook_snapshots (time, symbol, bids, asks)
                VALUES (now(), $1, $2, $3)
                """,
                symbol, json.dumps(bids), json.dumps(asks),
            )

    async def load_candles(self, symbol: str, timeframe: str, limit: int = 1000) -> list[Candle]:
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT time, open, high, low, close, volume
                FROM ohlcv WHERE symbol = $1 AND timeframe = $2
                ORDER BY time DESC LIMIT $3
                """,
                symbol, timeframe, limit,
            )
        candles = [
            Candle(
                ts=int(r["time"].timestamp() * 1000),
                open=r["open"], high=r["high"], low=r["low"],
                close=r["close"], volume=r["volume"],
                symbol=symbol, timeframe=timeframe,
            )
            for r in reversed(rows)
        ]
        return candles
