"""Redis access: event streams, pub/sub, distributed locks, live bot state.

Streams (event bus — decouples ingestion from inference):
  wtb:ohlcv:{symbol}:{timeframe}   candle events
  wtb:signals:{symbol}             fused signals
Keys:
  wtb:status                       hash with live bot state (dashboard reads)
  wtb:lock:{scope}                 SET NX distributed locks (position opening)
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import redis.asyncio as aioredis

from wtb.core.logging import get_logger

log = get_logger(__name__)

STATUS_KEY = "wtb:status"


def ohlcv_stream(symbol: str, timeframe: str) -> str:
    return f"wtb:ohlcv:{symbol.upper()}:{timeframe.lower()}"


def signal_stream(symbol: str) -> str:
    return f"wtb:signals:{symbol.upper()}"


def lock_key(scope: str) -> str:
    return f"wtb:lock:{scope}"


class RedisClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self._client: aioredis.Redis | None = None

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisClient not connected — call connect() first")
        return self._client

    async def connect(self) -> None:
        if self._client is None:
            self._client = aioredis.from_url(self.url, decode_responses=True)
            await self.client.ping()
            log.info("redis_connected", url=self.url)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- event bus ----------------------------------------------------------- #
    async def publish_candle(self, symbol: str, timeframe: str, payload: dict[str, Any]) -> None:
        await self.client.xadd(
            ohlcv_stream(symbol, timeframe),
            {"data": json.dumps(payload, default=str)},
            maxlen=10_000, approximate=True,
        )

    async def publish_signal(self, symbol: str, payload: dict[str, Any]) -> None:
        await self.client.xadd(
            signal_stream(symbol),
            {"data": json.dumps(payload, default=str)},
            maxlen=1_000, approximate=True,
        )

    # -- live state (dashboard reads this instead of bot_status.json) --------- #
    async def set_status(self, status: dict[str, Any]) -> None:
        await self.client.set(STATUS_KEY, json.dumps(status, default=str))

    async def get_status(self) -> dict[str, Any]:
        raw = await self.client.get(STATUS_KEY)
        return json.loads(raw) if raw else {}

    # -- distributed lock: prevents double position opening ------------------- #
    @asynccontextmanager
    async def acquire_lock(self, scope: str, ttl_seconds: float = 30.0) -> AsyncIterator[bool]:
        token = f"lock-owner-{id(self)}"
        key = lock_key(scope)
        got = await self.client.set(key, token, nx=True, ex=int(max(1, ttl_seconds)))
        try:
            yield bool(got)
        finally:
            if got:
                current = await self.client.get(key)
                if current == token:
                    await self.client.delete(key)
