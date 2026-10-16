"""Async data collector: fetch -> validate -> dedupe -> persist -> publish.

Replaces v0.0.1's O(n^2) deque scan and per-fetch full-CSV rewrite with an
O(1) timestamp set, idempotent TimescaleDB upserts, and Redis Streams events.
A small CSV snapshot is still written (Parquet-ready) as an offline fallback.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import polars as pl

from wtb.core.logging import get_logger
from wtb.data.contracts import Candle
from wtb.data.indicators import add_indicators, candles_to_frame
from wtb.db.redis_client import RedisClient
from wtb.db.timescale import TimescaleStore
from wtb.exchanges.base import ExchangeAdapter

log = get_logger(__name__)


class Collector:
    def __init__(
        self,
        exchange: ExchangeAdapter,
        symbol: str,
        timeframe: str,
        data_dir: Path,
        max_points: int = 10_000,
        timescale: TimescaleStore | None = None,
        redis: RedisClient | None = None,
        indicators: "IndicatorsConfig | None" = None,
        save_csv_snapshot: bool = True,
        warmup_pad: int = 10,
    ) -> None:
        from wtb.core.config import IndicatorsConfig
        self.exchange = exchange
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_points = max_points
        self.timescale = timescale
        self.redis = redis
        self.indicators = indicators or IndicatorsConfig()
        self.save_csv_snapshot = save_csv_snapshot
        self._warmup_pad = warmup_pad

        self._buffer: deque[Candle] = deque(maxlen=max_points)
        self._seen: set[int] = set()          # O(1) dedup on candle timestamps
        self._csv_path = self.data_dir / f"{self.symbol}_{self.timeframe}.csv"

    # ------------------------------------------------------------------ load #
    def load_history(self) -> int:
        """Load previous CSV history so the model has data after restarts."""
        if not self._csv_path.exists():
            return 0
        try:
            df = pl.read_csv(self._csv_path)
            loaded = 0
            for row in df.iter_rows(named=True):
                try:
                    candle = Candle.model_validate(row)
                    self._append(candle)
                    loaded += 1
                except Exception:
                    continue
            log.info("history_loaded", symbol=self.symbol, rows=loaded)
            return loaded
        except Exception as exc:
            log.warning("history_load_failed", symbol=self.symbol, error=str(exc))
            return 0

    # ----------------------------------------------------------------- fetch #
    async def fetch_new_data(self, limit: int = 100) -> list[Candle]:
        """Fetch fresh candles, dedupe, persist and publish. Returns new candles."""
        candles = await self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit)
        fresh: list[Candle] = []
        for candle in candles:
            if candle.ts not in self._seen:
                self._append(candle)
                fresh.append(candle)
        if fresh:
            await self._persist(fresh)
            log.debug("candles_fetched", symbol=self.symbol, new=len(fresh), total=len(self._buffer))
        return fresh

    def _append(self, candle: Candle) -> None:
        if candle.ts in self._seen:
            return
        if len(self._buffer) == self._buffer.maxlen:
            self._seen.discard(self._buffer[0].ts)
        self._buffer.append(candle)
        self._seen.add(candle.ts)

    # --------------------------------------------------------------- persist #
    async def _persist(self, fresh: list[Candle]) -> None:
        if self.timescale is not None:
            try:
                await self.timescale.upsert_candles(fresh)
            except Exception as exc:
                log.error("timescale_upsert_failed", error=str(exc))
        if self.redis is not None:
            try:
                for candle in fresh:
                    await self.redis.publish_candle(
                        self.symbol, self.timeframe, candle.model_dump()
                    )
            except Exception as exc:
                log.error("redis_publish_failed", error=str(exc))
        self._save_snapshot()

    def _save_snapshot(self) -> None:
        if not self.save_csv_snapshot:
            return
        try:
            self.frame().write_csv(self._csv_path)
        except Exception as exc:
            log.warning("csv_snapshot_failed", error=str(exc))

    # ----------------------------------------------------------------- views #
    def frame(self) -> pl.DataFrame:
        return candles_to_frame(list(self._buffer))

    def feature_frame(self, features: list[str]) -> pl.DataFrame:
        """Full frame with indicator columns added, ready for the model."""
        df = add_indicators(self.frame(), self.indicators)
        from wtb.data.indicators import feature_matrix
        return feature_matrix(df, features)

    def latest_price(self) -> float | None:
        return self._buffer[-1].close if self._buffer else None

    def has_enough(self, lookback: int, pad: int | None = None) -> bool:
        if pad is None:
            pad = self._warmup_pad
        return len(self._buffer) >= lookback + pad
