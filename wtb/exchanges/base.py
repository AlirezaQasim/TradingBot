"""Async exchange adapter interface.

Every adapter is fully async (aiohttp / ccxt.async_support), wrapped in a
per-exchange circuit breaker, and returns validated contracts. The v0.0.1
adapters were blocking `requests` calls that swallowed every error.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from wtb.core.config import ExchangeConfig
from wtb.core.resilience import AsyncCircuitBreaker, breaker_for
from wtb.data.contracts import (
    Candle,
    OrderBook,
    OrderResult,
    OrderType,
    Side,
    Ticker,
)


class ExchangeError(RuntimeError):
    """Typed exchange failure (HTTP error, bad payload, auth failure)."""


class UnsupportedSymbolError(ExchangeError):
    """The symbol does not exist on this exchange (distinct from network noise)."""


class ExchangeAdapter(ABC):
    """Common async interface implemented by every exchange."""

    def __init__(self, config: ExchangeConfig) -> None:
        self.config = config
        self.name = config.name or config.adapter
        self.breaker: AsyncCircuitBreaker = breaker_for(
            self.name,
            fail_max=config.breaker_fail_max,
            reset_timeout=config.breaker_reset_timeout,
        )

    # -- market data ------------------------------------------------------- #
    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str = "4h", limit: int = 100
    ) -> list[Candle]: ...

    @abstractmethod
    async def fetch_order_book(self, symbol: str, depth: int = 10) -> OrderBook: ...

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Ticker: ...

    @abstractmethod
    async def validate_symbol(self, symbol: str) -> bool: ...

    # -- trading ----------------------------------------------------------- #
    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool: ...

    @abstractmethod
    async def fetch_open_orders(self, symbol: str | None = None) -> list[OrderResult]: ...

    # -- lifecycle ---------------------------------------------------------- #
    async def close(self) -> None:  # optional override
        return None

    async def __aenter__(self) -> "ExchangeAdapter":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
