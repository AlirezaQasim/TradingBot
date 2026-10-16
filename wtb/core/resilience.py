"""Fault-tolerance primitives: retries with backoff + per-exchange circuit breakers.

Every outbound network call in the bot goes through one of these decorators so a
flapping exchange API degrades gracefully instead of killing the loop.
"""
from __future__ import annotations

import functools
from typing import Awaitable, Callable, TypeVar

import pybreaker
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from wtb.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class ExchangeUnavailableError(RuntimeError):
    """Raised when an exchange endpoint keeps failing (breaker open / retries spent)."""


def retry_network(attempts: int = 3, min_wait: float = 1.0, max_wait: float = 10.0):
    """Exponential-backoff retry for transient network failures."""
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )


class AsyncCircuitBreaker(pybreaker.CircuitBreaker):
    """Circuit breaker that works with async callables.

    pybreaker 1.x supports coroutines natively; this subclass only adds
    structured logging on state transitions.
    """

    def __init__(self, name: str, fail_max: int = 5, reset_timeout: float = 60.0):
        super().__init__(fail_max=fail_max, reset_timeout=reset_timeout, name=name)
        self._log = get_logger(f"breaker.{name}")

    async def call_async(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        try:
            return await super().call_async(func, *args, **kwargs)
        except pybreaker.CircuitBreakerError:
            self._log.warning("circuit_open", exchange=self.name)
            raise ExchangeUnavailableError(
                f"Circuit breaker '{self.name}' is open — exchange assumed down"
            ) from None

    def __call__(self, func):
        if asyncio_is_coroutine(func):
            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
            return awrapper
        return super().__call__(func)


def asyncio_is_coroutine(func) -> bool:
    import inspect
    return inspect.iscoroutinefunction(func)


def breaker_for(exchange_name: str, fail_max: int = 5, reset_timeout: float = 60.0) -> AsyncCircuitBreaker:
    """Factory so each exchange gets its own isolated breaker."""
    return AsyncCircuitBreaker(name=exchange_name, fail_max=fail_max, reset_timeout=reset_timeout)
