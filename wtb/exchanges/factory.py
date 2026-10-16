"""Exchange adapter factory."""
from __future__ import annotations

from wtb.core.config import ExchangeConfig
from wtb.exchanges.base import ExchangeAdapter
from wtb.exchanges.nobitex import NobitexAdapter
from wtb.exchanges.wallex import WallexAdapter


def get_exchange(exchange_key: str, config: ExchangeConfig) -> ExchangeAdapter:
    """Build the async adapter for ``exchange_key`` (config section name)."""
    adapter = config.adapter.lower()
    if adapter == "nobitex":
        return NobitexAdapter(config)
    if adapter == "wallex":
        return WallexAdapter(config)
    if adapter == "ccxt":
        from wtb.exchanges.ccxt_async import CcxtAdapter
        return CcxtAdapter(config, exchange_id=exchange_key)
    raise ValueError(f"Unknown adapter type '{adapter}' for exchange '{exchange_key}'")
