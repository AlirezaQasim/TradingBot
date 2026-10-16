"""Generic ccxt adapter using ``ccxt.async_support`` (Binance, Bybit, ...)."""
from __future__ import annotations

import asyncio

import ccxt.async_support as ccxt_async

from wtb.core.config import ExchangeConfig
from wtb.core.logging import get_logger
from wtb.data.contracts import (
    Candle,
    OrderBook,
    OrderBookLevel,
    OrderResult,
    OrderType,
    Side,
    Ticker,
)
from wtb.exchanges.base import ExchangeAdapter, ExchangeError, UnsupportedSymbolError

log = get_logger(__name__)

CCXT_TF = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}


class CcxtAdapter(ExchangeAdapter):
    def __init__(self, config: ExchangeConfig, exchange_id: str) -> None:
        super().__init__(config)
        exchange_id = exchange_id.lower()
        if not hasattr(ccxt_async, exchange_id):
            raise ExchangeError(f"Unknown ccxt exchange id: '{exchange_id}'")
        klass = getattr(ccxt_async, exchange_id)
        self.client = klass({
            "apiKey": config.api_key or None,
            "secret": config.api_secret or None,
            "enableRateLimit": True,
            "timeout": int(config.timeout_seconds * 1000),
            "options": {"defaultType": "spot"},
        })
        if config.testnet:
            self.client.set_sandbox_mode(True)

    async def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 100) -> list[Candle]:
        tf = CCXT_TF.get(timeframe.lower())
        if tf is None:
            raise ExchangeError(f"Unsupported timeframe '{timeframe}' for ccxt")
        rows = await self.breaker.call_async(self.client.fetch_ohlcv, symbol, tf, None, limit)
        out: list[Candle] = []
        for row in rows or []:
            try:
                out.append(Candle(
                    ts=int(row[0]), open=float(row[1]), high=float(row[2]),
                    low=float(row[3]), close=float(row[4]), volume=float(row[5]),
                    symbol=symbol.upper(), timeframe=timeframe.lower(),
                ))
            except Exception as exc:
                log.warning("bad_candle_skipped", exchange=self.name, symbol=symbol, error=str(exc))
        return out

    async def fetch_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        raw = await self.breaker.call_async(self.client.fetch_order_book, symbol, depth)
        bids = [OrderBookLevel(price=float(p), quantity=float(q)) for p, q in raw.get("bids", [])][:depth]
        asks = [OrderBookLevel(price=float(p), quantity=float(q)) for p, q in raw.get("asks", [])][:depth]
        if not bids or not asks:
            raise ExchangeError(f"{self.name}: empty order book for {symbol}")
        return OrderBook(symbol=symbol.upper(), bids=bids, asks=asks,
                         ts=int(raw.get("timestamp") or 0) or None)  # type: ignore[arg-type]

    async def fetch_ticker(self, symbol: str) -> Ticker:
        raw = await self.breaker.call_async(self.client.fetch_ticker, symbol)
        last = raw.get("last") or raw.get("close")
        if not last:
            raise ExchangeError(f"{self.name}: ticker for {symbol} has no last price")
        return Ticker(
            symbol=symbol.upper(),
            last=float(last),
            bid=float(raw["bid"]) if raw.get("bid") else None,
            ask=float(raw["ask"]) if raw.get("ask") else None,
            high=float(raw["high"]) if raw.get("high") else None,
            low=float(raw["low"]) if raw.get("low") else None,
            volume=float(raw["baseVolume"]) if raw.get("baseVolume") else None,
        )

    async def validate_symbol(self, symbol: str) -> bool:
        try:
            await self.client.load_markets()
            return symbol in self.client.markets
        except Exception as exc:
            log.warning("symbol_validation_error", exchange=self.name, symbol=symbol, error=str(exc))
            return False

    async def create_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        params: dict = {}
        if client_order_id:
            params["clientOrderId"] = client_order_id
        raw = await self.breaker.call_async(
            self.client.create_order, symbol, order_type.value, side.value, quantity, price, params
        )
        return OrderResult(
            order_id=str(raw.get("id", "")),
            client_order_id=client_order_id or raw.get("clientOrderId"),
            symbol=symbol.upper(), side=side, order_type=order_type,
            quantity=float(raw.get("amount", quantity)),
            price=float(raw["price"]) if raw.get("price") else price,
            status=str(raw.get("status", "open")),
            filled_quantity=float(raw.get("filled", 0) or 0),
            raw=raw,
        )

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        try:
            await self.breaker.call_async(self.client.cancel_order, order_id, symbol)
            return True
        except Exception as exc:
            log.warning("cancel_failed", exchange=self.name, order_id=order_id, error=str(exc))
            return False

    async def fetch_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        raws = await self.breaker.call_async(self.client.fetch_open_orders, symbol)
        out: list[OrderResult] = []
        for raw in raws or []:
            try:
                out.append(OrderResult(
                    order_id=str(raw.get("id", "")),
                    client_order_id=raw.get("clientOrderId"),
                    symbol=str(raw.get("symbol", "")).replace("/", "").upper(),
                    side=Side(str(raw.get("side", "buy"))),
                    order_type=OrderType(str(raw.get("type", "limit")).split(":")[0]),
                    quantity=float(raw.get("amount", 0) or 0),
                    price=float(raw["price"]) if raw.get("price") else None,
                    status=str(raw.get("status", "open")),
                    filled_quantity=float(raw.get("filled", 0) or 0),
                    raw=raw,
                ))
            except Exception:
                continue
        return out

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:  # pragma: no cover
            pass
        finally:
            if not self.client.session.closed:
                await asyncio.gather(self.client.session.close(), return_exceptions=True)
