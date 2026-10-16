"""Wallex async adapter (aiohttp).

Public API (api.wallex.ir):
- GET /v1/udf/history        OHLCV (TradingView UDF)
- GET /v1/depth              order book
- GET /v1/markets            market stats (symbol validation, last price)
- GET /v1/account/orders     open orders (auth)
- POST /v1/account/orders    place order (auth)
- DELETE /v1/account/orders  cancel (auth)
Auth: ``X-WALLEX-API-KEY`` header.

Wallex quotes toman markets as ``*TMAN``; the adapter transparently maps the
config's ``*IRT`` aliases. The v0.0.1 adapter returned ``[]`` for OHLCV and
mock order ids — trading was impossible.
"""
from __future__ import annotations

import time
from typing import Any

import aiohttp

from wtb.core.config import ExchangeConfig
from wtb.core.logging import get_logger
from wtb.core.resilience import retry_network
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

RESOLUTION = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
TF_SECONDS = {"1": 60, "5": 300, "15": 900, "30": 1800, "60": 3600, "240": 14400, "D": 86400}


def to_wallex_symbol(symbol: str) -> str:
    """BTCIRT -> BTCTMN (Wallex uses TMAN as the toman quote)."""
    s = symbol.upper()
    if s.endswith("IRT"):
        return s[:-3] + "TMAN"
    return s


class WallexAdapter(ExchangeAdapter):
    def __init__(self, config: ExchangeConfig) -> None:
        super().__init__(config)
        self._base = (config.base_url or "https://api.wallex.ir").rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"X-WALLEX-API-KEY": self.config.api_key} if self.config.api_key else {}
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                headers=headers,
            )
        return self._session

    @retry_network(attempts=3)
    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        session = await self._get_session()
        async with session.request(method, f"{self._base}{path}", **kwargs) as resp:
            body = await resp.json(content_type=None)
            if resp.status == 404:
                raise UnsupportedSymbolError(f"Wallex 404 on {path}")
            if resp.status in (401, 403):
                raise ExchangeError(f"Wallex auth failed ({resp.status}) on {path}")
            if resp.status >= 500:
                raise ConnectionError(f"Wallex {resp.status} on {path}")
            if not isinstance(body, dict) or not body.get("success", True):
                raise ExchangeError(f"Wallex error on {path}: {body.get('message', body)}")
            return body

    # -- market data ------------------------------------------------------- #
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 100) -> list[Candle]:
        resolution = RESOLUTION.get(timeframe.lower())
        if resolution is None:
            raise ExchangeError(f"Unsupported timeframe '{timeframe}' for Wallex")
        tf_sec = TF_SECONDS[resolution]
        to_ts = int(time.time())
        from_ts = to_ts - limit * tf_sec

        payload = await self.breaker.call_async(
            self._request, "GET", "/v1/udf/history",
            params={"symbol": to_wallex_symbol(symbol), "resolution": resolution,
                    "from": str(from_ts), "to": str(to_ts)},
        )
        result = payload.get("result", {})
        if result.get("s") != "ok" or not result.get("t"):
            return []

        out: list[Candle] = []
        for i, ts in enumerate(result["t"]):
            try:
                out.append(Candle(
                    ts=int(ts) * 1000,
                    open=float(result["o"][i]), high=float(result["h"][i]),
                    low=float(result["l"][i]), close=float(result["c"][i]),
                    volume=float(result["v"][i]),
                    symbol=symbol.upper(), timeframe=timeframe.lower(),
                ))
            except Exception as exc:
                log.warning("bad_candle_skipped", exchange="wallex", symbol=symbol, error=str(exc))
        out.sort(key=lambda c: c.ts)
        return out

    async def fetch_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        payload = await self.breaker.call_async(
            self._request, "GET", "/v1/depth", params={"symbol": to_wallex_symbol(symbol)}
        )
        result = payload.get("result", {})
        bids = [OrderBookLevel(price=float(p), quantity=float(q)) for p, q in result.get("bid", [])][:depth]
        asks = [OrderBookLevel(price=float(p), quantity=float(q)) for p, q in result.get("ask", [])][:depth]
        if not bids or not asks:
            raise ExchangeError(f"Wallex empty order book for {symbol}")
        return OrderBook(symbol=symbol.upper(), bids=bids, asks=asks)

    async def fetch_ticker(self, symbol: str) -> Ticker:
        payload = await self.breaker.call_async(self._request, "GET", "/v1/markets")
        stats = payload.get("result", {}).get("markets", {})
        info = stats.get(to_wallex_symbol(symbol))
        if not info:
            raise UnsupportedSymbolError(f"Wallex has no market '{symbol}'")
        stats = info.get("stats", info)
        return Ticker(
            symbol=symbol.upper(),
            last=float(stats.get("lastPrice", 0) or 0) or float(info.get("lastPrice", 0)),
            bid=float(stats["bestBuy"]) if stats.get("bestBuy") else None,
            ask=float(stats["bestSell"]) if stats.get("bestSell") else None,
            high=float(stats["dayHigh"]) if stats.get("dayHigh") else None,
            low=float(stats["dayLow"]) if stats.get("dayLow") else None,
            volume=float(stats["volumeSrc"]) if stats.get("volumeSrc") else None,
        )

    async def validate_symbol(self, symbol: str) -> bool:
        try:
            await self.fetch_ticker(symbol)
            return True
        except UnsupportedSymbolError:
            return False
        except Exception as exc:
            log.warning("symbol_validation_error", exchange="wallex", symbol=symbol, error=str(exc))
            return False

    # -- trading ----------------------------------------------------------- #
    async def create_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: float,
        price: float | None = None,
        client_order_id: str | None = None,
    ) -> OrderResult:
        body: dict[str, Any] = {
            "symbol": to_wallex_symbol(symbol),
            "type": order_type.value.upper(),   # LIMIT | MARKET
            "side": side.value.upper(),         # BUY | SELL
            "quantity": f"{quantity:.10f}".rstrip("0").rstrip("."),
        }
        if order_type == OrderType.LIMIT:
            if price is None:
                raise ExchangeError("Limit order requires a price")
            body["price"] = f"{price:.10f}".rstrip("0").rstrip(".")
        if client_order_id:
            body["clientOrderId"] = client_order_id

        payload = await self.breaker.call_async(self._request, "POST", "/v1/account/orders", json=body)
        order = payload.get("result", {}).get("order", payload.get("result", {}))
        return OrderResult(
            order_id=str(order.get("id", order.get("clientOrderId", ""))),
            client_order_id=client_order_id or order.get("clientOrderId"),
            symbol=symbol.upper(), side=side, order_type=order_type,
            quantity=quantity, price=float(order["price"]) if order.get("price") else price,
            status=str(order.get("status", "open")),
            raw=order,
        )

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        payload = await self.breaker.call_async(
            self._request, "DELETE", "/v1/account/orders",
            params={"clientOrderId": order_id},
        )
        return payload.get("success", False)

    async def fetch_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        params: dict[str, Any] = {"status": "OPEN"}
        if symbol:
            params["symbol"] = to_wallex_symbol(symbol)
        payload = await self.breaker.call_async(
            self._request, "GET", "/v1/account/orders", params=params
        )
        out: list[OrderResult] = []
        for o in payload.get("result", {}).get("orders", []):
            try:
                out.append(OrderResult(
                    order_id=str(o.get("clientOrderId", o.get("id", ""))),
                    client_order_id=o.get("clientOrderId"),
                    symbol=str(o.get("symbol", "")).upper().replace("TMAN", "IRT"),
                    side=Side(str(o.get("side", "BUY")).lower()),
                    order_type=OrderType(str(o.get("type", "LIMIT")).lower()),
                    quantity=float(o.get("origQty", 0) or 0),
                    price=float(o["price"]) if o.get("price") else None,
                    status=str(o.get("status", "open")),
                    raw=o,
                ))
            except Exception:
                continue
        return out

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
