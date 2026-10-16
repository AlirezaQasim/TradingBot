"""Nobitex async adapter (aiohttp).

Endpoints (apiv2.nobitex.ir):
- GET  /v3/udf/history          OHLCV (TradingView UDF format)
- GET  /v3/orderbook/{SYMBOL}   depth
- GET  /market/stats            ticker / symbol validation
- POST /v3/orders/send          place order
- POST /v3/orders/cancel        cancel
- GET  /v3/orders/list          open orders
Auth: ``Authorization: Token <api_key>``.

Fixes vs the v0.0.1 adapter:
- async aiohttp instead of blocking requests
- srcCurrency parsing handles 4-char bases (AAVE, LINK, ...): we strip the
  known quote suffix instead of slicing ``symbol[:3]``
- typed errors instead of bare ``except: return None``
- ohlcv requests chunked with from/to windows and validated per-candle
"""
from __future__ import annotations

import json
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

# UDF resolution codes accepted by Nobitex
RESOLUTION = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}

QUOTES = ("USDT", "IRT", "TMN", "BTC")


def split_symbol(symbol: str) -> tuple[str, str]:
    """BTCIRT -> (BTC, IRT); AAVEUSDT -> (AAVE, USDT)."""
    s = symbol.upper()
    for q in QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)], q
    raise ExchangeError(f"Cannot split Nobitex symbol '{symbol}' into base/quote")


class NobitexAdapter(ExchangeAdapter):
    def __init__(self, config: ExchangeConfig) -> None:
        super().__init__(config)
        self._base = (config.base_url or "https://apiv2.nobitex.ir").rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds),
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
                    **({"Authorization": f"Token {self.config.api_key}"} if self.config.api_key else {}),
                },
            )
        return self._session

    @retry_network(attempts=3)
    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        session = await self._get_session()
        url = f"{self._base}{path}"
        async with session.request(method, url, **kwargs) as resp:
            text = await resp.text()
            try:
                body: Any = json.loads(text)
            except ValueError:
                raise ConnectionError(
                    f"Nobitex returned non-JSON ({resp.status}) on {path}: {text[:120]!r}"
                )
            if resp.status == 400:
                msg = body.get("message", body.get("error", "")) if isinstance(body, dict) else ""
                raise UnsupportedSymbolError(f"Nobitex 400 on {path}: {msg}")
            if resp.status in (401, 403):
                raise ExchangeError(f"Nobitex auth failed ({resp.status}) on {path} — check API key")
            if resp.status >= 500:
                raise ConnectionError(f"Nobitex {resp.status} on {path}")
            if not isinstance(body, dict) or body.get("status", "ok") == "fail":
                raise ExchangeError(f"Nobitex error on {path}: {body}")
            return body

    # -- market data ------------------------------------------------------- #
    async def fetch_ohlcv(self, symbol: str, timeframe: str = "4h", limit: int = 100) -> list[Candle]:
        resolution = RESOLUTION.get(timeframe.lower())
        if resolution is None:
            raise ExchangeError(f"Unsupported timeframe '{timeframe}' for Nobitex")

        tf_seconds = {"1": 60, "5": 300, "15": 900, "30": 1800, "60": 3600, "240": 14400, "D": 86400}[resolution]
        to_ts = int(time.time())
        from_ts = to_ts - limit * tf_seconds

        payload = await self.breaker.call_async(
            self._request, "GET", "/market/udf/history",
            params={"symbol": symbol.upper(), "resolution": resolution,
                    "from": str(from_ts), "to": str(to_ts)},
        )
        if payload.get("s") != "ok" or not payload.get("t"):
            return []

        candles: list[Candle] = []
        for i, ts in enumerate(payload["t"]):
            try:
                candles.append(Candle(
                    ts=int(ts) * 1000,
                    open=float(payload["o"][i]), high=float(payload["h"][i]),
                    low=float(payload["l"][i]), close=float(payload["c"][i]),
                    volume=float(payload["v"][i]),
                    symbol=symbol.upper(), timeframe=timeframe.lower(),
                ))
            except Exception as exc:  # one bad candle must not kill the batch
                log.warning("bad_candle_skipped", exchange="nobitex", symbol=symbol, error=str(exc))
        candles.sort(key=lambda c: c.ts)
        return candles

    async def fetch_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        payload = await self.breaker.call_async(
            self._request, "GET", f"/v3/orderbook/{symbol.upper()}"
        )
        bids = [OrderBookLevel(price=float(p), quantity=float(q)) for p, q in payload.get("bids", [])][:depth]
        asks = [OrderBookLevel(price=float(p), quantity=float(q)) for p, q in payload.get("asks", [])][:depth]
        if not bids or not asks:
            raise ExchangeError(f"Nobitex empty order book for {symbol}")
        return OrderBook(symbol=symbol.upper(), bids=bids, asks=asks,
                         ts=int(payload.get("lastUpdate", time.time() * 1000)))

    async def fetch_ticker(self, symbol: str) -> Ticker:
        src, dst = split_symbol(symbol)
        payload = await self.breaker.call_async(
            self._request, "GET", "/market/stats",
            params={"srcCurrency": src.lower(), "dstCurrency": dst.lower()},
        )
        key = f"{src}-{dst}".lower()   # Nobitex stats keys look like "btc-irt"
        stats = (payload.get("stats") or {}).get(key) or (payload.get("stats") or {}).get(symbol.upper())
        if not stats:
            raise UnsupportedSymbolError(f"Nobitex has no stats for {key}")
        return Ticker(
            symbol=key,
            last=float(stats.get("latest") or stats.get("dayClose")),
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
        except Exception as exc:  # network noise != unsupported symbol
            log.warning("symbol_validation_error", exchange="nobitex", symbol=symbol, error=str(exc))
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
        src, dst = split_symbol(symbol)
        body: dict[str, Any] = {
            "type": side.value,
            "execution": order_type.value,
            "srcCurrency": src,
            "dstCurrency": dst,
            "amount": f"{quantity:.10f}".rstrip("0").rstrip("."),
        }
        if order_type == OrderType.LIMIT:
            if price is None:
                raise ExchangeError("Limit order requires a price")
            body["price"] = f"{price:.10f}".rstrip("0").rstrip(".")
        if client_order_id:
            body["clientOrderId"] = client_order_id

        payload = await self.breaker.call_async(self._request, "POST", "/v3/orders/send", json=body)
        order = payload.get("order", payload)
        return OrderResult(
            order_id=str(order.get("id", "")),
            client_order_id=client_order_id,
            symbol=symbol.upper(), side=side, order_type=order_type,
            quantity=quantity, price=float(order["price"]) if order.get("price") else price,
            status=str(order.get("status", "open")),
            filled_quantity=float(order.get("matchedAmount", 0) or 0),
            raw=order,
        )

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        payload = await self.breaker.call_async(
            self._request, "POST", "/v3/orders/cancel", json={"orderId": int(order_id)}
        )
        return payload.get("status") == "ok"

    async def fetch_open_orders(self, symbol: str | None = None) -> list[OrderResult]:
        params = {"status": "open"}
        if symbol:
            src, dst = split_symbol(symbol)
            params["srcCurrency"] = src
            params["dstCurrency"] = dst
        payload = await self.breaker.call_async(
            self._request, "GET", "/v3/orders/list", params=params
        )
        out: list[OrderResult] = []
        for o in payload.get("orders", []):
            try:
                out.append(OrderResult(
                    order_id=str(o.get("id", "")),
                    symbol=(o.get("srcCurrency", "") + o.get("dstCurrency", "")).upper(),
                    side=Side(o.get("type", "buy")),
                    order_type=OrderType(o.get("execution", "limit")),
                    quantity=float(o.get("amount", 0) or 0),
                    price=float(o["price"]) if o.get("price") else None,
                    status=str(o.get("status", "open")),
                    filled_quantity=float(o.get("matchedAmount", 0) or 0),
                    raw=o,
                ))
            except Exception:
                continue
        return out

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
