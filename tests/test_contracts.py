"""Tests: data contracts validation (bad payloads must be rejected)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from wtb.data.contracts import Candle, OrderBook, OrderBookLevel, OrderResult, Side


def ts(ms: int) -> int:
    return ms


def test_valid_candle_passes():
    c = Candle(ts=ts(1_700_000_000_000), open=100, high=110, low=95, close=105, volume=12)
    assert c.close == 105


def test_inverted_ohlc_rejected():
    with pytest.raises(ValidationError):
        Candle(ts=ts(1_700_000_000_000), open=110, high=100, low=95, close=105, volume=12)


def test_nonpositive_price_rejected():
    with pytest.raises(ValidationError):
        Candle(ts=ts(1_700_000_000_000), open=0, high=110, low=95, close=105, volume=12)


def test_implausible_timestamp_rejected():
    with pytest.raises(ValidationError):
        Candle(ts=ts(100), open=100, high=110, low=95, close=105, volume=12)  # 1970


def test_future_timestamp_rejected():
    import time
    with pytest.raises(ValidationError):
        Candle(ts=int(time.time() * 1000) + 10 * 86_400_000,
               open=100, high=110, low=95, close=105, volume=12)


def test_orderbook_best_prices():
    ob = OrderBook(
        symbol="BTCIRT",
        bids=[OrderBookLevel(price=99, quantity=1), OrderBookLevel(price=98, quantity=2)],
        asks=[OrderBookLevel(price=101, quantity=1), OrderBookLevel(price=102, quantity=2)],
    )
    assert ob.best_bid == 99 and ob.best_ask == 101 and ob.mid == 100


def test_order_result_side_validation():
    with pytest.raises(ValidationError):
        OrderResult(order_id="1", symbol="X", side="invalid", order_type="market",
                    quantity=1)


def test_side_enum_roundtrip():
    assert Side("buy") == Side.BUY
