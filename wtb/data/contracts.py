"""Strict data contracts (Pydantic v2).

Bad payloads are rejected at the boundary — the v0.0.1 bot fed whatever an
exchange returned straight into its model.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional

import pydantic
from pydantic import BaseModel, Field, field_validator, model_validator


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Signal(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class Candle(BaseModel):
    ts: int = Field(..., description="Open time, epoch milliseconds", ge=0)
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)
    symbol: str = ""
    timeframe: str = ""

    @model_validator(mode="after")
    def _ohlc_consistent(self) -> "Candle":
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError(f"Inconsistent OHLC: o={self.open} h={self.high} l={self.low} c={self.close}")
        return self

    @field_validator("ts")
    @classmethod
    def _sane_timestamp(cls, v: int) -> int:
        # Reject timestamps before 2010 or more than a day in the future.
        now_ms = int(time.time() * 1000)
        if v < 1_262_304_000_000 or v > now_ms + 86_400_000:
            raise ValueError(f"Implausible candle timestamp: {v}")
        return v


class OrderBookLevel(BaseModel):
    price: float = Field(..., gt=0)
    quantity: float = Field(..., ge=0)


class OrderBook(BaseModel):
    symbol: str
    ts: int = Field(default_factory=lambda: int(time.time() * 1000))
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


class Ticker(BaseModel):
    symbol: str
    ts: int = Field(default_factory=lambda: int(time.time() * 1000))
    last: float = Field(..., gt=0)
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: Optional[float] = None


class OrderResult(BaseModel):
    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float = Field(..., gt=0)
    price: Optional[float] = None
    status: str = "open"
    filled_quantity: float = 0.0
    raw: dict = Field(default_factory=dict)


class OrderBookFeatures(BaseModel):
    """Microstructure features computed from an order book snapshot."""
    spread: float
    spread_pct: float
    volume_imbalance: float          # (bidV - askV) / (bidV + askV), in [-1, 1]
    bid_slope: float
    ask_slope: float
    bid_vwap: float
    ask_vwap: float
    spoof_ratio_bid: float           # top-level share of total bid volume
    spoof_ratio_ask: float
    best_bid: float
    best_ask: float


class BotSignal(BaseModel):
    """Final fused signal emitted by the signal generator."""
    signal: Signal
    ai_prob: Optional[float] = None
    ai_vote: Signal = Signal.HOLD
    wave_vote: Signal = Signal.HOLD
    orderbook_vote: Signal = Signal.HOLD
    score_buy: int = 0
    score_sell: int = 0
    price: float
    ts: int = Field(default_factory=lambda: int(time.time() * 1000))
    reason: str = ""
