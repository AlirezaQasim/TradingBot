"""Technical indicators and feature engineering on Polars.

Fixes vs v0.0.1:
- Polars instead of pandas (10-50x faster on tick/OHLCV data, lazy evaluation)
- RSI uses Wilder's smoothing with a division-by-zero epsilon (v0.0.1 produced
  ``inf`` on zero-loss windows)
- One implementation shared by training and inference (v0.0.1 computed features
  twice, in the loop and in prepare_data, and the two could drift)
- All periods configurable via ``IndicatorsConfig`` (config: indicators:)
"""
from __future__ import annotations

import polars as pl

from wtb.core.config import IndicatorsConfig

DEFAULTS = IndicatorsConfig()
_EPS = 1e-10


def add_returns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1.0).fill_nan(0.0).alias("returns")
    )


def add_rsi(df: pl.DataFrame, period: int = DEFAULTS.rsi_period) -> pl.DataFrame:
    delta = pl.col("close").diff()
    gain = delta.clip(0.0)
    loss = (-delta).clip(0.0)
    # Wilder's smoothing == EWM with alpha=1/period
    avg_gain = gain.ewm_mean(alpha=1 / period, adjust=False, ignore_nulls=True)
    avg_loss = loss.ewm_mean(alpha=1 / period, adjust=False, ignore_nulls=True)
    rs = (avg_gain + _EPS) / (avg_loss + _EPS)
    return df.with_columns((100.0 - 100.0 / (1.0 + rs)).alias("rsi"))


def add_macd(df: pl.DataFrame, fast: int = DEFAULTS.macd_fast,
             slow: int = DEFAULTS.macd_slow, signal: int = DEFAULTS.macd_signal) -> pl.DataFrame:
    ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False, ignore_nulls=True)
    ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False, ignore_nulls=True)
    df = df.with_columns((ema_fast - ema_slow).alias("macd"))
    return df.with_columns(
        pl.col("macd").ewm_mean(span=signal, adjust=False, ignore_nulls=True).alias("macd_signal")
    )


def add_bollinger(df: pl.DataFrame, period: int = DEFAULTS.bb_period,
                  nstd: float = DEFAULTS.bb_std) -> pl.DataFrame:
    mid = pl.col("close").rolling_mean(period)
    std = pl.col("close").rolling_std(period)
    df = df.with_columns(
        (mid + nstd * std).alias("bb_upper"),
        (mid - nstd * std).alias("bb_lower"),
    )
    return df.with_columns(
        ((pl.col("bb_upper") - pl.col("bb_lower")) / (pl.col("bb_upper") + _EPS)).alias("bb_width")
    )


ALL_INDICATORS = ("returns", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_width")


def add_indicators(df: pl.DataFrame, ind: IndicatorsConfig | None = None) -> pl.DataFrame:
    """Add every supported indicator column. ``df`` needs open/high/low/close/volume."""
    ind = ind or DEFAULTS
    df = add_returns(df)
    df = add_rsi(df, ind.rsi_period)
    df = add_macd(df, ind.macd_fast, ind.macd_slow, ind.macd_signal)
    df = add_bollinger(df, ind.bb_period, ind.bb_std)
    return df


def candles_to_frame(candles) -> pl.DataFrame:
    """list[Candle] -> Polars DataFrame sorted by ts."""
    rows = [c.model_dump() for c in candles]
    if not rows:
        return pl.DataFrame(schema={
            "ts": pl.Int64, "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64,
            "symbol": pl.Utf8, "timeframe": pl.Utf8,
        })
    return pl.DataFrame(rows).sort("ts")


def feature_matrix(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Extract the model's feature columns, forward-filling warm-up NaNs."""
    return df.select(features).with_columns(pl.all().forward_fill()).with_columns(pl.all().fill_nan(0.0))


# --------------------------------------------------------------------------- #
# Order-book microstructure (port of v0.0.1 compute_order_book_features)
# --------------------------------------------------------------------------- #
def compute_order_book_features(book) -> "OrderBookFeatures":
    import numpy as np

    from wtb.data.contracts import OrderBookFeatures

    bid_p = np.array([lvl.price for lvl in book.bids], dtype=float)
    bid_q = np.array([lvl.quantity for lvl in book.bids], dtype=float)
    ask_p = np.array([lvl.price for lvl in book.asks], dtype=float)
    ask_q = np.array([lvl.quantity for lvl in book.asks], dtype=float)

    best_bid, best_ask = float(bid_p[0]), float(ask_p[0])
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    spread_pct = spread / (mid + _EPS)

    bid_vol, ask_vol = float(bid_q.sum()), float(ask_q.sum())
    volume_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + _EPS)

    bid_slope = float(np.polyfit(np.arange(len(bid_q)), bid_q, 1)[0]) if len(bid_q) > 1 else 0.0
    ask_slope = float(np.polyfit(np.arange(len(ask_q)), ask_q, 1)[0]) if len(ask_q) > 1 else 0.0

    bid_vwap = float((bid_p * bid_q).sum() / (bid_vol + _EPS))
    ask_vwap = float((ask_p * ask_q).sum() / (ask_vol + _EPS))

    spoof_ratio_bid = float(bid_q[0] / (bid_vol + _EPS))
    spoof_ratio_ask = float(ask_q[0] / (ask_vol + _EPS))

    return OrderBookFeatures(
        spread=spread, spread_pct=spread_pct,
        volume_imbalance=volume_imbalance,
        bid_slope=bid_slope, ask_slope=ask_slope,
        bid_vwap=bid_vwap, ask_vwap=ask_vwap,
        spoof_ratio_bid=spoof_ratio_bid, spoof_ratio_ask=spoof_ratio_ask,
        best_bid=best_bid, best_ask=best_ask,
    )
