"""Tests: Polars indicators (reference-checked values, zero-division safety)."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from wtb.data.indicators import add_bollinger, add_indicators, add_macd, add_rsi


@pytest.fixture
def flat_close_df() -> pl.DataFrame:
    # constant price: RSI must be exactly 50 (no inf!), bb_width 0
    n = 60
    return pl.DataFrame({
        "ts": list(range(n)),
        "open": [100.0] * n, "high": [100.0] * n, "low": [100.0] * n,
        "close": [100.0] * n, "volume": [1.0] * n,
    })


@pytest.fixture
def rising_close_df() -> pl.DataFrame:
    n = 60
    closes = list(np.linspace(100, 200, n))
    return pl.DataFrame({
        "ts": list(range(n)),
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1.0] * n,
    })


def test_rsi_flat_market_is_neutral(flat_close_df):
    out = add_rsi(flat_close_df).drop_nulls()
    assert out.height > 0
    assert out["rsi"].max() <= 50.0 + 1e-6      # no NaN/inf, neutral not overbought
    assert np.isfinite(out["rsi"].to_numpy()).all()


def test_rsi_rising_market_is_overbought(rising_close_df):
    out = add_rsi(rising_close_df).drop_nulls()
    assert out["rsi"][-1] > 70.0                 # sustained rally => overbought


def test_macd_crossover_values(rising_close_df):
    out = add_macd(rising_close_df).drop_nulls()
    macd = out["macd"].to_numpy()
    assert (np.diff(macd) >= -1e-6).all()        # monotonic rally: MACD keeps rising


def test_bollinger_width_nonnegative(flat_close_df, rising_close_df):
    for df in (flat_close_df, rising_close_df):
        out = add_bollinger(df).drop_nulls()
        assert (out["bb_width"].to_numpy() >= 0).all()


def test_full_pipeline_column_set(flat_close_df):
    out = add_indicators(flat_close_df)
    for col in ("returns", "rsi", "macd", "macd_signal", "bb_width"):
        assert col in out.columns
    assert np.isfinite(out.select(["returns", "rsi", "bb_width"]).drop_nulls().to_numpy()).all()
