"""Tests: position lifecycle FSM + signal generator ensemble."""
from __future__ import annotations

import pytest

from wtb.core.config import SignalConfig
from wtb.core.statemachine import InvalidTransition, PositionLifecycle, PositionState
from wtb.data.contracts import OrderBookFeatures, Signal
from wtb.signals.generator import SignalGenerator


# ------------------------------------------------------------------------ FSM
def test_happy_path():
    p = PositionLifecycle()
    assert p.is_flat
    p.transition(PositionState.PENDING_OPEN)
    p.transition(PositionState.OPEN)
    assert p.is_open
    p.transition(PositionState.PENDING_CLOSE)
    p.transition(PositionState.FLAT)
    assert p.is_flat


def test_no_double_open():
    p = PositionLifecycle()
    p.transition(PositionState.PENDING_OPEN)
    p.transition(PositionState.OPEN)
    with pytest.raises(InvalidTransition):
        p.transition(PositionState.PENDING_OPEN)


def test_open_to_flat_directly_is_illegal():
    p = PositionLifecycle()
    p.transition(PositionState.PENDING_OPEN)
    p.transition(PositionState.OPEN)
    with pytest.raises(InvalidTransition):
        p.transition(PositionState.FLAT)


# -------------------------------------------------------------------- signals
def make_ob(spoof_bid: float, spoof_ask: float, imbalance: float, spread_pct: float = 0.001):
    return OrderBookFeatures(
        spread=1.0, spread_pct=spread_pct, volume_imbalance=imbalance,
        bid_slope=0.0, ask_slope=0.0, bid_vwap=100.0, ask_vwap=101.0,
        spoof_ratio_bid=spoof_bid, spoof_ratio_ask=spoof_ask,
        best_bid=100.0, best_ask=101.0,
    )


def test_strong_ai_buy_wins():
    gen = SignalGenerator(SignalConfig())
    sig = gen.generate(ai_prob=0.9, rsi=50.0, bb_width=0.05, price=100.0)
    assert sig.signal == Signal.BUY
    assert sig.ai_vote == Signal.BUY


def test_ai_sell_with_overbought_veto():
    """AI says sell (prob < 0.4) but RSI < 30 => vetoed to hold."""
    gen = SignalGenerator(SignalConfig())
    sig = gen.generate(ai_prob=0.2, rsi=20.0, bb_width=0.05, price=100.0)
    assert sig.signal == Signal.HOLD
    assert "veto" in sig.reason


def test_ai_buy_with_overbought_rsi_vetoed():
    gen = SignalGenerator(SignalConfig())
    sig = gen.generate(ai_prob=0.9, rsi=75.0, bb_width=0.05, price=100.0)
    assert sig.signal == Signal.HOLD


def test_orderbook_spoof_sell():
    gen = SignalGenerator(SignalConfig())
    ob = make_ob(spoof_bid=0.1, spoof_ask=0.8, imbalance=-0.5)
    sig = gen.generate(ai_prob=0.5, rsi=50.0, bb_width=0.05, price=100.0, orderbook=ob)
    assert sig.orderbook_vote == Signal.SELL


def test_wide_spread_forces_hold():
    gen = SignalGenerator(SignalConfig())
    ob = make_ob(spoof_bid=0.0, spoof_ask=0.0, imbalance=0.0, spread_pct=0.02)
    sig = gen.generate(ai_prob=0.9, rsi=50.0, bb_width=0.05, price=100.0, orderbook=ob)
    assert sig.signal == Signal.HOLD
    assert "spread" in sig.reason


def test_weighted_ensemble_ai_beats_single_vote():
    cfg = SignalConfig(ai_weight=2, wave_weight=1, orderbook_weight=1)
    gen = SignalGenerator(cfg)
    ob = make_ob(spoof_bid=0.8, spoof_ask=0.1, imbalance=0.5)   # orderbook: buy (1)
    sig = gen.generate(ai_prob=0.2, rsi=50.0, bb_width=0.05,    # ai: sell (2)
                       price=100.0, orderbook=ob)
    assert sig.signal == Signal.SELL                            # 2 > 1
