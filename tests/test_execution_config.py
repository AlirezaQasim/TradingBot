"""Tests: execution customization — fees, slippage, trailing stop, short gate."""
from __future__ import annotations

import pytest

from wtb.core.config import ExecutionConfig, RiskConfig
from wtb.data.contracts import Side
from wtb.execution.executor import PaperExecutor
from wtb.risk.manager import RiskManager


def make_risk() -> RiskManager:
    return RiskManager(RiskConfig(
        stop_loss_pct=0.02, take_profit_pct=0.04, max_daily_loss=1e9,
        max_consecutive_losses=1000, profit_target=1e9, position_size_pct=1.0,
    ), initial_balance=1000.0)


def make_executor(**exec_kwargs) -> PaperExecutor:
    return PaperExecutor(engine=None, exchange_name="test",
                         execution=ExecutionConfig(**exec_kwargs))


@pytest.mark.asyncio
async def test_fee_reduces_pnl():
    ex = make_executor(fee_pct=0.001, slippage_pct=0.0)
    risk = make_risk()
    await ex.open_position(Side.BUY, price=100.0, risk=risk, symbol="T")
    pnl = await ex.close_position(price=100.0, reason="signal", risk=risk, symbol="T")
    # flat trade: gross 0, fees = 0.1% of 1000 notional on each side
    assert pnl == pytest.approx(-1000 * 0.001 * 2)
    assert risk.state.balance == pytest.approx(1000 + pnl)


@pytest.mark.asyncio
async def test_slippage_makes_entry_more_expensive_for_long():
    ex = make_executor(fee_pct=0.0, slippage_pct=0.001)
    risk = make_risk()
    pos = await ex.open_position(Side.BUY, price=100.0, risk=risk, symbol="T")
    assert pos.entry_price == pytest.approx(100.0 * 1.001)   # buy fills high
    # exit fills low for a closing sell
    pnl = await ex.close_position(price=100.0, reason="signal", risk=risk, symbol="T")
    assert pnl < 0                                            # round-trip slippage cost


@pytest.mark.asyncio
async def test_trailing_stop_ratchets_long():
    ex = make_executor(fee_pct=0.0, slippage_pct=0.0, trailing_stop_pct=0.01)
    risk = make_risk()
    pos = await ex.open_position(Side.BUY, price=100.0, risk=risk, symbol="T")
    assert pos.stop_loss == pytest.approx(98.0)
    await ex.check_exits(price=110.0)                         # price rallies
    assert pos.stop_loss == pytest.approx(110.0 * 0.99)       # stop ratcheted up
    assert await ex.check_exits(price=108.8) == "stop_loss"   # trail caught it


@pytest.mark.asyncio
async def test_trailing_stop_never_widens():
    ex = make_executor(fee_pct=0.0, slippage_pct=0.0, trailing_stop_pct=0.01)
    risk = make_risk()
    pos = await ex.open_position(Side.BUY, price=100.0, risk=risk, symbol="T")
    await ex.check_exits(price=110.0)
    best_stop = pos.stop_loss
    await ex.check_exits(price=101.0)                         # pullback
    assert pos.stop_loss == best_stop                          # never moves down


@pytest.mark.asyncio
async def test_trailing_stop_short_side():
    ex = make_executor(fee_pct=0.0, slippage_pct=0.0, trailing_stop_pct=0.01)
    risk = make_risk()
    pos = await ex.open_position(Side.SELL, price=100.0, risk=risk, symbol="T")
    assert pos.stop_loss == pytest.approx(102.0)
    await ex.check_exits(price=90.0)                          # price falls: short profits
    assert pos.stop_loss == pytest.approx(90.0 * 1.01)        # stop ratcheted down


@pytest.mark.asyncio
async def test_no_trailing_by_default():
    ex = make_executor(fee_pct=0.0, slippage_pct=0.0)         # trailing None
    risk = make_risk()
    pos = await ex.open_position(Side.BUY, price=100.0, risk=risk, symbol="T")
    await ex.check_exits(price=120.0)
    assert pos.stop_loss == pytest.approx(98.0)               # unchanged
