"""Tests: paper executor end-to-end (open, SL/TP exits, close, balance math).

Frictionless fills (fee=0, slippage=0) so the PnL math is exact; realistic
fee/slippage/trailing behaviour is covered in test_execution_config.py.
"""
from __future__ import annotations

import pytest

from wtb.core.config import ExecutionConfig, RiskConfig
from wtb.data.contracts import Side
from wtb.execution.executor import PaperExecutor
from wtb.risk.manager import RiskManager


@pytest.fixture
def risk() -> RiskManager:
    cfg = RiskConfig(stop_loss_pct=0.02, take_profit_pct=0.04,
                     max_daily_loss=10.0, max_consecutive_losses=100,
                     profit_target=1e9, position_size_pct=1.0)
    return RiskManager(cfg, initial_balance=1000.0)


@pytest.fixture
def executor() -> PaperExecutor:
    frictionless = ExecutionConfig(fee_pct=0.0, slippage_pct=0.0, trailing_stop_pct=None)
    return PaperExecutor(engine=None, exchange_name="test", execution=frictionless)


@pytest.mark.asyncio
async def test_open_and_close_long_profit(risk, executor):
    pos = await executor.open_position(Side.BUY, price=100.0, risk=risk, symbol="TEST")
    assert pos is not None
    assert pos.quantity == pytest.approx(10.0)      # 1000 * 1.0 / 100
    assert pos.stop_loss == pytest.approx(98.0)
    assert pos.take_profit == pytest.approx(104.0)

    pnl = await executor.close_position(price=105.0, reason="signal",
                                        risk=risk, symbol="TEST")
    assert pnl == pytest.approx(50.0)               # (105-100)*10
    assert risk.state.balance == pytest.approx(1050.0)
    assert executor.position is None


@pytest.mark.asyncio
async def test_stop_loss_exit_detected(risk, executor):
    await executor.open_position(Side.BUY, price=100.0, risk=risk, symbol="TEST")
    assert await executor.check_exits(price=98.0) == "stop_loss"    # SL = 100*(1-0.02)
    assert await executor.check_exits(price=99.5) is None           # above the stop
    assert await executor.check_exits(price=104.0) == "take_profit" # TP = 100*1.04


@pytest.mark.asyncio
async def test_loss_feeds_risk_state(risk, executor):
    await executor.open_position(Side.BUY, price=100.0, risk=risk, symbol="TEST")
    pnl = await executor.close_position(price=98.0, reason="stop_loss",
                                        risk=risk, symbol="TEST")
    assert pnl == pytest.approx(-20.0)
    assert risk.state.daily_loss == pytest.approx(20.0)
    assert risk.state.consecutive_losses == 1


@pytest.mark.asyncio
async def test_no_double_open(risk, executor):
    first = await executor.open_position(Side.BUY, price=100.0, risk=risk, symbol="TEST")
    second = await executor.open_position(Side.BUY, price=100.0, risk=risk, symbol="TEST")
    assert first is not None and second is None


@pytest.mark.asyncio
async def test_short_paper_simulation(risk, executor):
    pos = await executor.open_position(Side.SELL, price=100.0, risk=risk, symbol="TEST")
    assert pos.stop_loss == pytest.approx(102.0)     # mirrored
    assert pos.take_profit == pytest.approx(96.0)
    pnl = await executor.close_position(price=95.0, reason="take_profit",
                                        risk=risk, symbol="TEST")
    assert pnl == pytest.approx(50.0)                # short profits when price falls
