"""Tests: risk manager circuit breakers + position sizing (the 100>20 bug class)."""
from __future__ import annotations

import pytest

from wtb.core.config import RiskConfig
from wtb.risk.manager import RiskManager, RiskViolation, RiskState


def make_manager(**overrides) -> RiskManager:
    cfg = RiskConfig(
        stop_loss_pct=0.02, take_profit_pct=0.04,
        max_daily_loss=10.0, max_consecutive_losses=3, profit_target=50.0,
        **overrides,
    )
    return RiskManager(cfg, initial_balance=20.0)


def test_position_size_is_pct_of_balance():
    m = make_manager()
    qty = m.position_quantity(price=100.0)
    assert qty == pytest.approx(20.0 * 0.95 / 100.0)      # 95% of 20 => 0.19 shares
    assert m.can_afford(100.0, qty)


def test_sizing_never_exceeds_balance():
    """v0.0.1 bug: position_size(100) > balance(20) -> every open rejected."""
    m = make_manager()
    qty = m.position_quantity(price=1.0)
    assert m.can_afford(1.0, qty)                          # always affordable by construction
    assert 1.0 * qty <= m.state.balance + 1e-9


def test_max_position_notional_cap():
    m = make_manager(max_position_notional=5.0)
    qty = m.position_quantity(price=10.0)
    assert 10.0 * qty == pytest.approx(5.0)                # capped, not 19


def test_daily_loss_limit_trips():
    m = make_manager()
    m.on_position_closed(pnl=-10.0)
    with pytest.raises(RiskViolation, match="daily loss"):
        m.check_can_trade()


def test_consecutive_losses_trip_and_reset():
    m = make_manager()
    m.on_position_closed(-1)
    m.on_position_closed(-1)
    m.check_can_trade()                                    # streak 2 < 3: ok
    m.on_position_closed(-1)
    with pytest.raises(RiskViolation, match="consecutive"):
        m.check_can_trade()
    m.on_position_closed(+5)                               # a win resets the streak
    m.state.daily_loss = 0.0
    m.check_can_trade()


def test_profit_target_enforced():
    m = make_manager()
    m.state.balance = 51.0
    with pytest.raises(RiskViolation, match="profit target"):
        m.check_can_trade()


def test_kill_switch_blocks_everything():
    m = make_manager()
    m.engage_kill_switch()
    with pytest.raises(RiskViolation, match="kill switch"):
        m.check_can_trade()
    m.release_kill_switch()
    m.check_can_trade()


def test_daily_loss_resets_next_day():
    m = make_manager()
    m.on_position_closed(-10.0)
    import datetime
    m.state.day = datetime.date.today() - datetime.timedelta(days=1)
    m.check_can_trade()                                    # new day => reset
    assert m.state.daily_loss == 0.0
