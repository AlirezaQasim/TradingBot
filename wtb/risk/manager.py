"""Risk manager: position sizing, circuit breakers, idempotency, kill switch.

Fixes vs v0.0.1:
- Position size is a **percentage of current balance** (v0.0.1 used an absolute
  ``position_size: 100`` against a 20 balance — no trade could ever open).
- ``profit_target`` is actually enforced (v0.0.1 loaded it and ignored it).
- Every rejection/halt is recorded as a RiskEvent in PostgreSQL.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date

from wtb.core.config import RiskConfig
from wtb.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class TradeRecord:
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    reason: str
    ts: float


@dataclass
class RiskState:
    balance: float
    daily_loss: float = 0.0
    consecutive_losses: int = 0
    day: date = field(default_factory=date.today)
    kill_switch: bool = False


class RiskViolation(Exception):
    """Trading blocked by a risk rule — carries the reason for observability."""


class RiskManager:
    def __init__(self, cfg: RiskConfig, initial_balance: float) -> None:
        self.cfg = cfg
        self.state = RiskState(balance=initial_balance)

    # ------------------------------------------------------------- controls #
    def check_can_trade(self) -> None:
        """Raise RiskViolation if any circuit breaker trips."""
        s = self.state
        if s.kill_switch:
            raise RiskViolation("kill switch engaged")
        if s.day != date.today():
            s.day = date.today()
            s.daily_loss = 0.0
        if s.daily_loss >= self.cfg.max_daily_loss:
            raise RiskViolation(
                f"daily loss {s.daily_loss:.2f} >= limit {self.cfg.max_daily_loss:.2f}"
            )
        if s.consecutive_losses >= self.cfg.max_consecutive_losses:
            raise RiskViolation(
                f"{s.consecutive_losses} consecutive losses >= limit "
                f"{self.cfg.max_consecutive_losses}"
            )
        if s.balance >= self.cfg.profit_target:
            raise RiskViolation(
                f"profit target reached: balance {s.balance:.2f} >= {self.cfg.profit_target:.2f}"
            )

    def position_quantity(self, price: float) -> float:
        """Shares bought with position_size_pct of current balance."""
        notional = self.state.balance * min(1.0, self.cfg.position_size_pct)
        if self.cfg.max_position_notional is not None:
            notional = min(notional, self.cfg.max_position_notional)
        return notional / (price + 1e-12)

    def can_afford(self, price: float, quantity: float) -> bool:
        return price * quantity <= self.state.balance + 1e-9

    def new_idempotency_key(self, symbol: str) -> str:
        return f"{symbol}-{date.today().isoformat()}-{uuid.uuid4().hex[:12]}"

    # -------------------------------------------------------------- feedback #
    def on_position_opened(self, price: float, quantity: float) -> None:
        pass  # balance accounting handled by the executor

    def on_position_closed(self, pnl: float) -> None:
        s = self.state
        s.balance += pnl
        if pnl < 0:
            s.daily_loss += abs(pnl)
            s.consecutive_losses += 1
        elif pnl > 0:
            s.consecutive_losses = 0
        log.info("risk_state_updated", balance=round(s.balance, 4),
                 daily_loss=round(s.daily_loss, 4), streak=s.consecutive_losses)

    def engage_kill_switch(self) -> None:
        self.state.kill_switch = True
        log.error("kill_switch_engaged")

    def release_kill_switch(self) -> None:
        self.state.kill_switch = False
        log.warning("kill_switch_released")
