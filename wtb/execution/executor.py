"""Execution layer.

- :class:`PaperExecutor`  — simulated fills against internal balance (default).
- :class:`LiveExecutor`   — real orders through the exchange adapter, gated by
  ``trading.live_trading`` and protected by idempotency keys + the exchange
  circuit breaker. Spot only (no shorts); shorts stay a paper simulation.

Both executors share the same interface so the engine is mode-agnostic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from wtb.core.logging import get_logger
from wtb.core.statemachine import PositionLifecycle, PositionState
from wtb.data.contracts import OrderType, Side
from wtb.db.postgres import Engine, OrderRecord, Position, utcnow
from wtb.exchanges.base import ExchangeAdapter
from wtb.risk.manager import RiskManager

log = get_logger(__name__)


@dataclass
class ManagedPosition:
    side: Side
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: float = field(default_factory=time.time)
    lifecycle: PositionLifecycle = field(default_factory=PositionLifecycle)
    order_id: str | None = None          # exchange order id (live mode)
    idempotency_key: str = ""
    entry_fee: float = 0.0               # fees paid at open (netted at close)
    best_price: float = 0.0              # best seen price (trailing-stop anchor)

    def __post_init__(self) -> None:
        if self.best_price == 0.0:
            self.best_price = self.entry_price

    def gross_pnl(self, price: float) -> float:
        """Long: (exit - entry) * qty. Short: (entry - exit) * qty (paper only)."""
        direction = 1.0 if self.side == Side.BUY else -1.0
        return (price - self.entry_price) * self.quantity * direction

    def pnl(self, price: float) -> float:
        return self.gross_pnl(price)  # fees handled by executors

    def ratchet_trailing_stop(self, price: float, trailing_pct: float | None) -> None:
        """Move the stop in the position's favour; never widen it."""
        if not trailing_pct or trailing_pct <= 0:
            return
        if self.side == Side.BUY:
            self.best_price = max(self.best_price, price)
            self.stop_loss = max(self.stop_loss, self.best_price * (1.0 - trailing_pct))
        else:
            self.best_price = min(self.best_price, price)
            self.stop_loss = min(self.stop_loss, self.best_price * (1.0 + trailing_pct))


class Executor(Protocol):
    position: ManagedPosition | None

    async def open_position(
        self, side: Side, price: float, risk: RiskManager, symbol: str
    ) -> ManagedPosition | None: ...

    async def close_position(
        self, price: float, reason: str, risk: RiskManager, symbol: str
    ) -> float | None: ...

    async def check_exits(self, price: float) -> str | None:
        """Return 'stop_loss' / 'take_profit' if an exit level was crossed."""


# --------------------------------------------------------------------------- #
class PaperExecutor:
    """Simulated fills with configurable fee + slippage (the default executor)."""

    def __init__(self, engine: Engine | None, exchange_name: str = "",
                 execution: "ExecutionConfig | None" = None) -> None:
        from wtb.core.config import ExecutionConfig
        self.engine = engine
        self.exchange_name = exchange_name
        self.execution = execution or ExecutionConfig()
        self.position: ManagedPosition | None = None

    def _fill(self, price: float, side: Side, closing: bool = False) -> float:
        """Adverse fill: buys pay above mid, sells hit below mid — always.

        (An opening BUY and a short-covering BUY both cross the spread upward;
        an opening short-SALE and a closing SELL both cross it downward, so the
        direction depends only on the side, not on whether it opens or closes.)
        """
        s = self.execution.slippage_pct
        return price * (1 + s) if side == Side.BUY else price * (1 - s)

    def _fee(self, notional: float) -> float:
        return notional * self.execution.fee_pct

    async def open_position(
        self, side: Side, price: float, risk: RiskManager, symbol: str
    ) -> ManagedPosition | None:
        if self.position is not None and self.position.lifecycle.is_open:
            log.warning("open_ignored", reason="position already open")
            return None
        fill = self._fill(price, side, closing=False)
        quantity = risk.position_quantity(fill)
        if not risk.can_afford(fill, quantity):
            log.warning("open_rejected", reason="insufficient balance",
                        balance=risk.state.balance, cost=fill * quantity)
            return None

        sl_dist = fill * risk.cfg.stop_loss_pct
        tp_dist = fill * risk.cfg.take_profit_pct
        pos = ManagedPosition(
            side=side,
            quantity=quantity,
            entry_price=fill,
            # long: SL below / TP above; short: mirrored
            stop_loss=fill - sl_dist if side == Side.BUY else fill + sl_dist,
            take_profit=fill + tp_dist if side == Side.BUY else fill - tp_dist,
            idempotency_key=risk.new_idempotency_key(symbol),
            entry_fee=self._fee(fill * quantity),
        )
        pos.lifecycle.transition(PositionState.PENDING_OPEN)
        pos.lifecycle.transition(PositionState.OPEN)
        self.position = pos

        if self.engine is not None:
            async with self.engine.session_factory() as session:
                session.add(Position(
                    symbol=symbol, exchange=self.exchange_name, side=side.value,
                    quantity=quantity, entry_price=fill,
                    stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                    state="open", idempotency_key=pos.idempotency_key,
                ))
                await session.commit()

        log.info("paper_position_opened", symbol=symbol, side=side.value,
                 quantity=round(quantity, 8), fill=fill, slip=self.execution.slippage_pct,
                 fee=round(pos.entry_fee, 6), sl=pos.stop_loss, tp=pos.take_profit)
        return pos

    async def close_position(
        self, price: float, reason: str, risk: RiskManager, symbol: str
    ) -> float | None:
        pos = self.position
        if pos is None or not pos.lifecycle.is_open:
            return None
        pos.lifecycle.transition(PositionState.PENDING_CLOSE)
        exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
        fill = self._fill(price, exit_side, closing=True)
        exit_fee = self._fee(fill * pos.quantity)
        pnl = pos.gross_pnl(fill) - pos.entry_fee - exit_fee
        pos.lifecycle.transition(PositionState.FLAT)
        risk.on_position_closed(pnl)
        await self._mark_closed(symbol, pos, fill, pnl, reason)
        log.info("paper_position_closed", symbol=symbol, reason=reason,
                 entry=pos.entry_price, exit=fill, fees=round(pos.entry_fee + exit_fee, 6),
                 pnl=round(pnl, 6), balance=round(risk.state.balance, 4))
        self.position = None
        return pnl

    async def _mark_closed(self, symbol: str, pos: ManagedPosition,
                           exit_price: float, pnl: float, reason: str) -> None:
        """Close the matching Position row (v0.0.1/earlier builds left stale rows)."""
        if self.engine is None:
            return
        from sqlalchemy import update

        from wtb.db.postgres import Position
        try:
            async with self.engine.session_factory() as session:
                await session.execute(
                    update(Position)
                    .where(Position.idempotency_key == pos.idempotency_key)
                    .values(state="closed", exit_price=exit_price, pnl=pnl,
                            closed_at=utcnow(), close_reason=reason)
                )
                await session.commit()
        except Exception as exc:
            log.warning("position_close_record_failed", error=str(exc))

    async def check_exits(self, price: float) -> str | None:
        pos = self.position
        if pos is None or not pos.lifecycle.is_open:
            return None
        pos.ratchet_trailing_stop(price, self.execution.trailing_stop_pct)
        if pos.side == Side.BUY:
            if price <= pos.stop_loss:
                return "stop_loss"
            if price >= pos.take_profit:
                return "take_profit"
        else:
            if price >= pos.stop_loss:
                return "stop_loss"
            if price <= pos.take_profit:
                return "take_profit"
        return None


# --------------------------------------------------------------------------- #
class LiveExecutor:
    """Real order execution — only used when ``trading.live_trading: true``."""

    def __init__(
        self,
        exchange: ExchangeAdapter,
        engine: Engine | None,
        exchange_name: str = "",
        execution: "ExecutionConfig | None" = None,
    ) -> None:
        from wtb.core.config import ExecutionConfig
        self.exchange = exchange
        self.engine = engine
        self.exchange_name = exchange_name or exchange.name
        self.execution = execution or ExecutionConfig()
        self.position: ManagedPosition | None = None

    async def open_position(
        self, side: Side, price: float, risk: RiskManager, symbol: str
    ) -> ManagedPosition | None:
        if self.position is not None and self.position.lifecycle.is_open:
            log.warning("open_ignored", reason="position already open")
            return None

        idem = risk.new_idempotency_key(symbol)
        order = await self.exchange.create_order(
            symbol=symbol, side=side, order_type=OrderType.MARKET,
            quantity=risk.position_quantity(price), client_order_id=idem,
        )

        quantity = order.filled_quantity or order.quantity
        fill_price = order.price or price
        sl_dist = fill_price * risk.cfg.stop_loss_pct
        tp_dist = fill_price * risk.cfg.take_profit_pct
        pos = ManagedPosition(
            side=side, quantity=quantity, entry_price=fill_price,
            # spot: SL/TP are enforced client-side by the engine loop
            stop_loss=fill_price - sl_dist if side == Side.BUY else fill_price + sl_dist,
            take_profit=fill_price + tp_dist if side == Side.BUY else fill_price - tp_dist,
            order_id=order.order_id, idempotency_key=idem,
        )
        pos.lifecycle.transition(PositionState.PENDING_OPEN)
        pos.lifecycle.transition(PositionState.OPEN)
        self.position = pos

        if self.engine is not None:
            async with self.engine.session_factory() as session:
                session.add(OrderRecord(
                    symbol=symbol, exchange=self.exchange_name, side=side.value,
                    order_type=OrderType.MARKET.value, quantity=quantity,
                    price=fill_price, exchange_order_id=order.order_id,
                    status=order.status, idempotency_key=idem,
                ))
                await session.commit()

        log.info("live_order_placed", symbol=symbol, side=side.value,
                 quantity=quantity, order_id=order.order_id, idem=idem)
        return pos

    async def close_position(
        self, price: float, reason: str, risk: RiskManager, symbol: str
    ) -> float | None:
        pos = self.position
        if pos is None or not pos.lifecycle.is_open:
            return None
        pos.lifecycle.transition(PositionState.PENDING_CLOSE)
        exit_side = Side.SELL if pos.side == Side.BUY else Side.BUY
        idem = risk.new_idempotency_key(symbol)
        try:
            order = await self.exchange.create_order(
                symbol=symbol, side=exit_side, order_type=OrderType.MARKET,
                quantity=pos.quantity, client_order_id=idem,
            )
            exit_price = order.price or price
        except Exception as exc:
            pos.lifecycle.transition(PositionState.OPEN)   # roll back transition
            log.error("live_close_failed", error=str(exc), idem=idem)
            return None

        pnl = pos.pnl(exit_price)
        pos.lifecycle.transition(PositionState.FLAT)
        risk.on_position_closed(pnl)
        self.position = None
        log.info("live_position_closed", symbol=symbol, reason=reason,
                 exit=exit_price, pnl=round(pnl, 6))
        return pnl

    async def check_exits(self, price: float) -> str | None:
        pos = self.position
        if pos is None or not pos.lifecycle.is_open:
            return None
        pos.ratchet_trailing_stop(price, self.execution.trailing_stop_pct)
        if pos.side == Side.BUY:
            if price <= pos.stop_loss:
                return "stop_loss"
            if price >= pos.take_profit:
                return "take_profit"
        return None


def build_executor(mode: str, exchange: ExchangeAdapter | None, engine: Engine | None,
                   exchange_name: str = "",
                   execution: "ExecutionConfig | None" = None) -> Executor:
    if mode == "live":
        if exchange is None:
            raise ValueError("Live executor requires an exchange adapter")
        return LiveExecutor(exchange, engine, exchange_name, execution)
    return PaperExecutor(engine, exchange_name, execution)
