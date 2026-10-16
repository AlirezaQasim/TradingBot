"""Transactional state (positions, orders, risk events) — SQLAlchemy 2.0 async ORM."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from wtb.core.logging import get_logger

log = get_logger(__name__)


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    pass


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    exchange: Mapped[str] = mapped_column(String(32), default="")
    side: Mapped[str] = mapped_column(String(8))                     # buy | sell (paper simulation)
    quantity: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    stop_loss: Mapped[float] = mapped_column(Float)
    take_profit: Mapped[float] = mapped_column(Float)
    state: Mapped[str] = mapped_column(String(16), default="open", index=True)  # open | closed
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)  # signal | stop_loss | take_profit
    idempotency_key: Mapped[str] = mapped_column(String(64), index=True)

    orders: Mapped[list["OrderRecord"]] = relationship(back_populates="position")


class OrderRecord(Base):
    __tablename__ = "orders"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_orders_idempotency"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    position_id: Mapped[int | None] = mapped_column(ForeignKey("positions.id"), nullable=True)
    exchange: Mapped[str] = mapped_column(String(32), default="")
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    order_type: Mapped[str] = mapped_column(String(8), default="market")
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="new")
    idempotency_key: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    position: Mapped[Position | None] = relationship(back_populates="orders")


class RiskEvent(Base):
    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    kind: Mapped[str] = mapped_column(String(32))       # daily_loss_limit | consecutive_losses | kill_switch | ...
    detail: Mapped[str] = mapped_column(Text, default="")


class Engine:
    """Async engine + session factory for the app database."""

    def __init__(self, app_dsn: str) -> None:
        # SQLAlchemy async needs the asyncpg driver scheme
        url = app_dsn.replace("postgresql://", "postgresql+asyncpg://")
        self.engine = create_async_engine(url, echo=False, pool_size=5, max_overflow=5)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def migrate(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("postgres_migrated")

    async def close(self) -> None:
        await self.engine.dispose()


async def record_risk_event(session: AsyncSession, kind: str, detail: str = "") -> None:
    session.add(RiskEvent(kind=kind, detail=detail))
    await session.commit()
    log.warning("risk_event", kind=kind, detail=detail)
