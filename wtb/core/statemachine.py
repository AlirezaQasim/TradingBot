"""Position lifecycle state machine.

Enforces that a position can only move through valid transitions
(FLAT -> PENDING_OPEN -> OPEN -> PENDING_CLOSE -> FLAT, plus closed-out
states). The v0.0.1 bot could "open" an already-open position silently.

Implemented as an explicit transition table: trivially testable, serializable,
and dependency-free.
"""
from __future__ import annotations

from enum import Enum


class PositionState(str, Enum):
    FLAT = "flat"
    PENDING_OPEN = "pending_open"
    OPEN = "open"
    PENDING_CLOSE = "pending_close"
    CLOSED = "closed"


# Allowed transitions: source -> {valid targets}
TRANSITIONS: dict[PositionState, set[PositionState]] = {
    PositionState.FLAT: {PositionState.PENDING_OPEN},
    PositionState.PENDING_OPEN: {PositionState.OPEN, PositionState.FLAT},
    PositionState.OPEN: {PositionState.PENDING_CLOSE},
    PositionState.PENDING_CLOSE: {PositionState.FLAT, PositionState.CLOSED},
    PositionState.CLOSED: set(),
}


class InvalidTransition(RuntimeError):
    def __init__(self, current: PositionState, target: PositionState):
        super().__init__(f"Illegal position transition {current.value} -> {target.value}")
        self.current = current
        self.target = target


class PositionLifecycle:
    """Small explicit FSM — no magic, easy to persist and inspect."""

    def __init__(self, state: PositionState = PositionState.FLAT) -> None:
        self.state = state

    def can(self, target: PositionState) -> bool:
        return target in TRANSITIONS[self.state]

    def transition(self, target: PositionState) -> PositionState:
        if not self.can(target):
            raise InvalidTransition(self.state, target)
        self.state = target
        return self.state

    @property
    def is_open(self) -> bool:
        return self.state in (PositionState.OPEN, PositionState.PENDING_CLOSE)

    @property
    def is_flat(self) -> bool:
        return self.state in (PositionState.FLAT, PositionState.CLOSED)

    def __repr__(self) -> str:  # pragma: no cover
        return f"PositionLifecycle({self.state.value})"
