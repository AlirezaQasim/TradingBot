"""Structured logging via structlog.

JSON lines to file (queryable in Loki/Grafana), pretty console output when a
TTY is attached. Log files rotate at 10 MB, 5 backups kept.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    log_file: str = "webtraderbot.log",
    json_console: bool = False,
) -> None:
    """Configure structlog + stdlib logging once for the whole process."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)

    handlers: list[logging.Handler] = []
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(
            log_dir / log_file, maxBytes=10 * 1024 * 1024, backupCount=5,
        ))

    console = logging.StreamHandler(sys.stderr)
    handlers.append(console)

    is_tty = sys.stderr.isatty()
    renderer = (
        structlog.dev.ConsoleRenderer(colors=is_tty)
        if is_tty and not json_console
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # stdlib records (aiohttp, ccxt, celery...) flow through the same handlers;
    # structlog renders its own events natively.
    formatter = logging.Formatter("%(message)s")
    for h in handlers:
        h.setFormatter(formatter)
        root.addHandler(h)

    for noisy in ("urllib3", "asyncio", "aiosqlite", "ccxt"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, root.level))


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
