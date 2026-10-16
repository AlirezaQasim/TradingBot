"""TimescaleDB schema (executed by scripts/migrate.py).

Created in ``wtb_ts``:
- ohlcv                 hypertable (time, symbol, timeframe)
- orderbook_snapshots   hypertable
- predictions           model predictions for audit/retraining
"""
from __future__ import annotations

TIMESCALE_DDL = [
    # --- OHLCV ------------------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS ohlcv (
        time        TIMESTAMPTZ NOT NULL,
        symbol      TEXT        NOT NULL,
        timeframe   TEXT        NOT NULL,
        open        DOUBLE PRECISION NOT NULL,
        high        DOUBLE PRECISION NOT NULL,
        low         DOUBLE PRECISION NOT NULL,
        close       DOUBLE PRECISION NOT NULL,
        volume      DOUBLE PRECISION NOT NULL
    )
    """,
    "SELECT create_hypertable('ohlcv', by_range('time'), if_not_exists => TRUE)",
    "CREATE INDEX IF NOT EXISTS ohlcv_symbol_tf_time_idx ON ohlcv (symbol, timeframe, time DESC)",
    # --- order book snapshots --------------------------------------------- #
    """
    CREATE TABLE IF NOT EXISTS orderbook_snapshots (
        time        TIMESTAMPTZ NOT NULL,
        symbol      TEXT        NOT NULL,
        bids        JSONB       NOT NULL,
        asks        JSONB       NOT NULL
    )
    """,
    "SELECT create_hypertable('orderbook_snapshots', by_range('time'), if_not_exists => TRUE)",
    # --- predictions ------------------------------------------------------- #
    """
    CREATE TABLE IF NOT EXISTS predictions (
        time        TIMESTAMPTZ NOT NULL,
        symbol      TEXT        NOT NULL,
        timeframe   TEXT        NOT NULL,
        model_name  TEXT        NOT NULL,
        model_version TEXT      NOT NULL,
        prob        DOUBLE PRECISION,
        signal      TEXT,
        features    JSONB
    )
    """,
    "SELECT create_hypertable('predictions', by_range('time'), if_not_exists => TRUE)",
    # --- retention: compress candles older than 30 days --------------------- #
    "ALTER TABLE ohlcv SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol, timeframe')",
    """
    SELECT add_compression_policy('ohlcv', INTERVAL '30 days', if_not_exists => TRUE)
    """,
]
