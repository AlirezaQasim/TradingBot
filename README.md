# WebTraderBot v0.1.0

Production-grade, fault-tolerant **AI crypto trading bot** for the Iranian market
(Nobitex, Wallex) and global exchanges (Binance, Bybit via ccxt) — rebuilt from the
v0.0.1 prototype on an async, strictly-typed, observable stack.

**Runs on a Raspberry Pi (aarch64) with system Python — no venv required.**

## The idea

A deep-learning model (LSTM/GRU, PyTorch, exported to ONNX) predicts whether the
price N steps ahead will be higher. Its probability is fused — weighted vote —
with a technical "wave" signal (RSI + Bollinger width) and order-book
microstructure signals (spoofing ratio, volume imbalance). A risk manager gates
every decision (position sizing as % of balance, daily-loss circuit breaker,
consecutive-loss breaker, profit target, kill switch), and an executor places
paper trades (default) or real orders (opt-in).

## Architecture

```
┌─────────────┐   HTTPS/REST   ┌──────────────────┐
│  Exchanges  │◄──────────────►│ Async adapters    │  aiohttp / ccxt.async_support
│ (Nobitex,   │                │ + circuit breakers│  (tenacity retries, pybreaker)
│  Binance…)  │                └────────┬─────────┘
└─────────────┘                         │ validated Pydantic contracts
                                        ▼
                            ┌─────────────────────┐
                            │   Redis Streams     │  event bus (decouples
                            └──────────┬──────────┘  ingestion from inference)
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
           ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
           │ TimescaleDB  │   │  Indicators  │   │   Polars     │
           │  (hypertables│   │  RSI/MACD/BB │   │  frames      │
           │  + compression)  └──────┬───────┘   └──────────────┘
           └──────────────┘          │
                                     ▼
                          ┌────────────────────┐
                          │ ONNX Runtime /     │  direction probability
                          │ PyTorch LSTM/GRU   │
                          └────────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │ Signal generator   │  weighted ensemble
                          └────────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │ Risk manager       │  sizing, breakers, idempotency
                          └────────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │ Executor           │  paper (default) / live (gated)
                          └────────┬───────────┘
                                   ▼
                          ┌────────────────────┐
                          │ FastAPI + Streamlit│  /health /status /ws + dashboard
                          └────────────────────┘
```

### Layout

| Path | Purpose |
|---|---|
| `wtb/core/` | config (Pydantic v2, YAML + .env, hot reload), structlog logging, tenacity/pybreaker resilience, position FSM |
| `wtb/exchanges/` | async adapters: `nobitex.py`, `wallex.py`, `ccxt_async.py`, factory |
| `wtb/data/` | contracts (Candle/OrderBook/Signal), async collector (dedup → Timescale + Redis Streams), Polars indicators |
| `wtb/ml/` | PyTorch LSTM/GRU, leakage-free feature pipeline, Optuna-ready training, ONNX export + runtime, MLflow tracking |
| `wtb/signals/` | weighted ensemble (AI 2 : wave 1 : order-book 1) with veto rules |
| `wtb/risk/` | % position sizing, daily-loss/streak/profit breakers, kill switch |
| `wtb/execution/` | `PaperExecutor` (default) and `LiveExecutor` (spot orders, gated) |
| `wtb/db/` | TimescaleDB (asyncpg hypertables), PostgreSQL ORM (SQLAlchemy 2.0), Redis client |
| `wtb/bot/engine.py` | async engine loop, hot config reload, graceful shutdown |
| `wtb/api/` | FastAPI: `/health`, `/status`, `/config`, `/bot/start|stop`, `/ws/status` |
| `wtb/workers/` | Celery (Redis broker): retrain, daily report |
| `dashboard/app.py` | Streamlit control panel (talks to the API, not a child process) |
| `legacy/` | archived v0.0.1 monolith (reference) |

## Setup (Raspberry Pi / Debian, system Python, no venv)

```bash
# services
sudo apt install -y redis-server postgresql
# TimescaleDB (apt repo supports arm64; see timescaledb docs) — optional,
# the bot degrades gracefully to CSV-only history without it

# python deps
pip install -r requirements.txt

# secrets — NEVER commit .env
cp .env.example .env        # then fill NOBITEX_API_KEY, TELEGRAM_*, POSTGRES_PASSWORD

# databases
sudo -u postgres psql -c "CREATE USER wtb WITH PASSWORD '<pw>';"
sudo -u postgres psql -c "CREATE DATABASE wtb_ts OWNER wtb;"
sudo -u postgres psql -c "CREATE DATABASE wtb_app OWNER wtb;"
python scripts/migrate.py   # hypertables + ORM tables
```

## Run

```bash
python scripts/run_api.py                    # FastAPI control plane (port 8000)
python scripts/run_bot.py                    # engine (paper mode by default)
# or start/stop the engine via API:  POST /bot/start  /bot/stop

streamlit run dashboard/app.py               # UI (reads the API)

# background workers (retraining, reports)
celery -A wtb.workers.tasks worker --loglevel=INFO --concurrency=1
```

- **Swagger docs:** http://127.0.0.1:8000/docs
- **Health:** http://127.0.0.1:8000/health
- Logs: `logs/webtraderbot.log` (JSON, rotated 10 MB × 5)

## Enabling live trading (when you are ready)

1. Set `trading.live_trading: true` in `config/config.yaml` (hot-reloaded).
2. Ensure `NOBITEX_API_KEY` has trading permission; the executor only supports
   **spot** (shorts remain a paper-only simulation).
3. Every order carries an idempotency key and goes through the exchange circuit
   breaker. Orders are recorded in `wtb_app.orders`.

## Tests

```bash
python -m pytest tests/ -q
```

Covers: data contracts (bad payloads rejected), indicators (RSI zero-division,
reference values), risk breakers, position FSM, ensemble vetoes, paper execution
math, config/secrets hygiene.

## Fixed from v0.0.1

| Issue in v0.0.1 | Fix |
|---|---|
| `position_size: 100` vs balance `20` — no trade could ever open | sizing as % of balance |
| `create_order` dead code — trading was 100% simulated silently | explicit paper/live modes, live gated by config |
| Live Nobitex key committed in `config.yaml` | `.env` + `${VAR}` placeholders (rotate the old key!) |
| Bot destructively rewrote `config.yaml` at startup | config is read-only to the bot, hot-reload via mtime |
| MinMaxScaler fit on the full dataset (look-ahead leakage) | scaler fit on train split only |
| Blocking `requests` + `time.sleep(60)` loop | asyncio end-to-end, graceful shutdown |
| O(n²) dedup + full CSV rewrite per cycle | O(1) timestamp set, idempotent Timescale upserts |
| QuestDB layer dead (wrong port, never called) | real TimescaleDB hypertables + retention policy |
| Dashboard spawned bot as child process (orphans) | engine runs behind FastAPI, supervised |
| DNS outage at startup deleted all symbols from config | cached validation with TTL, network errors ≠ bad symbols |
| RSI produced `inf` on zero-loss windows | Wilder smoothing with epsilon |
| No tests, no git, no manifest | pytest suite, git history, pinned requirements |
