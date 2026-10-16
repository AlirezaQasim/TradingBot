"""FastAPI control plane.

- /health          liveness + dependency readiness (Kubernetes-style)
- /status          live bot state (from Redis — the engine publishes it)
- /config          read current validated settings (secrets masked)
- PATCH /config    hot-edit ANY section: validated, atomic, ${VAR}-preserving;
                   the engine applies hot-swappable fields on the next iteration
- /config/schema   every customizable field with type/default/description
- /candles         collected OHLCV from TimescaleDB
- /signals         recent signals from the Redis stream
- /metrics         trading performance from PostgreSQL
- /bot/start|stop|retrain   engine lifecycle + manual model retrain
- /ws/status       WebSocket stream of live status for the dashboard

The Streamlit dashboard talks to THIS app instead of spawning the bot as a
child process (v0.0.1 orphaned the bot whenever the browser session ended).
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from wtb.core.config import (
    DEFAULT_CONFIG_PATH,
    Settings,
    get_settings,
    load_settings,
    patch_config,
)
from wtb.core.logging import get_logger, setup_logging
from wtb.db.redis_client import RedisClient, signal_stream

log = get_logger("api")

_engine_task: asyncio.Task | None = None
_engine_ref: Any | None = None
_redis: RedisClient | None = None
_retrain_task: asyncio.Task | None = None
_retrain_result: dict | None = None

SECRET_KEYS = {"api_key", "api_secret", "telegram_bot_token", "telegram_chat_id",
               "password", "postgres_password"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis
    settings = get_settings()
    setup_logging(level=settings.app.log_level)
    try:
        _redis = RedisClient(settings.database.redis_url)
        await _redis.connect()
    except Exception as exc:
        log.warning("redis_unavailable", error=str(exc))
        _redis = None
    yield
    await stop_engine(force=True)
    if _redis is not None:
        await _redis.close()


app = FastAPI(title="WebTraderBot API", version="0.1.0", lifespan=lifespan)

# CORS is configurable (api.cors_origins) so external dashboards can attach
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().api.cors_origins or [],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
async def start_engine() -> None:
    global _engine_task, _engine_ref
    if _engine_task is not None and not _engine_task.done():
        raise HTTPException(status_code=409, detail="Engine already running")
    from wtb.bot.engine import TradingEngine

    engine = TradingEngine(get_settings(), redis=_redis)
    _engine_ref = engine
    _engine_task = asyncio.create_task(engine.run_forever())
    log.info("engine_start_requested")


async def stop_engine(force: bool = False) -> None:
    global _engine_task, _engine_ref
    if _engine_ref is not None:
        try:
            await asyncio.wait_for(_engine_ref.stop(), timeout=10)
        except Exception as exc:
            if not force:
                raise HTTPException(status_code=500, detail=str(exc))
    if _engine_task is not None:
        _engine_task.cancel()
        try:
            await _engine_task
        except (asyncio.CancelledError, Exception):
            pass
    _engine_task, _engine_ref = None, None


# --------------------------------------------------------------------------- #
@app.get("/health")
async def health() -> JSONResponse:
    settings = get_settings()
    deps: dict[str, str] = {}
    if _redis is not None:
        try:
            await _redis.client.ping()
            deps["redis"] = "ok"
        except Exception as exc:
            deps["redis"] = f"error: {exc}"
    else:
        deps["redis"] = "unavailable"
    running = _engine_task is not None and not _engine_task.done()
    body = {
        "status": "healthy" if running or "error" not in str(deps) else "degraded",
        "engine_running": running,
        "retraining": _retrain_task is not None and not _retrain_task.done(),
        "environment": settings.app.environment,
        "dependencies": deps,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }
    return JSONResponse(status_code=200, content=body)


@app.get("/status")
async def status() -> dict[str, Any]:
    if _redis is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    data = await _redis.get_status()
    if not data:
        return {"running": False, "message": "Engine has not published status yet"}
    return data


# ------------------------------------------------------------------ config #
def _mask_secrets(data: dict) -> dict:
    for ex in data.get("exchanges", {}).values():
        if ex.get("api_key"):
            ex["api_key"] = "***"
        if ex.get("api_secret"):
            ex["api_secret"] = "***"
    n = data.get("notifications", {})
    for k in ("telegram_bot_token", "telegram_chat_id"):
        if n.get(k):
            n[k] = "***"
    return data


@app.get("/config")
async def config() -> dict[str, Any]:
    return _mask_secrets(get_settings().model_dump(mode="json"))


@app.get("/config/schema")
async def config_schema() -> dict[str, Any]:
    """Self-documenting customization surface: every field, type, default."""
    schema = Settings.model_json_schema()
    sections = {}
    for name, prop in schema["properties"].items():
        ref = prop.get("$ref", "")
        if ref:
            defs = schema["$defs"][ref.split("/")[-1]]
            sections[name] = {
                "title": defs.get("title", name),
                "fields": {
                    k: {"type": v.get("type") or v.get("$ref", "ref"),
                        "default": v.get("default"),
                        "description": v.get("description", "")}
                    for k, v in defs.get("properties", {}).items()
                },
            }
        else:
            sections[name] = {"title": prop.get("title", name), "fields": {}}
    return {"sections": sections,
            "restart_required_fields": [list(p) for p in Settings.RESTART_REQUIRED]}


class ConfigPatch(BaseModel):
    """Arbitrary partial config: {"section": {field: value, ...}, ...}."""
    patch: dict[str, Any]


def _reject_secrets(node: Any, path: str = "patch") -> None:
    """Walk the patch tree at ANY depth — secrets belong in .env, never in YAML."""
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key) in SECRET_KEYS:
                raise HTTPException(422, detail=(
                    f"'{path}.{key}' is a secret — set it in the .env file, "
                    "never via the API"))
            _reject_secrets(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _reject_secrets(value, f"{path}[{i}]")


@app.patch("/config")
async def update_config(body: ConfigPatch) -> dict[str, Any]:
    patch = body.patch
    _reject_secrets(patch)
    for section, values in patch.items():
        if not isinstance(values, dict):
            raise HTTPException(422, detail=f"Section '{section}' must be an object")
    try:
        new_settings = patch_config(patch, DEFAULT_CONFIG_PATH)
    except ValidationError as exc:
        raise HTTPException(422, detail=f"Invalid config: {exc.errors()[:3]}")
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))
    log.info("config_patched", sections=list(patch.keys()))
    return {"detail": "config updated (hot-reloaded by the engine)",
            "restart_required": load_settings(DEFAULT_CONFIG_PATH).restart_required(patch),
            "applied": _mask_secrets(new_settings.model_dump(mode="json"))}


# --------------------------------------------------------------------- data #
@app.get("/candles")
async def candles(symbol: str | None = None, timeframe: str | None = None,
                  limit: int = 200) -> dict[str, Any]:
    from wtb.db.timescale import TimescaleStore

    s = get_settings()
    symbol = (symbol or s.trading.symbol).upper()
    timeframe = (timeframe or s.trading.timeframe).lower()
    store = TimescaleStore(s.database.timescale_dsn)
    try:
        await store.connect()
        rows = await store.load_candles(symbol, timeframe, min(limit, 5000))
        return {"symbol": symbol, "timeframe": timeframe,
                "candles": [c.model_dump() for c in rows]}
    except Exception as exc:
        raise HTTPException(503, detail=f"TimescaleDB unavailable: {exc}")
    finally:
        await store.close()


@app.get("/signals")
async def signals(symbol: str | None = None, limit: int = 20) -> dict[str, Any]:
    import json as _json

    if _redis is None:
        raise HTTPException(503, detail="Redis unavailable")
    s = get_settings()
    symbol = (symbol or s.trading.symbol).upper()
    entries = await _redis.client.xrevrange(signal_stream(symbol), count=min(limit, 1000))
    out = []
    for _id, fields in entries:
        try:
            out.append(_json.loads(fields.get("data", "{}")))
        except Exception:
            continue
    return {"symbol": symbol, "signals": out}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Trading performance summary from the app database."""
    from sqlalchemy import select

    from wtb.db.postgres import Engine, Position

    s = get_settings()
    pg = Engine(s.database.app_dsn)
    try:
        async with pg.session_factory() as session:
            rows = (await session.execute(
                select(Position).order_by(Position.opened_at.desc()).limit(500)
            )).scalars().all()
        closed = [p for p in rows if p.state == "closed" and p.pnl is not None]
        wins = [p for p in closed if p.pnl > 0]
        return {
            "open_positions": sum(1 for p in rows if p.state == "open"),
            "closed_trades": len(closed),
            "win_rate": round(len(wins) / len(closed) * 100, 2) if closed else None,
            "total_pnl": round(sum(p.pnl for p in closed), 6),
            "avg_pnl": round(sum(p.pnl for p in closed) / len(closed), 6) if closed else None,
            "best_trade": round(max((p.pnl for p in closed), default=0.0), 6),
            "worst_trade": round(min((p.pnl for p in closed), default=0.0), 6),
        }
    finally:
        await pg.close()


# --------------------------------------------------------------------- bot  #
@app.post("/bot/start")
async def bot_start() -> dict[str, str]:
    await start_engine()
    return {"detail": "engine starting"}


@app.post("/bot/stop")
async def bot_stop() -> dict[str, str]:
    await stop_engine()
    return {"detail": "engine stopped"}


@app.post("/bot/retrain")
async def bot_retrain() -> dict[str, str]:
    """Retrain the model in the background; poll /health for 'retraining'."""
    global _retrain_task, _retrain_result
    if _engine_ref is None:
        raise HTTPException(409, detail="Engine not running — start it first")
    if _retrain_task is not None and not _retrain_task.done():
        raise HTTPException(409, detail="Retrain already in progress")
    _retrain_result = None

    async def _run() -> None:
        global _retrain_result
        try:
            _retrain_result = await _engine_ref.retrain()
        except Exception as exc:
            _retrain_result = {"error": str(exc)}

    _retrain_task = asyncio.create_task(_run())
    return {"detail": "retrain started — poll /health 'retraining' and /bot/retrain/result"}


@app.get("/bot/retrain/result")
async def retrain_result() -> dict[str, Any]:
    running = _retrain_task is not None and not _retrain_task.done()
    return {"retraining": running, "result": _retrain_result}


# -------------------------------------------------------------------- ws    #
@app.websocket("/ws/status")
async def ws_status(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            if _redis is None:
                await ws.send_json({"running": False, "error": "redis unavailable"})
            else:
                data = await _redis.get_status()
                await ws.send_json(data or {"running": False})
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return
    except Exception:
        return
