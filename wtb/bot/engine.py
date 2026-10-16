"""Async trading engine.

Replaces the v0.0.1 synchronous ``while True: time.sleep(60)`` loop:
- fully async; one asyncio.Task per concern
- non-destructive symbol validation (cached, backoff — the v0.0.1 bot deleted
  every symbol from config.yaml during a DNS outage and exited)
- publishes live status to Redis (dashboard/API read it — no more
  bot_status.json handoff, no more bot-as-subprocess-of-Streamlit)
- graceful shutdown on SIGINT/SIGTERM
"""
from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Optional

from wtb.core.config import Settings
from wtb.core.logging import get_logger, setup_logging
from wtb.core.resilience import ExchangeUnavailableError
from wtb.core.statemachine import PositionLifecycle
from wtb.data.collector import Collector
from wtb.data.contracts import Side, Signal
from wtb.data.indicators import compute_order_book_features
from wtb.db.postgres import Engine, record_risk_event
from wtb.db.redis_client import RedisClient
from wtb.db.timescale import TimescaleStore
from wtb.execution.executor import Executor, PaperExecutor, build_executor
from wtb.exchanges.base import ExchangeAdapter
from wtb.ml.service import ModelService
from wtb.risk.manager import RiskManager, RiskViolation
from wtb.signals.generator import SignalGenerator

log = get_logger("engine")


class TradingEngine:
    def __init__(self, settings: Settings, redis: RedisClient | None = None) -> None:
        self.settings = settings
        self.redis = redis
        self.exchange: Optional[ExchangeAdapter] = None
        self.collector: Optional[Collector] = None
        self.risk: Optional[RiskManager] = None
        self.executor: Optional[Executor] = None
        self.models: dict[str, ModelService] = {}
        self.signal_gen = SignalGenerator(settings.signals)
        self.ts_store: Optional[TimescaleStore] = None
        self.pg: Optional[Engine] = None

        self._running = asyncio.Event()
        self._stop_requested = asyncio.Event()
        self._iteration = 0
        self._last_error: str | None = None
        self._started_at: float | None = None
        self._validated_symbols: dict[str, bool] = {}
        self._validation_time: dict[str, float] = {}

    # -------------------------------------------------------------- startup #
    async def start(self) -> None:
        s = self.settings
        setup_logging(
            level=s.app.log_level, log_dir=s.app.log_dir,
            json_console=not sys.stderr.isatty(),
        )

        # infrastructure (optional deps degrade gracefully)
        if s.database.redis_url:
            try:
                self.redis = self.redis or RedisClient(s.database.redis_url)
                await self.redis.connect()
            except Exception as exc:
                log.warning("redis_unavailable", error=str(exc))
                self.redis = None
        try:
            self.ts_store = TimescaleStore(s.database.timescale_dsn)
            await self.ts_store.connect()
            await self.ts_store.migrate()
        except Exception as exc:
            log.warning("timescale_unavailable", error=str(exc))
            self.ts_store = None
        try:
            self.pg = Engine(s.database.app_dsn)
            await self.pg.migrate()
        except Exception as exc:
            log.warning("postgres_unavailable", error=str(exc))
            self.pg = None

        # exchange
        ex_cfg = s.exchange()
        from wtb.exchanges.factory import get_exchange
        self.exchange = get_exchange(s.active_exchange, ex_cfg)

        # risk + execution
        self.risk = RiskManager(s.risk, s.trading.initial_balance)
        mode = "live" if s.trading.live_trading else "paper"
        if s.trading.live_trading and not ex_cfg.api_key:
            log.error("live_trading_blocked", reason="no api key configured")
            mode = "paper"
        self.executor = build_executor(mode, self.exchange, self.pg, ex_cfg.name, s.execution)
        log.info("executor_mode", mode=mode)

        # collector + history
        symbol_valid = await self.ensure_symbol_valid(s.trading.symbol)
        if symbol_valid:
            log.info("trading_symbol_validated", symbol=s.trading.symbol)
        else:
            log.warning("trading_symbol_not_validated",
                        symbol=s.trading.symbol,
                        hint="network noise is retried; a truly unknown symbol will never trade")
        self.collector = Collector(
            self.exchange, s.trading.symbol, s.trading.timeframe,
            data_dir=s.app.data_dir, max_points=s.data.max_data_points,
            timescale=self.ts_store, redis=self.redis,
            indicators=s.indicators, save_csv_snapshot=s.data.save_csv_snapshot,
            warmup_pad=s.data.warmup_pad,
        )
        self.collector.load_history()

        # model
        model = self.model_service()
        if not model.load_or_create() and self.collector.has_enough(s.ai.lookback):
            log.info("no_model_found_training")
            try:
                await asyncio.to_thread(model.train, self.collector)
            except Exception as exc:
                log.warning("initial_train_failed", error=str(exc))

        self._running.set()
        self._started_at = datetime.now(tz=timezone.utc).timestamp()
        log.info("engine_started", exchange=s.active_exchange, symbol=s.trading.symbol,
                 timeframe=s.trading.timeframe, mode=mode)

    async def stop(self) -> None:
        self._stop_requested.set()
        self._running.clear()
        try:
            if self.exchange:
                await self.exchange.close()
        finally:
            if self.ts_store:
                await self.ts_store.close()
            if self.pg:
                await self.pg.close()
            if self.redis:
                await self.redis.close()
        log.info("engine_stopped", iterations=self._iteration)

    def model_service(self) -> ModelService:
        s = self.settings
        key = s.trading.symbol
        if key not in self.models:
            self.models[key] = ModelService(
                s.ai, s.trading.symbol, s.trading.timeframe, s.app.model_dir
            )
        return self.models[key]

    async def retrain(self) -> dict:
        """Force a full model retrain on the collected history (API-triggered)."""
        s = self.settings
        model = self.model_service()
        metrics = await asyncio.to_thread(model.train, self.collector)
        log.info("manual_retrain_done", artifact=model.key, **{
            k: v for k, v in metrics.items() if k != "rows"})
        return metrics

    # ------------------------------------------------------- symbol caching #
    async def ensure_symbol_valid(self, symbol: str) -> bool:
        """Cached validation — at most one network call per TTL, never destructive.

        Unlike v0.0.1 (which deleted "invalid" symbols from config.yaml during a
        DNS outage), a failed validation is only a warning; data fetching is the
        real test.
        """
        now = asyncio.get_event_loop().time()
        ttl = self.settings.data.symbol_validation_ttl
        stamp = self._validation_time.get(symbol, -1e18)
        if now - stamp < ttl:
            return self._validated_symbols.get(symbol, False)
        assert self.exchange is not None
        try:
            valid = await self.exchange.validate_symbol(symbol)
        except Exception:
            valid = False
        self._validated_symbols[symbol] = valid
        self._validation_time[symbol] = now
        return valid

    def _apply_hot_config(self, s) -> None:
        """Hot-swappable sections are pushed into live components each iteration.

        Structural changes (symbol, timeframe, AI shape, execution mode) need a
        restart — the API surfaces this via Settings.restart_required().
        """
        if self.risk is not None and self.risk.cfg is not s.risk:
            self.risk.cfg = s.risk
        if self.signal_gen.cfg is not s.signals:
            self.signal_gen.cfg = s.signals
        if self.executor is not None and getattr(self.executor, "execution", None) is not s.execution:
            self.executor.execution = s.execution
        if self.collector is not None:
            self.collector.indicators = s.indicators
            self.collector.save_csv_snapshot = s.data.save_csv_snapshot
            self.collector._warmup_pad = s.data.warmup_pad

    # ----------------------------------------------------------- one loop i #
    async def iterate(self) -> None:
        assert self.collector is not None and self.risk is not None
        self._iteration += 1

        # hot-reload settings (dashboard/API may have changed them) and apply
        # the hot-swappable sections to live components
        from wtb.core.config import get_settings
        self.settings = s = get_settings()
        self._apply_hot_config(s)

        fresh = await self.collector.fetch_new_data(limit=s.trading.fetch_limit)
        if not self.collector.has_enough(s.ai.lookback):
            log.info("insufficient_data", rows=len(self.collector.frame()),
                     need=s.ai.lookback + s.data.warmup_pad)
            return

        price = self.collector.latest_price()
        assert price is not None

        # -- exit management first (SL/TP/trailing) -------------------------- #
        exit_reason = await self.executor.check_exits(price)
        if exit_reason:
            await self.executor.close_position(price, exit_reason, self.risk, s.trading.symbol)
            if s.notifications.enable and s.notifications.notify_on_sl_tp and \
                    self._notification_needed():
                await self._notify(f"{exit_reason.replace('_', ' ').title()} hit on "
                                   f"{s.trading.symbol} @ {price}")

        # -- features ------------------------------------------------------- #
        fdf = self.collector.feature_frame(s.ai.features)
        fdf = fdf.drop_nulls()
        if fdf.height < s.ai.lookback:
            return
        raw_features = fdf.select(s.ai.features).to_numpy()
        last = fdf.row(-1, named=True)

        # -- order book (optional) ------------------------------------------ #
        ob_features = None
        if s.data.use_order_book_features and self.exchange is not None:
            try:
                book = await self.exchange.fetch_order_book(s.trading.symbol, s.data.order_book_depth)
                ob_features = compute_order_book_features(book)
                price = (ob_features["best_bid"] + ob_features["best_ask"]) / 2
            except Exception as exc:
                log.debug("orderbook_unavailable", error=str(exc))

        # -- AI prediction --------------------------------------------------- #
        model = self.model_service()
        ai_prob: Optional[float] = None
        try:
            ai_prob = model.predict(raw_features)
        except Exception as exc:
            log.warning("predict_failed", error=str(exc))
        if ai_prob is None and self.collector.has_enough(s.ai.min_training_samples):
            try:
                await asyncio.to_thread(model.train, self.collector)
                ai_prob = model.predict(raw_features)
            except Exception as exc:
                log.debug("train_failed", error=str(exc))

        # -- fused signal ----------------------------------------------------- #
        bot_signal = self.signal_gen.generate(
            ai_prob=ai_prob, rsi=last.get("rsi", 50.0), bb_width=last.get("bb_width", 0.0),
            price=price, orderbook=ob_features,
        )
        if self.redis is not None:
            try:
                await self.redis.publish_signal(s.trading.symbol, bot_signal.model_dump())
            except Exception:
                pass

        # -- act --------------------------------------------------------------- #
        await self._act_on_signal(bot_signal, price)

        # -- online learning ---------------------------------------------------- #
        if s.ai.online_learning and (self._iteration % max(1, s.ai.retrain_interval) == 0):
            try:
                await asyncio.to_thread(self.model_service().train, self.collector)
            except Exception as exc:
                log.debug("scheduled_retrain_failed", error=str(exc))
        elif s.ai.online_learning and ai_prob is not None:
            await asyncio.to_thread(self.model_service().incremental_learn, self.collector)

        # -- status -------------------------------------------------------------- #
        await self._publish_status(price, bot_signal.model_dump(), ai_prob)

    async def _act_on_signal(self, bot_signal, price: float) -> None:
        s = self.settings
        assert self.executor is not None and self.risk is not None
        pos = getattr(self.executor, "position", None)
        is_open = pos is not None and pos.lifecycle.is_open

        if bot_signal.signal == Signal.HOLD:
            return
        try:
            self.risk.check_can_trade()
        except RiskViolation as exc:
            if self._iteration % 10 == 1:
                log.info("trading_halted", reason=str(exc))
                if self.settings.notifications.notify_on_risk_halt and self._notification_needed():
                    await self._notify(f"Trading halted: {exc}")
            if self.pg is not None and self._iteration % 10 == 1:
                async with self.pg.session_factory() as session:
                    await record_risk_event(session, kind=str(exc)[:32], detail=str(exc))
            return

        if not is_open and bot_signal.signal in (Signal.BUY, Signal.SELL):
            side = Side.BUY if bot_signal.signal == Signal.BUY else Side.SELL
            if side == Side.SELL and s.trading.live_trading:
                # spot exchanges cannot short; a SELL while flat is meaningless live
                log.debug("sell_ignored_flat", reason="spot live mode cannot short")
                return
            if side == Side.SELL and not s.execution.allow_short:
                log.debug("sell_ignored_flat", reason="allow_short is disabled")
                return
            opened = await self.executor.open_position(side, price, self.risk, s.trading.symbol)
            if opened and s.notifications.notify_on_open and self._notification_needed():
                await self._notify(f"Opened {side.value} {s.trading.symbol} @ {price}")
        elif is_open and (
            (pos.side == Side.BUY and bot_signal.signal == Signal.SELL)
            or (pos.side == Side.SELL and bot_signal.signal == Signal.BUY)
        ):
            pnl = await self.executor.close_position(price, "signal", self.risk, s.trading.symbol)
            if pnl is not None and s.notifications.notify_on_close and self._notification_needed():
                await self._notify(f"Closed {s.trading.symbol} @ {price} (PnL {pnl:+.4f})")

    def _notification_needed(self) -> bool:
        return (
            self.settings.notifications.enable
            and bool(self.settings.notifications.telegram_bot_token)
            and bool(self.settings.notifications.telegram_chat_id)
        )

    async def _notify(self, text: str) -> None:
        try:
            from wtb.core.config import get_settings
            n = get_settings().notifications
            if not (n.telegram_bot_token and n.telegram_chat_id):
                return
            import aiohttp

            url = f"https://api.telegram.org/bot{n.telegram_bot_token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={
                    "chat_id": n.telegram_chat_id, "text": text, "parse_mode": "HTML",
                }, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        log.warning("telegram_failed", status=resp.status)
        except Exception as exc:
            log.debug("telegram_error", error=str(exc))

    async def _publish_status(self, price: float, signal_data: dict, ai_prob: float | None) -> None:
        if self.redis is None:
            return
        pos = getattr(self.executor, "position", None)
        status: dict[str, Any] = {
            "running": True,
            "iteration": self._iteration,
            "mode": "live" if self.settings.trading.live_trading else "paper",
            "exchange": self.settings.active_exchange,
            "symbol": self.settings.trading.symbol,
            "timeframe": self.settings.trading.timeframe,
            "price": price,
            "balance": self.risk.state.balance if self.risk else None,
            "daily_loss": self.risk.state.daily_loss if self.risk else None,
            "consecutive_losses": self.risk.state.consecutive_losses if self.risk else None,
            "position": {
                "side": pos.side.value, "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "unrealized_pnl": pos.pnl(price),
            } if pos and pos.lifecycle.is_open else None,
            "signal": signal_data,
            "ai_prob": ai_prob,
            "last_error": self._last_error,
            "started_at": self._started_at,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
        }
        try:
            await self.redis.set_status(status)
        except Exception:
            pass

    # ------------------------------------------------------------ main loop #
    async def run_forever(self) -> None:
        await self.start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            except NotImplementedError:  # pragma: no cover (Windows)
                pass

        while not self._stop_requested.is_set():
            try:
                await self.iterate()
                self._last_error = None
            except ExchangeUnavailableError as exc:
                self._last_error = str(exc)
                log.warning("exchange_unavailable", error=str(exc))
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                log.exception("loop_error", error=str(exc))
            try:
                await asyncio.wait_for(
                    self._stop_requested.wait(),
                    timeout=self.settings.trading.loop_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass
        await self.stop()
