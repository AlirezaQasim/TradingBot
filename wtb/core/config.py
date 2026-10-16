"""Configuration system.

YAML file (``config/config.yaml``) + ``.env`` secrets, validated by Pydantic v2.

Design rules:
- Secrets live ONLY in ``.env`` (or the process environment); the YAML refers to
  them with ``${VAR_NAME}`` placeholders that are expanded at load time.
- The bot NEVER writes the config file back (the v0.0.1 bot destructively
  rewrote ``config.yaml`` on startup -- that is gone).
- Hot reload: :class:`ConfigWatcher` revalidates the YAML when its mtime changes.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import ClassVar, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value):
    """Recursively expand ${VAR} placeholders from the process environment."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Section models
# --------------------------------------------------------------------------- #
class _Strict(BaseModel):
    """Reject unknown sections/fields — typos fail fast instead of being ignored."""
    model_config = ConfigDict(extra="forbid")


class AppConfig(_Strict):
    name: str = "webtraderbot"
    environment: Literal["development", "production"] = "development"
    data_dir: Path = PROJECT_ROOT / "market_data"
    model_dir: Path = PROJECT_ROOT / "models"
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"


class ExchangeConfig(_Strict):
    adapter: Literal["nobitex", "wallex", "ccxt"]
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    name: str = ""
    base_currency: str = "USDT"
    symbols: list[str] = Field(default_factory=list)
    testnet: bool = False
    timeout_seconds: float = 30.0
    # Circuit breaker tuning (per exchange)
    breaker_fail_max: int = 5
    breaker_reset_timeout: float = 60.0

    @field_validator("symbols")
    @classmethod
    def _strip_symbols(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out = []
        for s in v:
            s = s.strip().upper()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out


class TradingConfig(_Strict):
    symbol: str = "BTCIRT"
    timeframe: str = "4h"
    initial_balance: float = 20.0
    live_trading: bool = False
    loop_interval_seconds: int = 60
    # Candles requested per collector fetch (exchange-dependent cap ~1000)
    fetch_limit: int = Field(default=1000, ge=50, le=1000)

    @field_validator("symbol", "timeframe")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class ExecutionConfig(_Strict):
    """Paper-fill realism and live-order behaviour."""
    # Exchange fee charged on each fill (0.001 = 0.1% taker fee)
    fee_pct: float = Field(default=0.001, ge=0.0, lt=0.05)
    # Simulated slippage applied to paper fills (0.0005 = 0.05%)
    slippage_pct: float = Field(default=0.0005, ge=0.0, lt=0.05)
    # Trailing stop: ratchets the stop as price moves in favour.
    # null = disabled; 0.01 = trail 1% behind the best price.
    trailing_stop_pct: Optional[float] = Field(default=None, gt=0.0, lt=0.5)
    # Paper mode only: allow simulated short positions on SELL signals.
    allow_short: bool = True
    # Live orders: market or limit (limit uses the last seen price)
    order_type: Literal["market", "limit"] = "market"


class RiskConfig(_Strict):
    stop_loss_pct: float = Field(default=0.02, gt=0.0, lt=1.0)
    take_profit_pct: float = Field(default=0.04, gt=0.0, lt=1.0)
    # Fraction of current balance committed per trade. Replaces the v0.0.1
    # absolute ``position_size`` that (100 > 20) made every trade impossible.
    position_size_pct: float = Field(default=0.95, gt=0.0, le=1.0)
    # Absolute account-currency loss per day after which trading halts.
    max_daily_loss: float = Field(default=10.0, gt=0.0)
    max_consecutive_losses: int = Field(default=3, ge=1)
    profit_target: float = Field(default=50.0, gt=0.0)
    # Hard cap on single-position notional (account currency); None = unlimited.
    max_position_notional: Optional[float] = None


class IndicatorsConfig(_Strict):
    """Technical indicator periods — shared by training and inference."""
    rsi_period: int = Field(default=14, ge=2)
    macd_fast: int = Field(default=12, ge=2)
    macd_slow: int = Field(default=26, ge=3)
    macd_signal: int = Field(default=9, ge=2)
    bb_period: int = Field(default=20, ge=5)
    bb_std: float = Field(default=2.0, gt=0.0)


class AIConfig(_Strict):
    model_type: Literal["lstm", "gru"] = "lstm"
    lookback: int = Field(default=100, ge=10)
    future_steps: int = Field(default=5, ge=1)
    features: list[str] = Field(default_factory=lambda: [
        "open", "high", "low", "close", "volume",
        "returns", "rsi", "macd", "macd_signal", "bb_width",
    ])
    hidden_units: list[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 32
    min_training_samples: int = 500
    retrain_interval: int = 100
    online_learning: bool = True
    online_batch_size: int = 200
    online_epochs: int = 5
    seed: int = 42
    # Exported ONNX model takes priority over the raw PyTorch weights when present.
    prefer_onnx: bool = True
    # Chronological train/validation split ratio (fraction held out)
    test_ratio: float = Field(default=0.2, gt=0.0, lt=0.5)


class SignalConfig(_Strict):
    ai_weight: int = 2
    wave_weight: int = 1
    orderbook_weight: int = 1
    # AI probability -> vote conversion thresholds
    prob_buy_threshold: float = 0.6
    prob_sell_threshold: float = 0.4
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bb_width_max: float = 0.03
    max_spread_pct: float = 0.005
    spoof_ratio_threshold: float = 0.7
    imbalance_threshold: float = 0.3


class DataConfig(_Strict):
    max_data_points: int = 10_000
    order_book_depth: int = 10
    use_order_book_features: bool = True
    save_orderbook_snapshots: bool = False
    # Duplicate symbol validation is cached; refreshed at most this often (s).
    symbol_validation_ttl: int = 3600
    # Extra candles required beyond the model lookback before trading starts.
    warmup_pad: int = Field(default=10, ge=0)
    # CSV history snapshots (offline fallback) — disable to save SD-card wear.
    save_csv_snapshot: bool = True


class DatabaseConfig(_Strict):
    timescale_dsn: str = "postgresql://wtb:wtb@localhost:5432/wtb_ts"
    app_dsn: str = "postgresql://wtb:wtb@localhost:5432/wtb_app"
    redis_url: str = "redis://localhost:6379/0"


class ApiConfig(_Strict):
    host: str = "127.0.0.1"
    port: int = 8000
    # CORS origins for external dashboards; empty = same-origin only
    cors_origins: list[str] = Field(default_factory=list)


class NotificationsConfig(_Strict):
    enable: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    # Which events trigger a Telegram message
    notify_on_open: bool = True
    notify_on_close: bool = True
    notify_on_sl_tp: bool = True
    notify_on_risk_halt: bool = True


class MlflowConfig(_Strict):
    # MLflow 3.x: the filesystem backend is deprecated — use sqlite
    tracking_uri: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    experiment: str = "webtraderbot"


class Settings(_Strict):
    app: AppConfig = AppConfig()
    active_exchange: str = "nobitex"
    exchanges: dict[str, ExchangeConfig] = Field(default_factory=dict)
    trading: TradingConfig = TradingConfig()
    execution: ExecutionConfig = ExecutionConfig()
    risk: RiskConfig = RiskConfig()
    indicators: IndicatorsConfig = IndicatorsConfig()
    ai: AIConfig = AIConfig()
    signals: SignalConfig = SignalConfig()
    data: DataConfig = DataConfig()
    database: DatabaseConfig = DatabaseConfig()
    api: ApiConfig = ApiConfig()
    notifications: NotificationsConfig = NotificationsConfig()
    mlflow: MlflowConfig = MlflowConfig()

    def exchange(self, name: Optional[str] = None) -> ExchangeConfig:
        name = (name or self.active_exchange).lower()
        if name not in self.exchanges:
            raise KeyError(f"Exchange '{name}' is not configured")
        return self.exchanges[name]

    # Fields whose change requires an engine restart (structural ML / broker state)
    RESTART_REQUIRED: ClassVar[set[tuple]] = {
        ("active_exchange",),
        ("trading", "symbol"), ("trading", "timeframe"),
        ("trading", "live_trading"), ("trading", "initial_balance"),
        ("ai", "model_type"), ("ai", "features"), ("ai", "lookback"),
        ("ai", "future_steps"), ("ai", "hidden_units"), ("ai", "prefer_onnx"),
        ("indicators", "rsi_period"), ("indicators", "macd_fast"),
        ("indicators", "macd_slow"), ("indicators", "macd_signal"),
        ("indicators", "bb_period"), ("indicators", "bb_std"),
    }

    def restart_required(self, patch: dict) -> bool:
        """True when a patch touches a field that cannot be hot-applied."""
        for section, values in patch.items():
            if not isinstance(values, dict):
                if (section,) in self.RESTART_REQUIRED:
                    return True
                continue
            for key in values:
                if (section, key) in self.RESTART_REQUIRED:
                    return True
        return False


# --------------------------------------------------------------------------- #
# Loading + validated write-back
# --------------------------------------------------------------------------- #
def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    return Settings.model_validate(_expand_env(raw))


def patch_config(patch: dict, path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Deep-merge ``patch`` into config.yaml, validate, write atomically.

    The file's ``${VAR}`` placeholders are preserved (secrets never leak into
    YAML). The merged config is validated BEFORE writing; on any error the
    file is untouched. Returns the newly effective settings.
    """
    path = Path(path)
    raw: dict = yaml.safe_load(path.read_text()) or {}

    def deep_merge(base: dict, update: dict) -> dict:
        for k, v in update.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    merged = deep_merge(raw, patch)
    Settings.model_validate(_expand_env(merged))       # raise before writing
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True))
    tmp.replace(path)
    return load_settings(path)


class ConfigWatcher:
    """Mtime-based hot reloader. Returns the same Settings until the file changes."""

    def __init__(self, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._mtime: float = -1.0
        self._settings: Optional[Settings] = None

    def get(self, force_reload: bool = False) -> Settings:
        with self._lock:
            try:
                mtime = self._path.stat().st_mtime
            except FileNotFoundError:
                mtime = -1.0
            if force_reload or self._settings is None or mtime != self._mtime:
                self._settings = load_settings(self._path)
                self._mtime = mtime
            return self._settings


_watcher: Optional[ConfigWatcher] = None


def get_settings() -> Settings:
    """Process-wide hot-reloading settings accessor."""
    global _watcher
    if _watcher is None:
        _watcher = ConfigWatcher()
    return _watcher.get()
