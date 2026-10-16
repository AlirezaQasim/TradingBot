"""Tests: config system (env expansion, hot reload, no-secrets-in-yaml)."""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def _env(monkeypatch):
    monkeypatch.setenv("NOBITEX_API_KEY", "test-key-123")


def _load():
    from dotenv import load_dotenv
    from wtb.core.config import DEFAULT_CONFIG_PATH, load_settings
    # override=False: the environment (monkeypatched) wins over the repo .env
    load_dotenv(DEFAULT_CONFIG_PATH.parent.parent / ".env", override=False)
    return load_settings(DEFAULT_CONFIG_PATH)


def test_config_loads_and_resolves_env(_env):
    s = _load()
    assert s.active_exchange == "nobitex"
    assert s.exchange("nobitex").api_key == "test-key-123"
    assert s.trading.live_trading is False
    assert s.risk.position_size_pct <= 1.0


def test_config_has_no_plaintext_secret():
    raw = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
           + "/config/config.yaml")
    content = open(raw).read()
    assert "q0tEo" not in content                       # the leaked v0.0.1 key
    assert "${NOBITEX_API_KEY}" in content              # placeholder, not the value


def test_symbol_uppercasing_and_dedup():
    from wtb.core.config import ExchangeConfig
    ex = ExchangeConfig(adapter="nobitex", symbols=["btcirt", "BTCIRT ", "", "ethirt"])
    assert ex.symbols == ["BTCIRT", "ETHIRT"]


def test_invalid_adapter_rejected():
    from wtb.core.config import ExchangeConfig
    with pytest.raises(Exception):
        ExchangeConfig(adapter="kraken-unknown")
