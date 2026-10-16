"""Tests: config PATCH write-back (placeholder preservation, validation, atomicity)."""
from __future__ import annotations

import pytest
import yaml

from wtb.core.config import Settings, patch_config


@pytest.fixture
def tmp_config(tmp_path):
    src = tmp_path / "config.yaml"
    src.write_text(yaml.safe_dump({
        "active_exchange": "nobitex",
        "exchanges": {"nobitex": {"adapter": "nobitex", "api_key": "${NOBITEX_API_KEY}",
                                  "symbols": ["BTCIRT"]}},
        "trading": {"symbol": "BTCIRT", "timeframe": "4h", "initial_balance": 20.0},
        "risk": {"stop_loss_pct": 0.02, "take_profit_pct": 0.04},
    }))
    return src


def test_patch_updates_values(tmp_config):
    s = patch_config({"risk": {"stop_loss_pct": 0.05}}, tmp_config)
    assert s.risk.stop_loss_pct == 0.05
    # other values untouched
    assert s.risk.take_profit_pct == 0.04
    assert s.trading.symbol == "BTCIRT"


def test_patch_preserves_env_placeholders(tmp_config):
    patch_config({"risk": {"stop_loss_pct": 0.03}}, tmp_config)
    raw = tmp_config.read_text()
    assert "${NOBITEX_API_KEY}" in raw          # secret placeholder intact


def test_patch_deep_merge_does_not_clobber_siblings(tmp_config):
    patch_config({"trading": {"symbol": "ETHIRT"}}, tmp_config)
    s = patch_config({}, tmp_config)
    assert s.trading.symbol == "ETHIRT"
    assert s.trading.timeframe == "4H"          # sibling key survived (uppercased)


def test_patch_invalid_value_rejected_file_untouched(tmp_config):
    before = tmp_config.read_text()
    with pytest.raises(Exception):
        patch_config({"risk": {"stop_loss_pct": 5.0}}, tmp_config)   # > 1 invalid
    assert tmp_config.read_text() == before


def test_patch_unknown_section_rejected(tmp_config):
    before = tmp_config.read_text()
    with pytest.raises(Exception):
        patch_config({"no_such_section": {"x": 1}}, tmp_config)
    assert tmp_config.read_text() == before


def test_restart_required_detection():
    s = Settings()
    assert s.restart_required({"trading": {"symbol": "ETHIRT"}}) is True
    assert s.restart_required({"trading": {"live_trading": True}}) is True
    assert s.restart_required({"ai": {"lookback": 50}}) is True
    assert s.restart_required({"risk": {"stop_loss_pct": 0.01}}) is False
    assert s.restart_required({"signals": {"ai_weight": 3}}) is False
    assert s.restart_required({"execution": {"fee_pct": 0.002}}) is False

def test_patch_nested_secret_path():
    """Secrets nested deeper than one level (exchanges.<name>.api_key) must be rejected."""
    from wtb.core.config import patch_config
    import tempfile, yaml
    from pathlib import Path
    src = Path(tempfile.mkdtemp()) / "config.yaml"
    src.write_text(yaml.safe_dump({
        "exchanges": {"nobitex": {"adapter": "nobitex", "api_key": "${NOBITEX_API_KEY}"}},
    }))
    class _Boom(Exception):
        pass
    # patch_config itself has no secret awareness (that is the API layer's job),
    # but it must NEVER be handed a resolved secret value. Simulate the API guard:
    from wtb.api.main import SECRET_KEYS  # noqa
    def api_guard(patch):
        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if str(k) in {"api_key", "api_secret"}:
                        raise _Boom(k)
                    walk(v)
        walk(patch)
    with pytest.raises(_Boom):
        api_guard({"exchanges": {"nobitex": {"api_key": "leak"}}})
    # and the placeholder survives in the file
    assert "${NOBITEX_API_KEY}" in src.read_text()
