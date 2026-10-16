#!/usr/bin/env python3
"""Streamlit control plane.

Talks to the FastAPI backend (status, start/stop, config read AND write,
metrics, retrain) — the bot is a supervised engine task behind the API, NOT a
Streamlit child process, and refreshes use autorefresh (no blocking sleeps).

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_URL = os.environ.get("WTB_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="WebTraderBot", page_icon="📈", layout="wide")
st.title("📈 WebTraderBot — Control Panel")
st.caption("Multi-exchange | Adaptive AI | Fault-tolerant async engine")

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    st_autorefresh(interval=5000, key="status_refresh")
except ImportError:
    if st.button("Refresh now"):
        st.rerun()


@st.cache_data(ttl=4)
def api_get(path: str, timeout: float = 5.0):
    try:
        r = httpx.get(f"{API_URL}{path}", timeout=timeout)
        return r.json()
    except Exception as exc:
        return {"error": str(exc)}


def api_post(path: str, body: dict | None = None) -> tuple[bool, str]:
    try:
        r = httpx.post(f"{API_URL}{path}", json=body, timeout=20)
        return r.is_success, str(r.json().get("detail", r.text)[:200])
    except Exception as exc:
        return False, str(exc)


def api_patch(path: str, body: dict) -> tuple[bool, str]:
    try:
        r = httpx.patch(f"{API_URL}{path}", json=body, timeout=20)
        return r.is_success, str(r.json().get("detail", r.text)[:300])
    except Exception as exc:
        return False, str(exc)


cfg = api_get("/config")
health = api_get("/health")
status = api_get("/status")
metrics = api_get("/metrics")
running = bool(status.get("running"))

# --------------------------------------------------------------- live status #
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if running:
        st.success(f"Engine RUNNING — iteration {status.get('iteration', '—')} ({status.get('mode', '?').upper()})")
    else:
        st.info("Engine STOPPED")
with col2:
    if not running:
        if st.button("▶ START", use_container_width=True, type="primary"):
            ok, msg = api_post("/bot/start")
            (st.toast if ok else st.error)(msg)
            st.rerun()
    else:
        if st.button("⏹ STOP", use_container_width=True):
            ok, msg = api_post("/bot/stop")
            (st.toast if ok else st.error)(msg)
            st.rerun()
with col3:
    st.metric("API", "online" if "error" not in health else "offline")

if "error" in status and not running:
    st.warning(f"API error: {status['error']} — start the API with `python scripts/run_api.py`")
elif not running and status.get("message"):
    st.caption(status["message"])

if running:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbol", status.get("symbol", "—"))
    c2.metric("Price", f"{status.get('price', 0):,.0f}")
    c3.metric("Balance", f"{status.get('balance', 0):,.2f}")
    sig = status.get("signal") or {}
    c4.metric("Signal", str(sig.get("signal", "—")).upper(),
              f"buy {sig.get('score_buy', 0)} / sell {sig.get('score_sell', 0)}")

    c5, c6, c7, c8 = st.columns(4)
    ai_prob = status.get("ai_prob")
    c5.metric("AI prob (up)", f"{ai_prob:.3f}" if ai_prob is not None else "no model")
    c6.metric("Daily loss", f"{status.get('daily_loss', 0):,.2f}")
    c7.metric("Loss streak", status.get("consecutive_losses", 0))
    pos = status.get("position")
    c8.metric("Position", f"{pos['side'].upper()} {pos['quantity']:.6f}" if pos else "flat",
              f"PnL {pos['unrealized_pnl']:+.2f}" if pos else None)

st.divider()

# ------------------------------------------------------------------- tabs    #
tab_cfg, tab_ai, tab_perf, tab_svc = st.tabs(
    ["⚙ Settings (editable)", "🤖 AI / ML", "📊 Performance", "🗄 Services"])


def section_default(section: str) -> dict:
    return cfg.get(section, {}) if isinstance(cfg, dict) else {}


def num(label: str, value, key, fmt: str = "%.4f", step: float | None = None):
    return st.number_input(label, value=float(value or 0), step=step or 10 ** -4,
                           format=fmt, key=key)


# ------------------------------------------------------------------ settings #
with tab_cfg:
    if "error" in cfg:
        st.error("Config unavailable — is the API running?")
    else:
        st.caption("Edit any value → **Save** applies it live via `PATCH /config`. "
                   "Fields marked 🔒 need an engine restart; the API tells you after saving.")
        risk = section_default("risk")
        ex = section_default("execution")
        sg = section_default("signals")
        tr = section_default("trading")
        no = section_default("notifications")

        with st.expander("🛡 Risk & Sizing", expanded=True):
            a, b, c = st.columns(3)
            r_sl = a.number_input("Stop loss %", 0.1, 50.0, value=float(risk.get("stop_loss_pct", .02)) * 100, step=0.1, key="r_sl")
            r_tp = b.number_input("Take profit %", 0.1, 100.0, value=float(risk.get("take_profit_pct", .04)) * 100, step=0.1, key="r_tp")
            r_ps = c.slider("Position size (% of balance)", 1, 100, value=int(float(risk.get("position_size_pct", .95)) * 100), key="r_ps")
            a2, b2, c2 = st.columns(3)
            r_dl = a2.number_input("Max daily loss", 0.0, value=float(risk.get("max_daily_loss", 10)), key="r_dl")
            r_cl = b2.number_input("Max consecutive losses", 1, 100, value=int(risk.get("max_consecutive_losses", 3)), key="r_cl")
            r_pt = c2.number_input("Profit target", 0.0, value=float(risk.get("profit_target", 50)), key="r_pt")
            r_mn = st.number_input("Max position notional (0 = unlimited)", 0.0, value=float(risk.get("max_position_notional") or 0), key="r_mn")

        with st.expander("💵 Execution", expanded=True):
            e1, e2, e3 = st.columns(3)
            e_fee = e1.number_input("Fee % per fill", 0.0, 5.0, value=float(ex.get("fee_pct", .001)) * 100, step=0.01, key="e_fee")
            e_slip = e2.number_input("Slippage % (paper)", 0.0, 5.0, value=float(ex.get("slippage_pct", .0005)) * 100, step=0.01, key="e_slip")
            e_ts_raw = ex.get("trailing_stop_pct")
            e_ts_on = st.checkbox("Trailing stop", value=e_ts_raw is not None, key="e_ts_on")
            e_ts = st.number_input("Trailing % behind best price", 0.1, 50.0,
                                   value=float(e_ts_raw or 0.01) * 100, step=0.1, key="e_ts")
            e4, e5 = st.columns(2)
            e_short = e4.checkbox("Allow simulated shorts (paper)", value=bool(ex.get("allow_short", True)), key="e_short")
            e_otype = e5.selectbox("Live order type", ["market", "limit"],
                                   index=0 if ex.get("order_type", "market") == "market" else 1, key="e_otype")

        with st.expander("🧭 Signal thresholds"):
            s1, s2, s3 = st.columns(3)
            s_aw = s1.number_input("AI weight", 0, 10, value=int(sg.get("ai_weight", 2)), key="s_aw")
            s_ww = s2.number_input("Wave weight", 0, 10, value=int(sg.get("wave_weight", 1)), key="s_ww")
            s_ow = s3.number_input("Order-book weight", 0, 10, value=int(sg.get("orderbook_weight", 1)), key="s_ow")
            t1, t2, t3 = st.columns(3)
            s_buy = t1.slider("AI buy threshold", 0.5, 0.99, float(sg.get("prob_buy_threshold", .6)), key="s_buy")
            s_sell = t2.slider("AI sell threshold", 0.01, 0.5, float(sg.get("prob_sell_threshold", .4)), key="s_sell")
            s_spread = t3.number_input("Max spread % (hold above)", 0.0, 5.0, value=float(sg.get("max_spread_pct", .005)) * 100, step=0.01, key="s_spread")
            t4, t5 = st.columns(2)
            s_ob = t4.slider("RSI overbought", 50, 100, int(sg.get("rsi_overbought", 70)), key="s_ob")
            s_os = t5.slider("RSI oversold", 0, 50, int(sg.get("rsi_oversold", 30)), key="s_os")

        with st.expander("📡 Order Book Endpoint"):
            ob_url = st.text_input("Order book API URL", value="https://apiv2.nobitex.ir", key="ob_url")
            if st.button("Update endpoint", key="ob_update"):
                st.session_state["orderbook_url"] = ob_url
                # Validation/status label
                valid = ("nobitex" in ob_url or "apiv2" in ob_url) and ob_url.startswith(("http://", "https://"))
                st.session_state["ob_status"] = "✅ Valid Nobitex endpoint" if valid else "❌ Invalid: must contain nobitex/apiv2"
                st.toast(st.session_state["ob_status"])
            status_label = st.session_state.get("ob_status", "⏸ Not checked")
            st.markdown(f"**Status:** {status_label}")

        with st.expander("🔐 Exchange API Keys (manual — session only)"):
            ak = st.text_input("API Key", type="password", key="gui_api_key")
            as_ = st.text_input("API Secret", type="password", key="gui_api_secret")
            url_ = st.text_input("Base URL", value="https://api.nobitex.ir", key="gui_url")
            if st.button("Set session keys", key="gui_set_keys"):
                st.session_state["gui_keys"] = {"api_key": ak, "api_secret": as_, "url": url_}
                st.toast("Keys saved in session (not persisted to .env/config)")

        with st.expander("🔔 Notifications & loop"):
            n_browser = st.checkbox("Enable browser notification", value=False, key="n_browser")
            n1, n2 = st.columns(2)
            n_on = n1.checkbox("Enable Telegram", value=bool(no.get("enable", True)), key="n_on")
            n_open = n2.checkbox("Notify on position open", value=bool(no.get("notify_on_open", True)), key="n_open")
            n3, n4 = st.columns(2)
            n_close = n3.checkbox("Notify on close", value=bool(no.get("notify_on_close", True)), key="n_close")
            n_sl = n4.checkbox("Notify on SL/TP", value=bool(no.get("notify_on_sl_tp", True)), key="n_sl")
            loop_i = st.number_input("Loop interval (seconds)", 10, 3600, value=int(tr.get("loop_interval_seconds", 60)), key="loop_i")

        if st.button("💾 Save settings", type="primary", use_container_width=True):
            patch = {
                "risk": {
                    "stop_loss_pct": round(r_sl / 100, 6),
                    "take_profit_pct": round(r_tp / 100, 6),
                    "position_size_pct": round(r_ps / 100, 4),
                    "max_daily_loss": r_dl,
                    "max_consecutive_losses": int(r_cl),
                    "profit_target": r_pt,
                    "max_position_notional": r_mn if r_mn > 0 else None,
                },
                "execution": {
                    "fee_pct": round(e_fee / 100, 6),
                    "slippage_pct": round(e_slip / 100, 6),
                    "trailing_stop_pct": round(e_ts / 100, 6) if e_ts_on else None,
                    "allow_short": bool(e_short),
                    "order_type": e_otype,
                },
                "signals": {
                    "ai_weight": int(s_aw), "wave_weight": int(s_ww),
                    "orderbook_weight": int(s_ow),
                    "prob_buy_threshold": float(s_buy),
                    "prob_sell_threshold": float(s_sell),
                    "max_spread_pct": round(s_spread / 100, 6),
                    "rsi_overbought": int(s_ob), "rsi_oversold": int(s_os),
                },
                "notifications": {
                    "enable": bool(n_on), "notify_on_open": bool(n_open),
                    "notify_on_close": bool(n_close), "notify_on_sl_tp": bool(n_sl),
                },
                "trading": {"loop_interval_seconds": int(loop_i)},
            }
            ok, msg = api_patch("/config", {"patch": patch})
            if ok:
                st.toast("Settings saved — engine hot-reloads on the next iteration")
                st.info("Saved. 🔒 structural changes (symbol/timeframe/AI shape/live mode) need STOP → START.")
            else:
                st.error(msg)

# ------------------------------------------------------------------ ai/ml    #
with tab_ai:
    ai = cfg.get("ai", {}) if isinstance(cfg, dict) else {}
    x, y = st.columns(2)
    x.text_input("Model type", value=ai.get("model_type", ""), disabled=True)
    x.number_input("Lookback", value=int(ai.get("lookback", 0)), disabled=True)
    x.number_input("Future steps", value=int(ai.get("future_steps", 0)), disabled=True)
    y.checkbox("Online learning", value=ai.get("online_learning", False), disabled=True)
    y.number_input("Retrain interval", value=int(ai.get("retrain_interval", 0)), disabled=True)
    y.checkbox("Prefer ONNX runtime", value=ai.get("prefer_onnx", True), disabled=True)
    st.divider()
    st.subheader("Manual retrain")
    st.caption("Full retrain on collected history (background task).")
    r1, r2 = st.columns(2)
    if r1.button("🔄 Retrain now", use_container_width=True):
        ok, msg = api_post("/bot/retrain")
        (st.toast if ok else st.error)(msg)
    rr = api_get("/bot/retrain/result")
    if rr.get("retraining"):
        st.info("Retraining in progress…")
    elif rr.get("result"):
        res = rr["result"]
        if "error" in res:
            st.error(f"Retrain failed: {res['error']}")
        else:
            st.success(f"Last retrain: loss {res.get('loss', '—')}, val_acc {res.get('val_acc', '—')}")

# ------------------------------------------------------------------ perf     #
with tab_perf:
    if "error" in metrics:
        st.error("Metrics unavailable (app DB down?)")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Closed trades", metrics.get("closed_trades", 0))
        m2.metric("Win rate", f"{metrics['win_rate']}%" if metrics.get("win_rate") is not None else "—")
        m3.metric("Total PnL", f"{metrics.get('total_pnl', 0):+.4f}")
        m4.metric("Open positions", metrics.get("open_positions", 0))
        m5, m6 = st.columns(2)
        m5.metric("Best trade", f"{metrics.get('best_trade', 0):+.4f}")
        m6.metric("Worst trade", f"{metrics.get('worst_trade', 0):+.4f}")

# ------------------------------------------------------------------ services #
with tab_svc:
    h = health
    deps = h.get("dependencies", {})
    d1, d2 = st.columns(2)
    d1.metric("Redis", deps.get("redis", "?"))
    d1.metric("Engine", "running" if h.get("engine_running") else "stopped")
    d2.metric("Retraining", "yes" if h.get("retraining") else "no")
    d2.metric("Environment", h.get("environment", "?"))
    st.caption(f"API docs: {API_URL}/docs · Logs: logs/webtraderbot.log (JSON, rotated)")
