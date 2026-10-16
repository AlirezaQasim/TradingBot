"""Signal generator: AI probability + technical "wave" + order-book votes.

Port of v0.0.1 ``trading_signal`` with every magic number moved to config.
"""
from __future__ import annotations

from wtb.core.config import SignalConfig
from wtb.core.logging import get_logger
from wtb.data.contracts import BotSignal, OrderBookFeatures, Signal

log = get_logger(__name__)


def wave_signal(rsi: float, bb_width: float, cfg: SignalConfig, ai_vote: Signal = Signal.HOLD) -> Signal:
    """RSI/Bollinger 'wave' heuristic (v0.0.1 logic, configurable).

    During a volatility squeeze (RSI mid-range + narrow bands) the wave vote
    echoes the AI vote — the original bot's "follow the AI in calm markets"
    behaviour.
    """
    if 40.0 <= rsi <= 60.0 and bb_width < cfg.bb_width_max:
        return ai_vote              # squeeze: follow the AI
    if rsi > cfg.rsi_overbought:
        return Signal.SELL
    if rsi < cfg.rsi_oversold:
        return Signal.BUY
    return Signal.HOLD


def orderbook_signal(ob: OrderBookFeatures | None, cfg: SignalConfig) -> Signal:
    if ob is None:
        return Signal.HOLD
    spoof_sell = ob.spoof_ratio_ask > cfg.spoof_ratio_threshold and \
        ob.volume_imbalance < -cfg.imbalance_threshold
    spoof_buy = ob.spoof_ratio_bid > cfg.spoof_ratio_threshold and \
        ob.volume_imbalance > cfg.imbalance_threshold
    if spoof_sell:
        return Signal.SELL
    if spoof_buy:
        return Signal.BUY
    return Signal.HOLD


class SignalGenerator:
    def __init__(self, cfg: SignalConfig) -> None:
        self.cfg = cfg

    def generate(
        self,
        ai_prob: float | None,
        rsi: float,
        bb_width: float,
        price: float,
        orderbook: OrderBookFeatures | None = None,
    ) -> BotSignal:
        c = self.cfg

        ai_vote = Signal.HOLD
        if ai_prob is not None:
            if ai_prob > c.prob_buy_threshold:
                ai_vote = Signal.BUY
            elif ai_prob < c.prob_sell_threshold:
                ai_vote = Signal.SELL

        wave = wave_signal(rsi, bb_width, c, ai_vote)
        book = orderbook_signal(orderbook, c)

        scores = {Signal.BUY: 0, Signal.SELL: 0, Signal.HOLD: 0}
        scores[ai_vote] += c.ai_weight
        scores[wave] += c.wave_weight
        scores[book] += c.orderbook_weight
        final = max(scores, key=lambda s: scores[s])  # type: ignore[arg-type]

        # Veto rules (v0.0.1): don't buy into overbought / sell into oversold
        reason = "vote"
        if final == Signal.BUY and rsi > c.rsi_overbought:
            final, reason = Signal.HOLD, f"veto: buy with rsi {rsi:.1f} > {c.rsi_overbought}"
        elif final == Signal.SELL and rsi < c.rsi_oversold:
            final, reason = Signal.HOLD, f"veto: sell with rsi {rsi:.1f} < {c.rsi_oversold}"
        elif orderbook is not None and orderbook.spread_pct > c.max_spread_pct:
            final, reason = Signal.HOLD, f"veto: spread {orderbook.spread_pct:.4f} > {c.max_spread_pct}"

        return BotSignal(
            signal=final,
            ai_prob=ai_prob,
            ai_vote=ai_vote,
            wave_vote=wave,
            orderbook_vote=book,
            score_buy=scores[Signal.BUY],
            score_sell=scores[Signal.SELL],
            price=price,
            reason=reason,
        )
