"""
Position sizing, trailing stops, and portfolio risk management.

Rules:
  - Never risk more than RISK_PER_TRADE_PCT of equity per trade
  - Maximum MAX_POSITIONS open simultaneously
  - Maximum MAX_POSITIONS_PER_SECTOR per sector (diversification)
  - Trailing stop moves to breakeven at +1x ATR, locks profit at +1.5x ATR
  - Total portfolio heat capped at 10%
"""

import logging
import config

logger = logging.getLogger(__name__)


def position_size(equity: float, entry_price: float, stop_loss: float) -> int:
    """
    Risk-based position sizing.
    Shares = (equity × RISK_PCT) / (entry - stop)
    Capped at 20% of portfolio value per position.
    """
    if entry_price <= 0 or stop_loss <= 0 or stop_loss >= entry_price:
        logger.warning("Invalid prices: entry=%.2f stop=%.2f", entry_price, stop_loss)
        return 0

    dollar_risk = equity * config.RISK_PER_TRADE_PCT
    risk_per_share = entry_price - stop_loss
    shares = int(dollar_risk / risk_per_share)

    # Hard cap: never more than 20% of portfolio in one trade
    max_shares = int((equity * 0.20) / entry_price)
    return max(min(shares, max_shares), 0)


def risk_reward_ok(entry: float, stop_loss: float, take_profit: float) -> bool:
    """Reject the trade if risk:reward is below the minimum threshold."""
    risk = entry - stop_loss
    if risk <= 0:
        return False
    rr = (take_profit - entry) / risk
    ok = rr >= config.MIN_RISK_REWARD
    if not ok:
        logger.debug("RR %.2f below minimum %.2f", rr, config.MIN_RISK_REWARD)
    return ok


def can_open_position(open_positions: int) -> bool:
    return open_positions < config.MAX_POSITIONS


def sector_ok(symbol: str, open_state: dict) -> bool:
    """
    Return True if adding this symbol won't exceed MAX_POSITIONS_PER_SECTOR.
    open_state: {symbol: {...}} for all currently open positions.
    """
    sector = config.SECTOR_MAP.get(symbol, "unknown")
    count = sum(
        1 for sym in open_state
        if config.SECTOR_MAP.get(sym, "unknown") == sector
    )
    ok = count < config.MAX_POSITIONS_PER_SECTOR
    if not ok:
        logger.info("SKIP %s — sector '%s' already has %d positions", symbol, sector, count)
    return ok


def update_trailing_stop(
    entry_price: float,
    current_price: float,
    current_stop: float,
    atr: float,
) -> float:
    """
    Ratchet the stop loss upward as the trade moves in our favour.
    Never moves the stop down.

    Levels:
      profit >= 1.0x ATR → move stop to breakeven (entry_price)
      profit >= 1.5x ATR → move stop to entry + 0.5x ATR (lock some gain)
    """
    if not config.TRAILING_STOP_ENABLED or atr <= 0:
        return current_stop

    profit = current_price - entry_price
    candidate = current_stop

    if profit >= config.TRAILING_LOCK_ATR * atr:
        candidate = round(entry_price + 0.5 * atr, 4)
    elif profit >= config.TRAILING_BREAKEVEN_ATR * atr:
        candidate = round(entry_price, 4)

    return max(current_stop, candidate)   # never move stop down


def portfolio_heat(positions: list[dict], equity: float) -> float:
    """Fraction of equity currently at risk across all open positions."""
    total_risk = sum(
        (p["entry_price"] - p["stop_loss"]) * p["qty"]
        for p in positions
        if p.get("stop_loss") and p.get("entry_price") and p.get("qty")
    )
    return total_risk / equity if equity > 0 else 0.0


def heat_ok(positions: list[dict], equity: float) -> bool:
    """Return True if total heat is below 10% of equity."""
    heat = portfolio_heat(positions, equity)
    if heat >= 0.10:
        logger.warning("Portfolio heat %.1f%% — no new trades", heat * 100)
        return False
    return True
