"""
Earnings blackout filter — Priority 1.

Never enter a position within EARNINGS_BLACKOUT_DAYS of scheduled earnings.
Exit existing positions EARNINGS_EXIT_DAYS before earnings day.
Earnings gaps routinely blow past stop losses; this is the biggest
single source of catastrophic losses for swing traders.

Uses yfinance earnings_dates (free, no API key).
Cache is refreshed once per day to minimise network calls.
"""

import logging
from datetime import date

import yfinance as yf

logger = logging.getLogger(__name__)

# {symbol: [date, ...]}  — refreshed when _cache_day changes
_cache: dict[str, list[date]] = {}
_cache_day: date | None = None


def _refresh():
    global _cache_day
    today = date.today()
    if _cache_day == today:
        return
    _cache.clear()
    _cache_day = today


def _earnings_dates(symbol: str) -> list[date]:
    _refresh()
    if symbol in _cache:
        return _cache[symbol]

    dates: list[date] = []
    try:
        ticker = yf.Ticker(symbol)
        ed = ticker.earnings_dates
        if ed is not None and not ed.empty:
            today = date.today()
            for ts in ed.index:
                try:
                    d = ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])
                    if d >= today:
                        dates.append(d)
                except Exception:
                    pass
    except Exception as e:
        logger.debug("Earnings fetch failed for %s: %s", symbol, e)

    _cache[symbol] = dates
    return dates


def should_skip_entry(symbol: str, blackout_days: int = 5) -> bool:
    """Return True if earnings fall within the blackout window — skip new entry."""
    today = date.today()
    for d in _earnings_dates(symbol):
        if 0 <= (d - today).days <= blackout_days:
            logger.info("SKIP  %-6s — earnings %s (within %d-day blackout)", symbol, d, blackout_days)
            return True
    return False


def should_exit_before_earnings(symbol: str, exit_days: int = 1) -> bool:
    """Return True if we must exit an existing position before earnings gap risk."""
    today = date.today()
    for d in _earnings_dates(symbol):
        if 0 <= (d - today).days <= exit_days:
            logger.info("EXIT  %-6s — earnings %s (within %d-day exit window)", symbol, d, exit_days)
            return True
    return False
