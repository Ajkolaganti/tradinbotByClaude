"""
Relative strength ranking — Priority 2.

Ranks every watchlist symbol by how much it has outperformed SPY over
RS_LOOKBACK_PERIOD days.  Only the top RS_TOP_N symbols are considered
for new long entries each cycle.

Why it matters: buying the strongest stocks in a bull market is
consistently more profitable than buying randomly from the watchlist.
A stock with RS > 1.5 (50% stronger than SPY) is in institutional
accumulation — far higher chance of continuing higher.
"""

import logging
import pandas as pd

import config

logger = logging.getLogger(__name__)


def rs_score(symbol_df: pd.DataFrame, spy_df: pd.DataFrame, period: int = 20) -> float:
    """
    Relative Strength = stock return / SPY return over `period` bars.
    > 1.0 → outperforming   < 1.0 → underperforming   0 = no data
    """
    if len(symbol_df) < period + 1 or len(spy_df) < period + 1:
        return 0.0
    try:
        stock_ret = float(symbol_df["close"].iloc[-1]) / float(symbol_df["close"].iloc[-(period + 1)]) - 1
        spy_ret = float(spy_df["close"].iloc[-1]) / float(spy_df["close"].iloc[-(period + 1)]) - 1
        return round(stock_ret / spy_ret, 4) if spy_ret != 0 else 1.0
    except Exception:
        return 0.0


def rank(
    bars: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    period: int | None = None,
    top_n: int | None = None,
) -> list[str]:
    """
    Return the top_n symbols ranked by RS score (descending).
    Symbols with score <= 0 (bad data or underperformance) are excluded
    unless top_n would leave us with too few candidates.
    """
    period = period or config.RS_LOOKBACK_PERIOD
    top_n = top_n or config.RS_TOP_N

    scores: list[tuple[float, str]] = []
    for symbol, df in bars.items():
        if symbol in ("SPY", config.REGIME_SYMBOL):
            continue
        if symbol not in config.SECTOR_MAP:
            continue   # skip sector ETFs from the scored list
        score = rs_score(df, spy_df, period)
        scores.append((score, symbol))

    scores.sort(reverse=True)
    ranked = [sym for _, sym in scores[:top_n]]

    if scores:
        logger.info(
            "RS top-5: %s",
            [(s, f"{sc:.2f}") for sc, s in scores[:5]],
        )
    return ranked


def all_scores(bars: dict[str, pd.DataFrame], spy_df: pd.DataFrame, period: int | None = None) -> dict[str, float]:
    """Return RS score for every symbol — used by the /api/signals endpoint."""
    period = period or config.RS_LOOKBACK_PERIOD
    return {
        sym: rs_score(df, spy_df, period)
        for sym, df in bars.items()
        if sym in config.SECTOR_MAP
    }
