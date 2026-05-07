"""
Historical pattern analysis — the "history matters" module.

Why stock history matters:
  1. Price memory — support/resistance levels where many past orders clustered
     act as walls and floors. Other traders see the same levels; it becomes
     self-fulfilling.
  2. Per-stock RSI behaviour — AAPL reliably bounces from RSI 30, TSLA often
     doesn't.  Backtesting each stock's own RSI history tells you which signals
     to trust on that specific ticker.
  3. Volume profile — the price with the most historical volume (Point of Control)
     acts as a magnet.  Fading away from it tends to snap back.
  4. 52-week context — a stock near its 52-week low + oversold RSI is a much
     higher-probability bounce than the same RSI reading at a 52-week high.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Support & Resistance ──────────────────────────────────────────────────────

def _swing_highs(close: pd.Series, window: int = 5) -> list[float]:
    out = []
    for i in range(window, len(close) - window):
        if float(close.iloc[i]) == float(close.iloc[i - window : i + window + 1].max()):
            out.append(float(close.iloc[i]))
    return out


def _swing_lows(close: pd.Series, window: int = 5) -> list[float]:
    out = []
    for i in range(window, len(close) - window):
        if float(close.iloc[i]) == float(close.iloc[i - window : i + window + 1].min()):
            out.append(float(close.iloc[i]))
    return out


def support_resistance(df: pd.DataFrame) -> dict:
    """Nearest support and resistance within 10% of current price."""
    close = df["close"]
    price = float(close.iloc[-1])
    band = 0.10

    resistance = sorted(h for h in _swing_highs(close) if price < h <= price * (1 + band))
    support = sorted((l for l in _swing_lows(close) if price * (1 - band) <= l < price), reverse=True)

    return {
        "nearest_support": support[0] if support else None,
        "nearest_resistance": resistance[0] if resistance else None,
        "support_levels": support[:3],
        "resistance_levels": resistance[:3],
    }


# ── 52-week context ───────────────────────────────────────────────────────────

def week52_context(df: pd.DataFrame) -> dict:
    close = df["close"]
    price = float(close.iloc[-1])
    window = close.tail(252)
    high52 = float(window.max())
    low52 = float(window.min())
    pct_from_high = (price - high52) / high52 * 100
    pct_from_low = (price - low52) / low52 * 100 if low52 > 0 else 0

    return {
        "high_52w": round(high52, 2),
        "low_52w": round(low52, 2),
        "pct_from_52w_high": round(pct_from_high, 1),
        "pct_from_52w_low": round(pct_from_low, 1),
        "near_52w_low": pct_from_low < 10,     # within 10% of 52w low — strong bounce zone
        "near_52w_high": pct_from_high > -5,   # within 5% of 52w high — breakout potential
    }


# ── Volume Profile (Point of Control) ────────────────────────────────────────

def volume_profile(df: pd.DataFrame, bins: int = 20) -> dict:
    """
    Find the price level traded with the most volume (Point of Control).
    Price gravitates toward high-volume zones — useful for target setting.
    """
    close = df["close"]
    vol = df["volume"]
    price_min = float(close.min())
    price_max = float(close.max())
    if price_max == price_min:
        return {"point_of_control": price_min, "price_above_poc": True, "poc_distance_pct": 0}

    edges = np.linspace(price_min, price_max, bins + 1)
    vol_by_bin = np.zeros(bins)
    for i in range(len(close)):
        p = float(close.iloc[i])
        v = float(vol.iloc[i])
        idx = min(int((p - price_min) / (price_max - price_min) * bins), bins - 1)
        vol_by_bin[idx] += v

    poc_idx = int(np.argmax(vol_by_bin))
    poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2
    current = float(close.iloc[-1])

    return {
        "point_of_control": round(poc, 2),
        "price_above_poc": current > poc,
        "poc_distance_pct": round((current - poc) / poc * 100, 1),
    }


# ── Per-stock RSI win rate (the most powerful history feature) ────────────────

def rsi_win_rate(df: pd.DataFrame, rsi_min: float = 28, rsi_max: float = 45, forward_days: int = 5) -> dict:
    """
    On this specific stock's history: when RSI was in [rsi_min, rsi_max],
    what % of the time was price higher `forward_days` bars later?

    A win rate >= 60% means this stock genuinely bounces from the oversold zone.
    Below 50% means the RSI signal is unreliable for this ticker — skip it.
    """
    from strategy import _rsi  # avoid circular import at module level

    close = df["close"]
    rsi_vals = _rsi(close, 14)
    wins = 0
    total = 0

    for i in range(len(rsi_vals) - forward_days - 1):
        r = rsi_vals.iloc[i]
        if pd.isna(r):
            continue
        if rsi_min <= r <= rsi_max:
            entry = float(close.iloc[i])
            exit_p = float(close.iloc[i + forward_days])
            total += 1
            if exit_p > entry:
                wins += 1

    return {
        "hist_rsi_win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
        "hist_sample_size": total,
        "hist_forward_days": forward_days,
        "hist_signal_reliable": (wins / total >= 0.55) if total >= 10 else True,
    }


# ── Combined ──────────────────────────────────────────────────────────────────

def full_analysis(df: pd.DataFrame) -> dict:
    """Run all historical analyses. Returns {} on thin data or errors."""
    if len(df) < 60:
        return {}
    try:
        return {
            **support_resistance(df),
            **week52_context(df),
            **volume_profile(df),
            **rsi_win_rate(df),
        }
    except Exception as e:
        logger.warning("History analysis error: %s", e)
        return {}
