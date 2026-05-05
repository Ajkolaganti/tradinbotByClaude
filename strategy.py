"""
Multi-confirmation strategy: MACD + RSI + Stochastic RSI + 200 EMA regime filter.

Entry (ALL required):
  1. Market regime OK  — SPY above its 200 EMA (bull market only)
  2. Stock above 200 EMA — individual uptrend filter
  3. RSI(14) in 28-45   — oversold-recovering zone
  4. Stoch RSI(14) < 35 — secondary oversold confirmation
  5. MACD histogram crosses above 0 OR is positive & strengthening
  6. Price within 3% of EMA(21) — not parabolic
  7. Volume >= 1.2x 20-day average

Exit (ANY):
  - Trailing stop hit (updated dynamically, see risk.py)
  - Take profit: entry + 2.5x ATR
  - Partial exit: sell 50% at +1.5x ATR
  - RSI(14) > 68
  - MACD histogram crosses below 0
  - Held > MAX_HOLD_DAYS
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Literal

import config

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    action: Literal["BUY", "SELL", "HOLD"]
    symbol: str
    price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr: float = 0.0
    rsi: float = 0.0
    macd_hist: float = 0.0
    reason: str = ""


# ── Pure indicator functions (no external TA library needed) ──────────────────

def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stoch_rsi(close: pd.Series, rsi_window: int = 14, stoch_window: int = 14) -> pd.Series:
    """Stochastic RSI: where is the current RSI within its own N-period range."""
    rsi = _rsi(close, rsi_window)
    rsi_min = rsi.rolling(stoch_window).min()
    rsi_max = rsi.rolling(stoch_window).max()
    denom = (rsi_max - rsi_min).replace(0, np.nan)
    return ((rsi - rsi_min) / denom) * 100


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    h_l = high - low
    h_pc = (high - close.shift()).abs()
    l_pc = (low - close.shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.ewm(com=window - 1, adjust=False).mean()


# ── Indicator enrichment ──────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to an OHLCV dataframe."""
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    df["rsi"] = _rsi(close, 14)
    df["stoch_rsi"] = _stoch_rsi(close, 14, 14)
    df["ema21"] = _ema(close, 21)
    df["ema50"] = _ema(close, 50)
    df["ema200"] = _ema(close, 200)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(close)
    df["atr"] = _atr(high, low, close, 14)

    # Bollinger Bands (20, 2)
    df["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    df["vol_avg20"] = vol.rolling(20).mean()

    return df.dropna()


# ── Market regime ─────────────────────────────────────────────────────────────

def market_is_bullish(spy_df: pd.DataFrame) -> bool:
    """
    Return True when SPY is above its 200 EMA.
    In a bear market, mean-reversion buys get steamrolled — this filter
    eliminates the biggest source of losing trades.
    """
    if not config.MARKET_REGIME_ENABLED:
        return True
    if spy_df is None or spy_df.empty:
        logger.warning("No SPY data for regime check — assuming bullish")
        return True
    df = add_indicators(spy_df)
    if df.empty:
        return True
    cur = df.iloc[-1]
    bullish = float(cur["close"]) > float(cur["ema200"])
    logger.info("Market regime: SPY close=%.2f EMA200=%.2f → %s",
                cur["close"], cur["ema200"], "BULL" if bullish else "BEAR")
    return bullish


# ── Signal generation ─────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, symbol: str, bullish_market: bool = True) -> Signal:
    """Evaluate symbol data and return a Signal."""
    if len(df) < 210:   # need 200 bars for EMA200
        return Signal("HOLD", symbol, 0.0, reason="insufficient data (<210 bars)")

    df = add_indicators(df)
    if df.empty:
        return Signal("HOLD", symbol, 0.0, reason="indicators empty after dropna")

    cur = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(cur["close"])
    atr = float(cur["atr"])
    rsi = float(cur["rsi"])
    stoch_rsi = float(cur["stoch_rsi"])
    macd_hist = float(cur["macd_hist"])
    prev_macd_hist = float(prev["macd_hist"])
    ema200 = float(cur["ema200"])
    ema21 = float(cur["ema21"])
    vol_ratio = float(cur["volume"] / cur["vol_avg20"]) if cur["vol_avg20"] > 0 else 0.0

    # ── BUY conditions ─────────────────────────────────────────────────────────
    regime_ok = bullish_market                                        # 1. market filter
    above_ema200 = price > ema200 * 0.98                             # 2. stock uptrend (allow 2% below)
    rsi_ok = config.RSI_BUY_MIN <= rsi <= config.RSI_BUY_MAX        # 3. RSI oversold zone
    stoch_ok = stoch_rsi <= config.STOCH_RSI_BUY_MAX                 # 4. Stoch RSI confirmation
    macd_cross_up = (prev_macd_hist <= 0) and (macd_hist > 0)
    macd_strengthening = (macd_hist > 0) and (macd_hist > prev_macd_hist)
    macd_ok = macd_cross_up or macd_strengthening                    # 5. momentum building
    trend_ok = price <= ema21 * 1.03                                  # 6. not overextended
    vol_ok = vol_ratio >= config.VOLUME_SURGE_FACTOR                  # 7. volume surge
    price_ok = price >= config.MIN_STOCK_PRICE

    buy = regime_ok and above_ema200 and rsi_ok and stoch_ok and macd_ok and trend_ok and vol_ok and price_ok

    # ── SELL conditions ────────────────────────────────────────────────────────
    rsi_overbought = rsi > config.RSI_SELL
    macd_cross_down = (prev_macd_hist >= 0) and (macd_hist < 0)
    sell = rsi_overbought or macd_cross_down

    # ── Build signal ───────────────────────────────────────────────────────────
    stop_loss = round(price - config.STOP_LOSS_ATR_MULT * atr, 4)
    take_profit = round(price + config.TAKE_PROFIT_ATR_MULT * atr, 4)

    if buy:
        reason = (
            f"RSI={rsi:.1f} StochRSI={stoch_rsi:.1f} MACD_H={macd_hist:.4f} "
            f"VolRatio={vol_ratio:.2f} EMA200={ema200:.2f}"
        )
        return Signal("BUY", symbol, price, stop_loss, take_profit, atr, rsi, macd_hist, reason)

    if sell:
        reason = f"RSI={rsi:.1f} MACD_H={macd_hist:.4f} (exit signal)"
        return Signal("SELL", symbol, price, reason=reason)

    return Signal("HOLD", symbol, price, rsi=rsi, macd_hist=macd_hist, reason="no signal")


def check_exit(
    entry_price: float,
    current_bar: dict,
    days_held: int,
    symbol: str,
    trailing_stop: float,
    take_profit: float,
    half_sold: bool,
) -> tuple[str | None, str]:
    """
    Return (action, reason):
      action = 'SELL_ALL' | 'SELL_HALF' | None
    """
    price = float(current_bar["close"])
    rsi = float(current_bar.get("rsi", 50))
    macd_hist = float(current_bar.get("macd_hist", 0))
    prev_macd_hist = float(current_bar.get("prev_macd_hist", 0))
    atr = float(current_bar.get("atr", 0))
    partial_trigger = entry_price + config.PARTIAL_PROFIT_ATR * atr

    # Full exit triggers
    if trailing_stop > 0 and price <= trailing_stop:
        return "SELL_ALL", f"trailing stop hit @ {price:.2f} (stop={trailing_stop:.2f})"
    if take_profit > 0 and price >= take_profit:
        return "SELL_ALL", f"take profit hit @ {price:.2f} (tp={take_profit:.2f})"
    if rsi > config.RSI_SELL:
        return "SELL_ALL", f"RSI overbought ({rsi:.1f})"
    if (prev_macd_hist >= 0) and (macd_hist < 0):
        return "SELL_ALL", "MACD crossed below zero"
    if days_held >= config.MAX_HOLD_DAYS:
        return "SELL_ALL", f"max hold ({days_held}d) reached"

    # Partial exit trigger (50% off at +1.5x ATR)
    if config.PARTIAL_PROFIT_ENABLED and not half_sold and atr > 0 and price >= partial_trigger:
        return "SELL_HALF", f"partial profit @ {price:.2f} (+1.5xATR)"

    return None, ""
