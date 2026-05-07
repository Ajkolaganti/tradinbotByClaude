"""
Multi-confirmation strategy: MACD + RSI + StochRSI + 200 EMA regime filter.
Priorities implemented here: P3 (short selling), P4 (time filter + 15-min confirm), P5 (sector ETF).

Long entry (ALL required):
  1. Bull market   — SPY above 200 EMA
  2. Stock uptrend — price within 2% of its own 200 EMA
  3. RSI(14) 28-45 — oversold-recovering zone
  4. StochRSI < 35 — secondary oversold confirmation
  5. MACD histogram crosses above 0 OR positive & strengthening
  6. Price within 3% of EMA21 — not parabolic
  7. Volume >= 1.2× 20-day avg
  8. Sector ETF above its 20 EMA (optional, config-gated)
  9. Valid time-of-day window: 10-12 or 14-15 ET (optional, config-gated)
 10. 15-min intraday confirmation (optional, config-gated)

Short entry (ALL required, bear market only):
  1. Bear market — SPY below 200 EMA
  2. Stock below its 200 EMA
  3. RSI(14) > 60  — overbought / failed rally
  4. MACD histogram crosses below 0
  5. Price below EMA21 — downtrend confirmed
  6. Volume surge >= 1.2×
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class Signal:
    action: Literal["BUY", "SHORT", "SELL", "COVER", "HOLD"]
    symbol: str
    price: float
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr: float = 0.0
    rsi: float = 0.0
    macd_hist: float = 0.0
    reason: str = ""


# ── Pure indicator functions ──────────────────────────────────────────────────

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


def _stoch_rsi(close: pd.Series, rsi_w: int = 14, stoch_w: int = 14) -> pd.Series:
    rsi = _rsi(close, rsi_w)
    lo = rsi.rolling(stoch_w).min()
    hi = rsi.rolling(stoch_w).max()
    return ((rsi - lo) / (hi - lo).replace(0, np.nan)) * 100


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    line = fast_ema - slow_ema
    sig = _ema(line, signal)
    return line, sig, line - sig


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, w: int = 14) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=w - 1, adjust=False).mean()


# ── Indicator enrichment ──────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["rsi"] = _rsi(c, 14)
    df["stoch_rsi"] = _stoch_rsi(c)
    df["ema21"] = _ema(c, 21)
    df["ema50"] = _ema(c, 50)
    df["ema200"] = _ema(c, 200)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(c)
    df["atr"] = _atr(h, l, c, 14)
    df["bb_mid"] = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["vol_avg20"] = v.rolling(20).mean()

    return df.dropna()


# ── Market regime ─────────────────────────────────────────────────────────────

def market_is_bullish(spy_df: pd.DataFrame | None) -> bool:
    if not config.MARKET_REGIME_ENABLED:
        return True
    if spy_df is None or spy_df.empty:
        logger.warning("No SPY data — assuming bullish")
        return True
    df = add_indicators(spy_df)
    if df.empty:
        return True
    cur = df.iloc[-1]
    bull = float(cur["close"]) > float(cur["ema200"])
    logger.info("Regime: SPY %.2f vs EMA200 %.2f → %s", cur["close"], cur["ema200"], "BULL" if bull else "BEAR")
    return bull


# ── Time-of-day filter (P4) ───────────────────────────────────────────────────

def is_valid_trading_time() -> bool:
    """
    Allow trades from 10:00 AM to 3:30 PM ET.
    Skips the first 30 min (volatile open) and last 30 min (illiquid close).
    """
    if not config.TIME_FILTER_ENABLED:
        return True
    now = datetime.now(ET)
    minutes = now.hour * 60 + now.minute
    ok = (10 * 60) <= minutes <= (15 * 60 + 30)
    if not ok:
        logger.debug("Time filter: %02d:%02d ET outside 10:00-15:30 window", now.hour, now.minute)
    return ok


# ── Sector ETF confirmation (P5) ──────────────────────────────────────────────

def sector_etf_bullish(symbol: str, bars: dict[str, pd.DataFrame]) -> bool:
    """Return True if the symbol's sector ETF is above its 20-day EMA."""
    if not config.SECTOR_ETF_CONFIRMATION:
        return True
    sector = config.SECTOR_MAP.get(symbol, "unknown")
    etf = config.SECTOR_ETF_MAP.get(sector)
    if etf is None:
        return True
    df = bars.get(etf)
    if df is None or df.empty:
        return True   # no data → don't block
    close = df["close"]
    ema20 = _ema(close, 20).iloc[-1]
    ok = float(close.iloc[-1]) > float(ema20)
    if not ok:
        logger.info("SKIP %-6s — sector ETF %s below EMA20", symbol, etf)
    return ok


# ── 15-min intraday confirmation (P4) ─────────────────────────────────────────

def intraday_confirm_buy(df_15min: pd.DataFrame | None) -> bool:
    """
    Quick intraday check: 15-min RSI not overbought + price above EMA9.
    Returns True (don't block) when data is unavailable.
    """
    if not config.INTRADAY_CONFIRM_ENABLED:
        return True
    if df_15min is None or len(df_15min) < 15:
        return True
    close = df_15min["close"]
    rsi_val = float(_rsi(close, 14).iloc[-1])
    ema9 = float(_ema(close, 9).iloc[-1])
    price = float(close.iloc[-1])
    ok = rsi_val < 65 and price > ema9
    if not ok:
        logger.debug("15-min confirm failed: RSI=%.1f price=%.2f EMA9=%.2f", rsi_val, price, ema9)
    return ok


# ── Long signal ───────────────────────────────────────────────────────────────

def evaluate(
    df: pd.DataFrame,
    symbol: str,
    bullish_market: bool = True,
    bars: dict[str, pd.DataFrame] | None = None,
    df_15min: pd.DataFrame | None = None,
    hist_win_rate: float = 100.0,
) -> Signal:
    """Evaluate daily bars and return a long Signal."""
    if len(df) < 210:
        return Signal("HOLD", symbol, 0.0, reason="need 210 bars for EMA200")

    df = add_indicators(df)
    if df.empty:
        return Signal("HOLD", symbol, 0.0, reason="indicators empty")

    cur, prev = df.iloc[-1], df.iloc[-2]
    price = float(cur["close"])
    atr = float(cur["atr"])
    rsi = float(cur["rsi"])
    stoch = float(cur["stoch_rsi"])
    macd_h = float(cur["macd_hist"])
    prev_macd_h = float(prev["macd_hist"])
    ema200 = float(cur["ema200"])
    ema21 = float(cur["ema21"])
    vol_ratio = float(cur["volume"] / cur["vol_avg20"]) if cur["vol_avg20"] > 0 else 0.0

    # History check: skip if this stock historically doesn't bounce from RSI zone
    if hist_win_rate < config.HIST_WIN_RATE_MIN:
        return Signal("HOLD", symbol, price, rsi=rsi, macd_hist=macd_h,
                      reason=f"hist RSI win rate {hist_win_rate:.0f}% below {config.HIST_WIN_RATE_MIN:.0f}% threshold")

    # All buy conditions
    conds = {
        "regime": bullish_market,
        "ema200": price > ema200 * 0.98,
        "rsi": config.RSI_BUY_MIN <= rsi <= config.RSI_BUY_MAX,
        "stoch": stoch <= config.STOCH_RSI_BUY_MAX,
        "macd": (prev_macd_h <= 0 and macd_h > 0) or (macd_h > 0 and macd_h > prev_macd_h),
        "trend": price <= ema21 * 1.03,
        "vol": vol_ratio >= config.VOLUME_SURGE_FACTOR,
        "price": price >= config.MIN_STOCK_PRICE,
        "sector_etf": sector_etf_bullish(symbol, bars or {}),
        "time": is_valid_trading_time(),
        "intraday": intraday_confirm_buy(df_15min),
    }
    failed = [k for k, v in conds.items() if not v]

    if not failed:
        stop = round(price - config.STOP_LOSS_ATR_MULT * atr, 4)
        tp = round(price + config.TAKE_PROFIT_ATR_MULT * atr, 4)
        reason = f"RSI={rsi:.1f} StochRSI={stoch:.1f} MACD_H={macd_h:.4f} Vol={vol_ratio:.2f}x"
        return Signal("BUY", symbol, price, stop, tp, atr, rsi, macd_h, reason)

    # Sell trigger
    rsi_ob = rsi > config.RSI_SELL
    macd_down = (prev_macd_h >= 0) and (macd_h < 0)
    if rsi_ob or macd_down:
        return Signal("SELL", symbol, price, rsi=rsi, macd_hist=macd_h,
                      reason=f"RSI={rsi:.1f} MACD_H={macd_h:.4f}")

    return Signal("HOLD", symbol, price, rsi=rsi, macd_hist=macd_h,
                  reason=f"no signal — failed: {failed}")


# ── Short signal (P3 – bear market) ──────────────────────────────────────────

def evaluate_short(
    df: pd.DataFrame,
    symbol: str,
    bearish_market: bool = True,
) -> Signal:
    """Return a SHORT signal when bear-market conditions align."""
    if not config.SHORT_SELLING_ENABLED or not bearish_market:
        return Signal("HOLD", symbol, 0.0, reason="shorts disabled or bull market")
    if len(df) < 210:
        return Signal("HOLD", symbol, 0.0, reason="need 210 bars")

    df = add_indicators(df)
    if df.empty:
        return Signal("HOLD", symbol, 0.0, reason="indicators empty")

    cur, prev = df.iloc[-1], df.iloc[-2]
    price = float(cur["close"])
    atr = float(cur["atr"])
    rsi = float(cur["rsi"])
    macd_h = float(cur["macd_hist"])
    prev_macd_h = float(prev["macd_hist"])
    ema200 = float(cur["ema200"])
    ema21 = float(cur["ema21"])
    vol_ratio = float(cur["volume"] / cur["vol_avg20"]) if cur["vol_avg20"] > 0 else 0.0

    below_ema200 = price < ema200 * 1.02
    rsi_ob = rsi >= config.RSI_SHORT_MIN
    macd_cross_down = (prev_macd_h >= 0) and (macd_h < 0)
    below_ema21 = price < ema21
    vol_ok = vol_ratio >= config.VOLUME_SURGE_FACTOR
    price_ok = price >= config.MIN_STOCK_PRICE

    short = below_ema200 and rsi_ob and macd_cross_down and below_ema21 and vol_ok and price_ok
    if not short:
        return Signal("HOLD", symbol, price, rsi=rsi, macd_hist=macd_h, reason="no short signal")

    # For shorts: stop ABOVE entry, TP BELOW entry
    stop = round(price + config.STOP_LOSS_ATR_MULT * atr, 4)
    tp = round(price - config.TAKE_PROFIT_ATR_MULT * atr, 4)
    reason = f"SHORT RSI={rsi:.1f} MACD_H={macd_h:.4f} Vol={vol_ratio:.2f}x"
    return Signal("SHORT", symbol, price, stop, tp, atr, rsi, macd_h, reason)


# ── Exit checks ───────────────────────────────────────────────────────────────

def check_exit(
    entry_price: float,
    current_bar: dict,
    days_held: int,
    symbol: str,
    trailing_stop: float,
    take_profit: float,
    half_sold: bool,
) -> tuple[str | None, str]:
    """Check exit for a LONG position. Returns (action, reason) or (None, '')."""
    price = float(current_bar["close"])
    rsi = float(current_bar.get("rsi", 50))
    macd_h = float(current_bar.get("macd_hist", 0))
    prev_macd_h = float(current_bar.get("prev_macd_hist", 0))
    atr = float(current_bar.get("atr", 0))
    partial_trigger = entry_price + config.PARTIAL_PROFIT_ATR * atr

    if trailing_stop > 0 and price <= trailing_stop:
        return "SELL_ALL", f"trailing stop @ {price:.2f} (stop={trailing_stop:.2f})"
    if take_profit > 0 and price >= take_profit:
        return "SELL_ALL", f"take profit @ {price:.2f}"
    if rsi > config.RSI_SELL:
        return "SELL_ALL", f"RSI overbought ({rsi:.1f})"
    if (prev_macd_h >= 0) and (macd_h < 0):
        return "SELL_ALL", "MACD crossed below zero"
    if days_held >= config.MAX_HOLD_DAYS:
        return "SELL_ALL", f"max hold ({days_held}d)"
    if config.PARTIAL_PROFIT_ENABLED and not half_sold and atr > 0 and price >= partial_trigger:
        return "SELL_HALF", f"partial profit @ {price:.2f} (+{config.PARTIAL_PROFIT_ATR}×ATR)"

    return None, ""


def check_short_exit(
    entry_price: float,
    current_bar: dict,
    days_held: int,
    symbol: str,
    trailing_stop: float,
    take_profit: float,
    half_sold: bool,
) -> tuple[str | None, str]:
    """Check exit for a SHORT position (logic is inverted vs long)."""
    price = float(current_bar["close"])
    rsi = float(current_bar.get("rsi", 50))
    macd_h = float(current_bar.get("macd_hist", 0))
    prev_macd_h = float(current_bar.get("prev_macd_hist", 0))
    atr = float(current_bar.get("atr", 0))
    partial_trigger = entry_price - config.PARTIAL_PROFIT_ATR * atr

    # Short stop is ABOVE entry (loss if price rises)
    if trailing_stop > 0 and price >= trailing_stop:
        return "COVER_ALL", f"short stop @ {price:.2f} (stop={trailing_stop:.2f})"
    if take_profit > 0 and price <= take_profit:
        return "COVER_ALL", f"short TP @ {price:.2f}"
    if rsi < config.RSI_SHORT_EXIT:
        return "COVER_ALL", f"RSI oversold ({rsi:.1f}) — cover short"
    if (prev_macd_h <= 0) and (macd_h > 0):
        return "COVER_ALL", "MACD crossed above zero — cover short"
    if days_held >= config.MAX_HOLD_DAYS:
        return "COVER_ALL", f"max hold ({days_held}d)"
    if config.PARTIAL_PROFIT_ENABLED and not half_sold and atr > 0 and price <= partial_trigger:
        return "COVER_HALF", f"short partial profit @ {price:.2f}"

    return None, ""
