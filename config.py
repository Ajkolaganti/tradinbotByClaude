import os
from dotenv import load_dotenv

load_dotenv()

# ── Alpaca API ────────────────────────────────────────────────────────────────
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# ── LLM ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Risk Management ───────────────────────────────────────────────────────────
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "8"))
MAX_POSITIONS_PER_SECTOR = int(os.getenv("MAX_POSITIONS_PER_SECTOR", "2"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.02"))
STOP_LOSS_ATR_MULT = float(os.getenv("STOP_LOSS_ATR_MULT", "1.5"))
TAKE_PROFIT_ATR_MULT = float(os.getenv("TAKE_PROFIT_ATR_MULT", "2.5"))
MIN_RISK_REWARD = float(os.getenv("MIN_RISK_REWARD", "1.5"))

# Trailing stop
TRAILING_STOP_ENABLED = os.getenv("TRAILING_STOP_ENABLED", "true").lower() == "true"
TRAILING_BREAKEVEN_ATR = float(os.getenv("TRAILING_BREAKEVEN_ATR", "1.0"))
TRAILING_LOCK_ATR = float(os.getenv("TRAILING_LOCK_ATR", "1.5"))

# Partial profit taking
PARTIAL_PROFIT_ENABLED = os.getenv("PARTIAL_PROFIT_ENABLED", "true").lower() == "true"
PARTIAL_PROFIT_ATR = float(os.getenv("PARTIAL_PROFIT_ATR", "1.5"))

# ── Strategy thresholds ───────────────────────────────────────────────────────
RSI_BUY_MAX = float(os.getenv("RSI_BUY_MAX", "45"))
RSI_BUY_MIN = float(os.getenv("RSI_BUY_MIN", "28"))
RSI_SELL = float(os.getenv("RSI_SELL", "68"))
STOCH_RSI_BUY_MAX = float(os.getenv("STOCH_RSI_BUY_MAX", "35"))
VOLUME_SURGE_FACTOR = float(os.getenv("VOLUME_SURGE_FACTOR", "1.2"))
MIN_STOCK_PRICE = float(os.getenv("MIN_STOCK_PRICE", "10.0"))
MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "7"))

# Short selling (bear market, inverted signals)
SHORT_SELLING_ENABLED = os.getenv("SHORT_SELLING_ENABLED", "true").lower() == "true"
RSI_SHORT_MIN = float(os.getenv("RSI_SHORT_MIN", "60"))  # overbought zone for shorts
RSI_SHORT_EXIT = float(os.getenv("RSI_SHORT_EXIT", "35"))

# ── Market Regime ─────────────────────────────────────────────────────────────
MARKET_REGIME_ENABLED = os.getenv("MARKET_REGIME_ENABLED", "true").lower() == "true"
REGIME_SYMBOL = os.getenv("REGIME_SYMBOL", "SPY")
REGIME_EMA_PERIOD = int(os.getenv("REGIME_EMA_PERIOD", "200"))

# ── Priority filters ──────────────────────────────────────────────────────────
# P1 – Earnings blackout
EARNINGS_BLACKOUT_DAYS = int(os.getenv("EARNINGS_BLACKOUT_DAYS", "5"))
EARNINGS_EXIT_DAYS = int(os.getenv("EARNINGS_EXIT_DAYS", "1"))

# P2 – Relative strength ranking
RS_LOOKBACK_PERIOD = int(os.getenv("RS_LOOKBACK_PERIOD", "20"))
RS_TOP_N = int(os.getenv("RS_TOP_N", "25"))

# P4 – Time-of-day filter (ET hours: 10-12 and 14-15)
TIME_FILTER_ENABLED = os.getenv("TIME_FILTER_ENABLED", "true").lower() == "true"

# P4 – 15-minute intraday confirmation
INTRADAY_CONFIRM_ENABLED = os.getenv("INTRADAY_CONFIRM_ENABLED", "true").lower() == "true"

# P5 – Sector ETF confirmation
SECTOR_ETF_CONFIRMATION = os.getenv("SECTOR_ETF_CONFIRMATION", "true").lower() == "true"

# History: skip entry if per-stock historical RSI win rate < threshold
HIST_WIN_RATE_MIN = float(os.getenv("HIST_WIN_RATE_MIN", "50.0"))

# ── Scheduler ─────────────────────────────────────────────────────────────────
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
HEALTH_PORT = int(os.getenv("PORT", "8080"))

# ── Sector ETF map ────────────────────────────────────────────────────────────
SECTOR_ETF_MAP: dict[str, str | None] = {
    "tech": "XLK",
    "finance": "XLF",
    "healthcare": "XLV",
    "energy": "XLE",
    "consumer": "XLY",
    "industrial": "XLI",
    "utilities": "XLU",
    "materials": "XLB",
    "realestate": "XLRE",
    "etf": None,
    "unknown": None,
}
SECTOR_ETFS: list[str] = [v for v in SECTOR_ETF_MAP.values() if v]

# ── Watchlist (65 liquid stocks, 10 sectors) ──────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "AMD": "tech",
    "GOOGL": "tech", "META": "tech", "AMZN": "tech", "TSLA": "tech",
    "CRM": "tech", "NFLX": "tech", "ORCL": "tech", "ADBE": "tech",
    "NOW": "tech", "SHOP": "tech", "UBER": "tech", "PLTR": "tech",
    # Finance
    "JPM": "finance", "BAC": "finance", "GS": "finance", "MS": "finance",
    "V": "finance", "MA": "finance", "WFC": "finance", "C": "finance",
    "AXP": "finance", "BLK": "finance",
    # Healthcare
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "ABBV": "healthcare", "MRNA": "healthcare", "REGN": "healthcare",
    "LLY": "healthcare", "TMO": "healthcare",
    # Energy
    "XOM": "energy", "CVX": "energy", "OXY": "energy", "SLB": "energy",
    "COP": "energy", "EOG": "energy",
    # Materials
    "FCX": "materials", "NEM": "materials", "GOLD": "materials",
    "LIN": "materials", "APD": "materials",
    # Consumer
    "DIS": "consumer", "SBUX": "consumer", "NKE": "consumer",
    "TGT": "consumer", "WMT": "consumer", "COST": "consumer",
    "HD": "consumer", "MCD": "consumer",
    # Industrials
    "CAT": "industrial", "DE": "industrial", "HON": "industrial",
    "GE": "industrial", "LMT": "industrial", "RTX": "industrial",
    "UPS": "industrial",
    # Utilities
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities",
    # Real Estate
    "AMT": "realestate", "PLD": "realestate", "SPG": "realestate",
    # Liquid ETFs
    "SPY": "etf", "QQQ": "etf", "IWM": "etf", "XLE": "etf", "XLF": "etf",
}

WATCHLIST: list[str] = list(SECTOR_MAP.keys())
