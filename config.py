import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca API
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Risk Management
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "8"))
MAX_POSITIONS_PER_SECTOR = int(os.getenv("MAX_POSITIONS_PER_SECTOR", "2"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.02"))   # 2% per trade
STOP_LOSS_ATR_MULT = float(os.getenv("STOP_LOSS_ATR_MULT", "1.5"))
TAKE_PROFIT_ATR_MULT = float(os.getenv("TAKE_PROFIT_ATR_MULT", "2.5"))
MIN_RISK_REWARD = float(os.getenv("MIN_RISK_REWARD", "1.5"))

# Trailing stop — moves stop up as price rises
TRAILING_STOP_ENABLED = os.getenv("TRAILING_STOP_ENABLED", "true").lower() == "true"
TRAILING_BREAKEVEN_ATR = float(os.getenv("TRAILING_BREAKEVEN_ATR", "1.0"))   # move to BE at +1x ATR
TRAILING_LOCK_ATR = float(os.getenv("TRAILING_LOCK_ATR", "1.5"))             # lock +0.5 ATR at +1.5x ATR

# Partial profit taking — sell half at PARTIAL_ATR, let rest run
PARTIAL_PROFIT_ENABLED = os.getenv("PARTIAL_PROFIT_ENABLED", "true").lower() == "true"
PARTIAL_PROFIT_ATR = float(os.getenv("PARTIAL_PROFIT_ATR", "1.5"))           # take 50% off at +1.5x ATR

# Strategy thresholds
RSI_BUY_MAX = float(os.getenv("RSI_BUY_MAX", "45"))
RSI_BUY_MIN = float(os.getenv("RSI_BUY_MIN", "28"))
RSI_SELL = float(os.getenv("RSI_SELL", "68"))
STOCH_RSI_BUY_MAX = float(os.getenv("STOCH_RSI_BUY_MAX", "35"))   # Stoch RSI < 35 = oversold confirmation
VOLUME_SURGE_FACTOR = float(os.getenv("VOLUME_SURGE_FACTOR", "1.2"))
MIN_STOCK_PRICE = float(os.getenv("MIN_STOCK_PRICE", "10.0"))
MAX_HOLD_DAYS = int(os.getenv("MAX_HOLD_DAYS", "7"))

# Market regime — only go long when market is healthy
MARKET_REGIME_ENABLED = os.getenv("MARKET_REGIME_ENABLED", "true").lower() == "true"
REGIME_SYMBOL = os.getenv("REGIME_SYMBOL", "SPY")    # use SPY as market proxy
REGIME_EMA_PERIOD = int(os.getenv("REGIME_EMA_PERIOD", "200"))

# Scheduler
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))
HEALTH_PORT = int(os.getenv("PORT", "8080"))

# ── Watchlist: 65 liquid stocks across 10 sectors ────────────────────────────
# Sector tagging drives the MAX_POSITIONS_PER_SECTOR diversification rule.

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

    # Consumer Discretionary
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

    # ETFs (sector = etf, treated separately — max 2 ETF positions)
    "SPY": "etf", "QQQ": "etf", "IWM": "etf", "XLE": "etf", "XLF": "etf",
}

WATCHLIST: list[str] = list(SECTOR_MAP.keys())
