"""
Entry point for Railway deployment.

Starts:
  1. Flask health-check server on PORT (Railway requires an open HTTP port)
  2. TradingBot in a background daemon thread
"""

import threading
import logging
import os

from flask import Flask, jsonify
import config
from bot import TradingBot

logger = logging.getLogger(__name__)
app = Flask(__name__)

_bot_status = {"running": False, "error": None}


@app.route("/health")
def health():
    return jsonify({"status": "ok", "bot_running": _bot_status["running"]}), 200


@app.route("/")
def index():
    return jsonify({
        "name": "AlpacaTradingBot",
        "paper": config.ALPACA_PAPER,
        "max_positions": config.MAX_POSITIONS,
        "risk_per_trade_pct": config.RISK_PER_TRADE_PCT,
        "watchlist_size": len(config.WATCHLIST),
        "bot_running": _bot_status["running"],
    })


def _run_bot():
    try:
        bot = TradingBot()
        _bot_status["running"] = True
        bot.run()
    except Exception as e:
        _bot_status["running"] = False
        _bot_status["error"] = str(e)
        logger.exception("Bot crashed: %s", e)


if __name__ == "__main__":
    bot_thread = threading.Thread(target=_run_bot, daemon=True, name="trading-bot")
    bot_thread.start()

    port = config.HEALTH_PORT
    logger.info("Health server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
