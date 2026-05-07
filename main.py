"""
Entry point — Flask API server + trading bot thread.

Endpoints:
  GET  /health            — Railway health check
  GET  /api/status        — bot + account overview
  GET  /api/positions     — open positions with live P&L
  GET  /api/trades        — last 50 closed trades
  GET  /api/signals       — last scan results for all watchlist symbols
  GET  /api/account       — equity, cash, daily P&L
  POST /api/chat          — AI chatbot (natural language trade commands)
  GET  /api/predict/:sym  — LLM stock direction prediction
"""

import logging
import threading

from flask import Flask, jsonify, request

# flask-cors is optional — CORS headers added manually if not installed
try:
    from flask_cors import CORS
    _has_cors = True
except ImportError:
    _has_cors = False

import config

logger = logging.getLogger(__name__)

app = Flask(__name__)
if _has_cors:
    CORS(app)


# Manual CORS fallback so the UI always works even without flask-cors
@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/", defaults={"path": ""}, methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def _preflight(path):
    return jsonify({}), 200


# ── Bot thread ────────────────────────────────────────────────────────────────

_bot = None           # TradingBot instance, set once thread starts
_bot_running = False
_bot_error: str | None = None
_chat_history: list[dict] = []


def _run_bot():
    global _bot, _bot_running, _bot_error
    try:
        # Lazy import — keeps module-level imports minimal so Flask always starts
        from bot import TradingBot
        _bot = TradingBot()
        _bot_running = True
        _bot.run()
    except Exception as e:
        _bot_running = False
        _bot_error = str(e)
        logger.exception("Bot crashed: %s", e)


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "bot_running": _bot_running}), 200


# ── Status ────────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/api/status")
def status():
    acct: dict = {}
    if _bot:
        try:
            a = _bot.trade_client.get_account()
            acct = {
                "equity": float(a.equity),
                "cash": float(a.cash),
                "buying_power": float(a.buying_power),
                "daily_pnl": round(float(a.equity) - float(a.last_equity), 2),
            }
        except Exception:
            pass

    return jsonify({
        "bot_running": _bot_running,
        "bot_paused": _bot._paused if _bot else False,
        "paper_trading": config.ALPACA_PAPER,
        "max_positions": config.MAX_POSITIONS,
        "watchlist_size": len(config.WATCHLIST),
        "llm_enabled": bool(config.ANTHROPIC_API_KEY),
        "short_selling": config.SHORT_SELLING_ENABLED,
        "error": _bot_error,
        **acct,
    })


# ── Positions ─────────────────────────────────────────────────────────────────

@app.route("/api/positions")
def positions():
    if not _bot:
        return jsonify([])
    try:
        alpaca_pos = {p.symbol: p for p in _bot.trade_client.get_all_positions()}
        result = []
        for symbol, pos in alpaca_pos.items():
            st = _bot.state.get(symbol, {})
            result.append({
                "symbol": symbol,
                "sector": config.SECTOR_MAP.get(symbol, "unknown"),
                "direction": st.get("direction", "long"),
                "qty": int(pos.qty),
                "entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "pnl_dollars": round(float(pos.unrealized_pl), 2),
                "pnl_pct": round(float(pos.unrealized_plpc) * 100, 2),
                "stop_loss": st.get("stop_loss"),
                "trailing_stop": st.get("trailing_stop"),
                "take_profit": st.get("take_profit"),
                "half_sold": st.get("half_sold", False),
                "entry_date": st.get("entry_date"),
                "atr": st.get("atr"),
            })
        return jsonify(result)
    except Exception as e:
        logger.error("Positions: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Trades ────────────────────────────────────────────────────────────────────

@app.route("/api/trades")
def trades():
    if not _bot:
        return jsonify([])
    try:
        activities = _bot.trade_client.get_activities()
        result = []
        for a in list(activities)[:50]:
            try:
                result.append({
                    "id": str(a.id),
                    "symbol": a.symbol,
                    "side": str(a.side),
                    "qty": float(a.qty),
                    "price": float(a.price),
                    "total": round(float(a.qty) * float(a.price), 2),
                    "timestamp": str(a.transaction_time),
                })
            except Exception:
                pass
        return jsonify(result)
    except Exception as e:
        logger.error("Trades: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Signals ───────────────────────────────────────────────────────────────────

@app.route("/api/signals")
def signals():
    if not _bot:
        return jsonify([])
    data = list(_bot._last_signals)
    for item in data:
        item["rs_score"] = round(_bot._last_rs_scores.get(item["symbol"], 0), 2)
    return jsonify(data)


# ── Account ───────────────────────────────────────────────────────────────────

@app.route("/api/account")
def account():
    if not _bot:
        return jsonify({"error": "bot not running yet"}), 503
    try:
        a = _bot.trade_client.get_account()
        last_eq = float(a.last_equity)
        eq = float(a.equity)
        return jsonify({
            "equity": eq,
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "portfolio_value": float(a.portfolio_value),
            "daily_pnl": round(eq - last_eq, 2),
            "daily_pnl_pct": round((eq - last_eq) / last_eq * 100, 2) if last_eq else 0,
        })
    except Exception as e:
        logger.error("Account: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Chatbot ───────────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def chat():
    global _chat_history
    if not config.ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not set in environment"}), 503
    if not _bot:
        return jsonify({"error": "bot not running yet"}), 503

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message field required"}), 400

    try:
        import llm_analyst  # lazy — anthropic pkg may not be installed
        reply, _chat_history = llm_analyst.chat(
            message=message,
            history=_chat_history[-30:],
            tool_executor=_bot.tool_executor,
        )
        return jsonify({"reply": reply, "history_length": len(_chat_history)})
    except Exception as e:
        logger.error("Chat error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/clear", methods=["POST"])
def chat_clear():
    global _chat_history
    _chat_history = []
    return jsonify({"cleared": True})


# ── LLM prediction ────────────────────────────────────────────────────────────

@app.route("/api/predict/<symbol>")
def predict(symbol: str):
    if not config.ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not set in environment"}), 503
    if not _bot:
        return jsonify({"error": "bot not running yet"}), 503

    symbol = symbol.upper()
    try:
        import llm_analyst       # lazy import
        import history_analyzer  # lazy import
        from strategy import add_indicators

        df = _bot._single_bars(symbol)
        if df.empty:
            return jsonify({"error": f"No data for {symbol}"}), 404

        df_ind = add_indicators(df)
        if df_ind.empty:
            return jsonify({"error": "Could not compute indicators"}), 500

        cur = df_ind.iloc[-1]
        hist = history_analyzer.full_analysis(df)
        indicators = {
            "price": round(float(cur["close"]), 2),
            "rsi": round(float(cur["rsi"]), 1),
            "stoch_rsi": round(float(cur["stoch_rsi"]), 1),
            "macd_hist": round(float(cur["macd_hist"]), 4),
            "price_vs_ema21": round((float(cur["close"]) / float(cur["ema21"]) - 1) * 100, 1),
            "price_vs_ema200": round((float(cur["close"]) / float(cur["ema200"]) - 1) * 100, 1),
            "vol_ratio": round(float(cur["volume"]) / float(cur["vol_avg20"]), 2) if cur["vol_avg20"] > 0 else 0,
            "atr": round(float(cur["atr"]), 2),
        }
        prediction = llm_analyst.predict_stock(symbol, indicators, hist, [])
        return jsonify({"symbol": symbol, "indicators": indicators, "history": hist, "prediction": prediction})
    except Exception as e:
        logger.error("Predict %s: %s", symbol, e)
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=_run_bot, daemon=True, name="trading-bot")
    t.start()
    port = config.HEALTH_PORT
    logger.info("API server listening on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
