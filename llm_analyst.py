"""
LLM integration via Claude API (anthropic SDK).

Two roles:
  1. Chatbot  — understands natural-language commands ("close AAPL",
     "what's my P&L?", "buy 5 MSFT", "analyze NVDA") and executes them
     via tool calls wired to the live bot.

  2. Stock predictor — given full technical + historical context,
     Claude reasons about likely direction and outputs a structured
     prediction with confidence, entry zone, target, and key factors.

Model: claude-opus-4-7 for predictions (most capable reasoning),
       claude-haiku-4-5-20251001 for simple chatbot commands (fast/cheap).
"""

import json
import logging
import os
from typing import Callable, Any

import anthropic

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        _client = anthropic.Anthropic(api_key=key)
    return _client


# ── Chatbot tool definitions ──────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "close_position",
        "description": "Sell all shares of an open position immediately at market price.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string", "description": "Ticker e.g. AAPL"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "close_all_positions",
        "description": "Close every open position immediately.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "buy_stock",
        "description": "Submit a market buy order for a stock.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "qty": {"type": "integer", "description": "Number of shares"},
            },
            "required": ["symbol", "qty"],
        },
    },
    {
        "name": "get_positions",
        "description": "Return all open positions with current price and unrealised P&L.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_account",
        "description": "Return account equity, cash, buying power, and today's P&L.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "analyze_stock",
        "description": "Run full technical + historical analysis on a stock and predict direction.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "pause_bot",
        "description": "Pause automated entries (existing positions still managed).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "resume_bot",
        "description": "Resume automated trading after a pause.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

_SYSTEM = (
    "You are TradinBot Assistant — the AI layer on top of an Alpaca paper-trading bot. "
    "You can close positions, buy stocks, show P&L, analyze stocks, and pause/resume the bot. "
    "Be concise. When executing trades, confirm the action in plain English. "
    "When analyzing stocks, explain RSI/MACD in plain language and give a clear BUY/HOLD/SELL opinion. "
    "Never reveal implementation details."
)


# ── Chatbot ───────────────────────────────────────────────────────────────────

def chat(
    message: str,
    history: list[dict],
    tool_executor: Callable[[str, dict], Any],
) -> tuple[str, list[dict]]:
    """
    Process one user message.
    Returns (assistant_reply, updated_history).

    history format: [{"role": "user"|"assistant", "content": str}, ...]
    tool_executor: fn(tool_name, tool_input_dict) → jsonable result
    """
    client = _get_client()
    messages = list(history) + [{"role": "user", "content": message}]

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=_SYSTEM,
        tools=TOOLS,
        messages=messages,
    )

    tool_results: list[dict] = []
    reply_text = ""

    for block in resp.content:
        if block.type == "text":
            reply_text += block.text
        elif block.type == "tool_use":
            logger.info("Chatbot tool: %s(%s)", block.name, block.input)
            try:
                result = tool_executor(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
            except Exception as e:
                logger.error("Tool %s failed: %s", block.name, e)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Error: {e}",
                    "is_error": True,
                })

    if tool_results:
        # Feed results back for a final natural-language response
        messages = messages + [
            {"role": "assistant", "content": resp.content},
            {"role": "user", "content": tool_results},
        ]
        final = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=_SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        reply_text = "".join(b.text for b in final.content if hasattr(b, "text"))
        updated_history = messages + [{"role": "assistant", "content": reply_text}]
    else:
        updated_history = messages + [{"role": "assistant", "content": reply_text}]

    return reply_text, updated_history


# ── Stock predictor ───────────────────────────────────────────────────────────

def predict_stock(
    symbol: str,
    indicators: dict,
    history: dict,
    recent_signals: list[str],
) -> dict:
    """
    Ask Claude Opus to reason about likely price direction over the next 3-5 days.
    Returns structured JSON with direction, confidence, reasoning, etc.
    """
    client = _get_client()

    prompt = f"""You are an expert quantitative trader. Analyze {symbol} and predict its price direction for the next 3-5 trading days.

TECHNICAL INDICATORS (current bar):
- RSI(14):              {indicators.get('rsi', 'N/A')}
- Stochastic RSI:       {indicators.get('stoch_rsi', 'N/A')}
- MACD Histogram:       {indicators.get('macd_hist', 'N/A')}
- Price vs EMA21:       {indicators.get('price_vs_ema21', 'N/A')}%
- Price vs EMA200:      {indicators.get('price_vs_ema200', 'N/A')}%
- Volume / 20d avg:     {indicators.get('vol_ratio', 'N/A')}×
- ATR(14):              ${indicators.get('atr', 'N/A')}
- Current price:        ${indicators.get('price', 'N/A')}

HISTORICAL CONTEXT:
- 52w high / low:       ${history.get('high_52w','N/A')} / ${history.get('low_52w','N/A')}
- % from 52w high:      {history.get('pct_from_52w_high','N/A')}%
- % from 52w low:       {history.get('pct_from_52w_low','N/A')}%
- Nearest support:      ${history.get('nearest_support','N/A')}
- Nearest resistance:   ${history.get('nearest_resistance','N/A')}
- Point of Control:     ${history.get('point_of_control','N/A')}
- RSI win-rate history: {history.get('hist_rsi_win_rate','N/A')}% (n={history.get('hist_sample_size',0)} samples)
- RSI signal reliable:  {history.get('hist_signal_reliable','unknown')}

RECENT BOT SIGNALS: {', '.join(recent_signals) if recent_signals else 'none'}

Respond with ONLY valid JSON (no markdown fences):
{{
  "direction": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <integer 0-100>,
  "entry_zone": "<price range string>",
  "target": "<price target string>",
  "stop_suggestion": "<stop loss price string>",
  "reasoning": "<2-3 sentences explaining the key factors>",
  "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "risk_level": "LOW" | "MEDIUM" | "HIGH"
}}"""

    try:
        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("LLM prediction parse failed for %s: %s", symbol, e)
        return {"direction": "NEUTRAL", "confidence": 0, "reasoning": "Parse error", "error": str(e)}
    except Exception as e:
        logger.error("LLM prediction failed for %s: %s", symbol, e)
        return {"direction": "NEUTRAL", "confidence": 0, "reasoning": str(e), "error": str(e)}
