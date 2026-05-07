"""
Trading bot orchestrator — all priorities wired together.

Cycle order:
  1. Market clock check
  2. Bull-fetch all symbols (watchlist + sector ETFs + open positions)
  3. Market regime (SPY 200 EMA)
  4. Check exits: earnings exit → trailing stop update → partial/full exit
  5. (Bull) RS-rank watchlist → earnings blackout → sector ETF → history
       → time filter → 15-min confirm → buy
  6. (Bear) Scan for short setups
"""

import json
import logging
import os
import time
from datetime import date
from typing import Any

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
import strategy
import risk
import earnings
import ranker
import history_analyzer
import llm_analyst

logger = logging.getLogger(__name__)
STATE_FILE = "positions.json"


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def _save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


class TradingBot:
    def __init__(self):
        _setup_logging()
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")

        self.trade_client = TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=config.ALPACA_PAPER,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
        )
        self.state: dict = _load_state()
        self._paused: bool = False
        self._last_signals: list[dict] = []    # for /api/signals
        self._last_rs_scores: dict[str, float] = {}

        logger.info("TradingBot ready — paper=%s  watchlist=%d", config.ALPACA_PAPER, len(config.WATCHLIST))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_market_open(self) -> bool:
        try:
            return self.trade_client.get_clock().is_open
        except Exception as e:
            logger.warning("Clock check failed: %s", e)
            return False

    def _get_equity(self) -> float:
        try:
            return float(self.trade_client.get_account().equity)
        except Exception as e:
            logger.error("Equity fetch: %s", e)
            return 0.0

    def _get_open_symbols(self) -> set[str]:
        try:
            return {p.symbol for p in self.trade_client.get_all_positions()}
        except Exception as e:
            logger.error("Positions fetch: %s", e)
            return set()

    def _get_position_qty(self, symbol: str) -> int:
        try:
            return int(self.trade_client.get_open_position(symbol).qty)
        except Exception:
            return 0

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _bulk_bars(self, symbols: list[str], limit: int = 220) -> dict[str, pd.DataFrame]:
        if not symbols:
            return {}
        try:
            raw = self.data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                limit=limit,
            ))
            df_all = raw.df
            if df_all.empty:
                return {}
            out: dict[str, pd.DataFrame] = {}
            if isinstance(df_all.index, pd.MultiIndex):
                for sym in symbols:
                    try:
                        df = df_all.xs(sym, level=0).reset_index()
                        df.columns = [c.lower() for c in df.columns]
                        out[sym] = df
                    except KeyError:
                        pass
            else:
                df_all = df_all.reset_index()
                df_all.columns = [c.lower() for c in df_all.columns]
                out[symbols[0]] = df_all
            return out
        except Exception as e:
            logger.error("Bulk bars failed: %s", e)
            return {}

    def _15min_bars(self, symbol: str, limit: int = 40) -> pd.DataFrame:
        try:
            raw = self.data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                limit=limit,
            ))
            df = raw.df
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level=0)
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            logger.debug("15min fetch failed %s: %s", symbol, e)
            return pd.DataFrame()

    def _single_bars(self, symbol: str, limit: int = 220) -> pd.DataFrame:
        result = self._bulk_bars([symbol], limit)
        return result.get(symbol, pd.DataFrame())

    # ── Order execution ───────────────────────────────────────────────────────

    def _submit(self, symbol: str, qty: int, side: OrderSide) -> bool:
        if qty <= 0:
            return False
        try:
            self.trade_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
            ))
            return True
        except Exception as e:
            logger.error("Order %s %s qty=%d: %s", side, symbol, qty, e)
            return False

    def _buy(self, sig: strategy.Signal, qty: int):
        if not self._submit(sig.symbol, qty, OrderSide.BUY):
            return
        self.state[sig.symbol] = {
            "direction": "long",
            "entry_price": sig.price,
            "stop_loss": sig.stop_loss,
            "take_profit": sig.take_profit,
            "trailing_stop": sig.stop_loss,
            "qty": qty,
            "half_sold": False,
            "entry_date": str(date.today()),
            "atr": sig.atr,
        }
        _save_state(self.state)
        logger.info("BUY  %-6s qty=%d @ ~$%.2f  SL=$%.2f  TP=$%.2f | %s",
                    sig.symbol, qty, sig.price, sig.stop_loss, sig.take_profit, sig.reason)

    def _short(self, sig: strategy.Signal, qty: int):
        if not self._submit(sig.symbol, qty, OrderSide.SELL):
            return
        self.state[sig.symbol] = {
            "direction": "short",
            "entry_price": sig.price,
            "stop_loss": sig.stop_loss,    # above entry for shorts
            "take_profit": sig.take_profit,  # below entry
            "trailing_stop": sig.stop_loss,
            "qty": qty,
            "half_sold": False,
            "entry_date": str(date.today()),
            "atr": sig.atr,
        }
        _save_state(self.state)
        logger.info("SHORT %-6s qty=%d @ ~$%.2f  SL=$%.2f  TP=$%.2f | %s",
                    sig.symbol, qty, sig.price, sig.stop_loss, sig.take_profit, sig.reason)

    def _sell_qty(self, symbol: str, qty: int, reason: str):
        direction = self.state.get(symbol, {}).get("direction", "long")
        side = OrderSide.SELL if direction == "long" else OrderSide.BUY  # BUY to cover short
        if not self._submit(symbol, qty, side):
            return
        entry = self.state.get(symbol, {}).get("entry_price", 0)
        action = "SELL" if direction == "long" else "COVER"
        logger.info("%s %-6s qty=%d entry~$%.2f | %s", action, symbol, qty, entry, reason)

    # ── Exit management ───────────────────────────────────────────────────────

    def _check_exits(self, open_symbols: set[str], bars: dict[str, pd.DataFrame]):
        for symbol in list(open_symbols):
            st = self.state.get(symbol)
            if not st:
                continue

            # P1 — exit before earnings gap
            if earnings.should_exit_before_earnings(symbol, config.EARNINGS_EXIT_DAYS):
                qty = self._get_position_qty(symbol) or st.get("qty", 0)
                self._sell_qty(symbol, qty, "earnings exit")
                self.state.pop(symbol, None)
                _save_state(self.state)
                continue

            df_raw = bars.get(symbol)
            if df_raw is None or df_raw.empty:
                continue
            df = strategy.add_indicators(df_raw)
            if df.empty:
                continue

            cur = df.iloc[-1].to_dict()
            prev = df.iloc[-2].to_dict()
            cur["prev_macd_hist"] = prev.get("macd_hist", 0)

            price = float(cur["close"])
            entry_price = float(st["entry_price"])
            atr = float(st["atr"])
            direction = st.get("direction", "long")

            # Update trailing stop
            if direction == "long":
                new_ts = risk.update_trailing_stop(entry_price, price, float(st.get("trailing_stop", st["stop_loss"])), atr)
            else:
                # For shorts, trailing stop moves DOWN
                new_ts = risk.update_short_trailing_stop(entry_price, price, float(st.get("trailing_stop", st["stop_loss"])), atr)

            if new_ts != st.get("trailing_stop"):
                logger.info("TRAIL %-6s → $%.2f", symbol, new_ts)
                st["trailing_stop"] = new_ts
                _save_state(self.state)

            # Days held
            try:
                days_held = (date.today() - date.fromisoformat(st.get("entry_date", str(date.today())))).days
            except Exception:
                days_held = 0

            bar_ctx = {**cur, "atr": atr}

            if direction == "long":
                action, reason = strategy.check_exit(
                    entry_price, bar_ctx, days_held, symbol,
                    float(st.get("trailing_stop", st["stop_loss"])),
                    float(st["take_profit"]), bool(st.get("half_sold", False)),
                )
            else:
                action, reason = strategy.check_short_exit(
                    entry_price, bar_ctx, days_held, symbol,
                    float(st.get("trailing_stop", st["stop_loss"])),
                    float(st["take_profit"]), bool(st.get("half_sold", False)),
                )

            total_qty = self._get_position_qty(symbol) or int(st.get("qty", 0))

            if action in ("SELL_ALL", "COVER_ALL"):
                self._sell_qty(symbol, total_qty, reason)
                self.state.pop(symbol, None)
                _save_state(self.state)

            elif action in ("SELL_HALF", "COVER_HALF"):
                half = max(total_qty // 2, 1)
                self._sell_qty(symbol, half, reason)
                st["half_sold"] = True
                st["qty"] = total_qty - half
                # After partial, move trailing stop to breakeven
                st["trailing_stop"] = entry_price if direction == "long" else entry_price
                _save_state(self.state)

    # ── Entry scanning — long (bull market) ──────────────────────────────────

    def _scan_entries(
        self,
        open_symbols: set[str],
        equity: float,
        bars: dict[str, pd.DataFrame],
        ranked_symbols: list[str],
    ):
        open_count = len(open_symbols)
        positions_meta = [{**v, "symbol": k} for k, v in self.state.items() if k in open_symbols]
        signals_log: list[dict] = []

        for symbol in ranked_symbols:
            if not risk.can_open_position(open_count):
                logger.info("Max positions reached")
                break
            if symbol in open_symbols:
                continue
            if not risk.heat_ok(positions_meta, equity):
                break
            if not risk.sector_ok(symbol, {k: v for k, v in self.state.items() if k in open_symbols}):
                continue

            # P1 — earnings blackout
            if earnings.should_skip_entry(symbol, config.EARNINGS_BLACKOUT_DAYS):
                continue

            df = bars.get(symbol)
            if df is None or df.empty:
                continue

            # Historical win rate check
            hist = history_analyzer.full_analysis(df)
            hist_wr = hist.get("hist_rsi_win_rate", 100.0)

            # 15-min intraday confirmation
            df_15min = self._15min_bars(symbol) if config.INTRADAY_CONFIRM_ENABLED else None

            sig = strategy.evaluate(
                df, symbol,
                bullish_market=True,
                bars=bars,
                df_15min=df_15min,
                hist_win_rate=hist_wr,
            )

            signals_log.append({
                "symbol": symbol,
                "action": sig.action,
                "rsi": round(sig.rsi, 1),
                "macd_hist": round(sig.macd_hist, 4),
                "reason": sig.reason,
                "rs_score": round(self._last_rs_scores.get(symbol, 0), 2),
                "hist_win_rate": hist_wr,
            })

            if sig.action != "BUY":
                continue
            if not risk.risk_reward_ok(sig.price, sig.stop_loss, sig.take_profit):
                continue

            qty = risk.position_size(equity, sig.price, sig.stop_loss)
            if qty == 0:
                logger.info("SKIP %-6s qty=0", symbol)
                continue

            self._buy(sig, qty)
            open_symbols.add(symbol)
            open_count += 1
            positions_meta.append({"symbol": symbol, "entry_price": sig.price,
                                    "stop_loss": sig.stop_loss, "qty": qty})
            time.sleep(0.3)

        self._last_signals = signals_log

    # ── Entry scanning — short (bear market, P3) ──────────────────────────────

    def _scan_short_entries(
        self,
        open_symbols: set[str],
        equity: float,
        bars: dict[str, pd.DataFrame],
    ):
        open_count = len(open_symbols)
        for symbol in config.WATCHLIST:
            if not risk.can_open_position(open_count):
                break
            if symbol in open_symbols:
                continue
            if earnings.should_skip_entry(symbol, config.EARNINGS_BLACKOUT_DAYS):
                continue

            df = bars.get(symbol)
            if df is None or df.empty:
                continue

            sig = strategy.evaluate_short(df, symbol, bearish_market=True)
            if sig.action != "SHORT":
                continue
            if not risk.risk_reward_ok(sig.price, sig.take_profit, sig.stop_loss):
                continue

            qty = risk.position_size(equity, sig.price, sig.take_profit)  # risk is TP distance for shorts
            if qty == 0:
                continue

            self._short(sig, qty)
            open_symbols.add(symbol)
            open_count += 1
            time.sleep(0.3)

    # ── Chatbot tool executor ─────────────────────────────────────────────────

    def tool_executor(self, tool_name: str, tool_input: dict) -> Any:
        """Called by llm_analyst.chat() to execute chatbot tool calls."""
        if tool_name == "close_position":
            symbol = tool_input["symbol"].upper()
            qty = self._get_position_qty(symbol)
            if qty == 0:
                return {"error": f"No open position in {symbol}"}
            self._sell_qty(symbol, qty, "chatbot: close")
            self.state.pop(symbol, None)
            _save_state(self.state)
            return {"success": True, "closed": symbol, "qty": qty}

        elif tool_name == "close_all_positions":
            closed = []
            for sym in list(self._get_open_symbols()):
                qty = self._get_position_qty(sym)
                if qty > 0:
                    self._sell_qty(sym, qty, "chatbot: close all")
                    self.state.pop(sym, None)
                    closed.append(sym)
            _save_state(self.state)
            return {"closed": closed}

        elif tool_name == "buy_stock":
            symbol = tool_input["symbol"].upper()
            qty = int(tool_input.get("qty", 1))
            df = self._single_bars(symbol, limit=10)
            price = float(df["close"].iloc[-1]) if not df.empty else 0
            stop = round(price * 0.97, 2)
            tp = round(price * 1.05, 2)
            sig = strategy.Signal("BUY", symbol, price, stop, tp, 0, 0, 0, "chatbot manual buy")
            self._buy(sig, qty)
            return {"success": True, "symbol": symbol, "qty": qty, "price": price}

        elif tool_name == "get_positions":
            try:
                positions = self.trade_client.get_all_positions()
                return [
                    {
                        "symbol": p.symbol,
                        "qty": p.qty,
                        "entry_price": float(p.avg_entry_price),
                        "current_price": float(p.current_price),
                        "pnl_dollars": float(p.unrealized_pl),
                        "pnl_pct": round(float(p.unrealized_plpc) * 100, 2),
                    }
                    for p in positions
                ]
            except Exception as e:
                return {"error": str(e)}

        elif tool_name == "get_account":
            try:
                acc = self.trade_client.get_account()
                return {
                    "equity": float(acc.equity),
                    "cash": float(acc.cash),
                    "buying_power": float(acc.buying_power),
                    "daily_pnl": round(float(acc.equity) - float(acc.last_equity), 2),
                }
            except Exception as e:
                return {"error": str(e)}

        elif tool_name == "analyze_stock":
            symbol = tool_input["symbol"].upper()
            df = self._single_bars(symbol)
            if df.empty:
                return {"error": f"No data for {symbol}"}
            df_ind = strategy.add_indicators(df)
            if df_ind.empty:
                return {"error": "Could not compute indicators"}
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
            return {"symbol": symbol, "indicators": indicators, "history": hist, "prediction": prediction}

        elif tool_name == "pause_bot":
            self._paused = True
            logger.info("Bot PAUSED by chatbot")
            return {"status": "paused"}

        elif tool_name == "resume_bot":
            self._paused = False
            logger.info("Bot RESUMED by chatbot")
            return {"status": "running"}

        return {"error": f"Unknown tool: {tool_name}"}

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_once(self):
        if not self._is_market_open():
            logger.info("Market closed")
            return

        logger.info("=== Cycle start ===")
        equity = self._get_equity()
        if equity <= 0:
            logger.error("Equity=0 — aborting")
            return
        logger.info("Equity: $%.2f  paused=%s", equity, self._paused)

        open_symbols = self._get_open_symbols()
        logger.info("Open (%d): %s", len(open_symbols), open_symbols or "none")

        # Bulk-fetch everything in one call
        all_syms = list(set(config.WATCHLIST) | set(config.SECTOR_ETFS) | open_symbols)
        logger.info("Fetching %d symbols…", len(all_syms))
        bars = self._bulk_bars(all_syms, limit=220)

        # Market regime
        spy_df = bars.get(config.REGIME_SYMBOL)
        bullish = strategy.market_is_bullish(spy_df)

        # Always check exits regardless of pause
        self._check_exits(open_symbols, bars)
        open_symbols = self._get_open_symbols()

        if not self._paused:
            if bullish:
                # P2: rank by relative strength, take top RS_TOP_N
                self._last_rs_scores = ranker.all_scores(bars, spy_df or pd.DataFrame())
                ranked = ranker.rank(bars, spy_df or pd.DataFrame())
                self._scan_entries(open_symbols, equity, bars, ranked)
            else:
                logger.warning("BEAR market — scanning shorts only")
                if config.SHORT_SELLING_ENABLED:
                    self._scan_short_entries(open_symbols, equity, bars)
        else:
            logger.info("Bot paused — skipping entries")

        logger.info("=== Cycle complete ===\n")

    def run(self):
        logger.info("Running — interval=%ds", config.CHECK_INTERVAL_SECONDS)
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.exception("Cycle error: %s", e)
            time.sleep(config.CHECK_INTERVAL_SECONDS)
