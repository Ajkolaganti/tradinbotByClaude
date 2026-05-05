"""
Trading bot orchestrator.

Cycle (every CHECK_INTERVAL_SECONDS, market hours only):
  1. Market regime check (SPY vs 200 EMA) — skip all buys in bear market
  2. Bulk-fetch bars for all watchlist symbols in one API call
  3. Check exits: trailing stop update → partial profit → full exit
  4. Scan entries: sector filter → signal → RR check → size → buy
"""

import logging
import time
import json
import os
from datetime import date

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
import strategy
import risk

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
        logger.info("TradingBot ready — paper=%s  watchlist=%d symbols",
                    config.ALPACA_PAPER, len(config.WATCHLIST))

    # ── Market status ─────────────────────────────────────────────────────────

    def _is_market_open(self) -> bool:
        try:
            return self.trade_client.get_clock().is_open
        except Exception as e:
            logger.warning("Clock check failed: %s", e)
            return False

    # ── Bulk data fetch ───────────────────────────────────────────────────────

    def _get_bulk_bars(self, symbols: list[str], limit: int = 220) -> dict[str, pd.DataFrame]:
        """
        Fetch bars for multiple symbols in ONE API call (much faster than one-by-one).
        Returns {symbol: df} mapping.
        """
        if not symbols:
            return {}
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                limit=limit,
            )
            raw = self.data_client.get_stock_bars(req)
            df_all = raw.df
            if df_all.empty:
                return {}

            result: dict[str, pd.DataFrame] = {}
            if isinstance(df_all.index, pd.MultiIndex):
                for sym in symbols:
                    try:
                        df = df_all.xs(sym, level=0).reset_index()
                        df.columns = [c.lower() for c in df.columns]
                        result[sym] = df
                    except KeyError:
                        pass
            else:
                df_all = df_all.reset_index()
                df_all.columns = [c.lower() for c in df_all.columns]
                result[symbols[0]] = df_all

            return result
        except Exception as e:
            logger.error("Bulk fetch failed: %s", e)
            return {}

    # ── Account helpers ───────────────────────────────────────────────────────

    def _get_equity(self) -> float:
        try:
            return float(self.trade_client.get_account().equity)
        except Exception as e:
            logger.error("Equity fetch failed: %s", e)
            return 0.0

    def _get_open_symbols(self) -> set[str]:
        try:
            return {p.symbol for p in self.trade_client.get_all_positions()}
        except Exception as e:
            logger.error("Positions fetch failed: %s", e)
            return set()

    def _get_position_qty(self, symbol: str) -> int:
        try:
            return int(self.trade_client.get_open_position(symbol).qty)
        except Exception:
            return 0

    # ── Order execution ───────────────────────────────────────────────────────

    def _submit_order(self, symbol: str, qty: int, side: OrderSide) -> bool:
        if qty <= 0:
            return False
        try:
            self.trade_client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY,
            ))
            return True
        except Exception as e:
            logger.error("Order %s %s qty=%d failed: %s", side, symbol, qty, e)
            return False

    def _buy(self, sig: strategy.Signal, qty: int):
        if not self._submit_order(sig.symbol, qty, OrderSide.BUY):
            return
        self.state[sig.symbol] = {
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

    def _sell_qty(self, symbol: str, qty: int, reason: str):
        if not self._submit_order(symbol, qty, OrderSide.SELL):
            return
        entry = self.state.get(symbol, {}).get("entry_price", 0)
        logger.info("SELL %-6s qty=%d entry~$%.2f | %s", symbol, qty, entry, reason)

    # ── Exit management ───────────────────────────────────────────────────────

    def _check_exits(self, open_symbols: set[str], bars: dict[str, pd.DataFrame]):
        for symbol in list(open_symbols):
            st = self.state.get(symbol)
            if not st:
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

            # 1. Update trailing stop
            new_trailing = risk.update_trailing_stop(
                entry_price=entry_price,
                current_price=price,
                current_stop=float(st.get("trailing_stop", st["stop_loss"])),
                atr=atr,
            )
            if new_trailing != st.get("trailing_stop"):
                logger.info("TRAIL %-6s stop raised to $%.2f", symbol, new_trailing)
                st["trailing_stop"] = new_trailing
                _save_state(self.state)

            # 2. Check exit conditions
            entry_date = st.get("entry_date", str(date.today()))
            try:
                days_held = (date.today() - date.fromisoformat(entry_date)).days
            except Exception:
                days_held = 0

            action, reason = strategy.check_exit(
                entry_price=entry_price,
                current_bar={**cur, "atr": atr},
                days_held=days_held,
                symbol=symbol,
                trailing_stop=float(st.get("trailing_stop", st["stop_loss"])),
                take_profit=float(st["take_profit"]),
                half_sold=bool(st.get("half_sold", False)),
            )

            total_qty = self._get_position_qty(symbol) or int(st.get("qty", 0))

            if action == "SELL_ALL":
                self._sell_qty(symbol, total_qty, reason)
                self.state.pop(symbol, None)
                _save_state(self.state)

            elif action == "SELL_HALF":
                half_qty = max(total_qty // 2, 1)
                self._sell_qty(symbol, half_qty, reason)
                st["half_sold"] = True
                st["qty"] = total_qty - half_qty
                # Move trailing stop to breakeven after partial profit
                st["trailing_stop"] = max(float(st.get("trailing_stop", st["stop_loss"])), entry_price)
                _save_state(self.state)

    # ── Entry scanning ────────────────────────────────────────────────────────

    def _scan_entries(
        self,
        open_symbols: set[str],
        equity: float,
        bars: dict[str, pd.DataFrame],
        bullish_market: bool,
    ):
        open_count = len(open_symbols)
        positions_meta = [
            {**v, "symbol": k} for k, v in self.state.items() if k in open_symbols
        ]

        for symbol in config.WATCHLIST:
            if not risk.can_open_position(open_count):
                logger.info("Max positions (%d) reached", config.MAX_POSITIONS)
                break
            if symbol in open_symbols:
                continue
            if not risk.heat_ok(positions_meta, equity):
                break
            if not risk.sector_ok(symbol, {k: v for k, v in self.state.items() if k in open_symbols}):
                continue

            df = bars.get(symbol)
            if df is None or df.empty:
                continue

            sig = strategy.evaluate(df, symbol, bullish_market=bullish_market)
            if sig.action != "BUY":
                logger.debug("SKIP %-6s RSI=%.1f MACD_H=%.4f | %s",
                             symbol, sig.rsi, sig.macd_hist, sig.reason)
                continue

            if not risk.risk_reward_ok(sig.price, sig.stop_loss, sig.take_profit):
                continue

            qty = risk.position_size(equity, sig.price, sig.stop_loss)
            if qty == 0:
                logger.info("SKIP %-6s qty=0 (equity too low or stop too close)", symbol)
                continue

            self._buy(sig, qty)
            open_symbols.add(symbol)
            open_count += 1
            positions_meta.append({
                "symbol": symbol, "entry_price": sig.price,
                "stop_loss": sig.stop_loss, "qty": qty,
            })
            time.sleep(0.3)   # gentle API throttle

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_once(self):
        if not self._is_market_open():
            logger.info("Market closed — sleeping")
            return

        logger.info("=== Cycle start ===")
        equity = self._get_equity()
        if equity <= 0:
            logger.error("Could not get equity — aborting")
            return
        logger.info("Equity: $%.2f", equity)

        open_symbols = self._get_open_symbols()
        logger.info("Open positions (%d): %s", len(open_symbols), open_symbols or "none")

        # Bulk-fetch all data in one request: open positions + watchlist
        all_symbols = list(set(config.WATCHLIST) | open_symbols)
        logger.info("Fetching bars for %d symbols (one API call)…", len(all_symbols))
        bars = self._get_bulk_bars(all_symbols, limit=220)

        # Market regime check
        spy_df = bars.get(config.REGIME_SYMBOL)
        bullish = strategy.market_is_bullish(spy_df)

        self._check_exits(open_symbols, bars)
        open_symbols = self._get_open_symbols()   # refresh after exits

        if bullish:
            self._scan_entries(open_symbols, equity, bars, bullish_market=True)
        else:
            logger.warning("BEAR market detected — no new entries this cycle")

        logger.info("=== Cycle complete ===\n")

    def run(self):
        logger.info("Bot running — interval=%ds", config.CHECK_INTERVAL_SECONDS)
        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.exception("Unhandled error: %s", e)
            time.sleep(config.CHECK_INTERVAL_SECONDS)
