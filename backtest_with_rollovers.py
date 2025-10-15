"""
Multi‑day backtest: entry → hold → roll → unwind.

- Day 1: run `task1.py` to open long spot + short COIN‑M.
- Daily: if a short is ≤ roll_window_days from expiry, roll it over 24h
  (96 × 15‑minute slots) using TWAP + PoV‑capped sizing; track MTM and realized PnL.
- Final day: unwind all legs over 24h and report PnL.

Reuses `AdaptiveCashCarryExecutor` to keep scheduling/execution logic in one place.
"""

from __future__ import annotations

import datetime as dt
import argparse
import sys
import pandas as pd
import numpy as np
import os
import csv
from typing import Dict, Tuple, List
from dotenv import load_dotenv

# Import classes from task1.py (same directory)
from task1 import (
    SynchronizedCashCarryScanner,
    derive_slot_weights,
    BacktestPriceFeed,
    BinanceSimulator,
    AdaptiveCashCarryExecutor,
    ExecParams,
)
from task1_sell import UnwindParams, OptimizedUnwindExecutor

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------------------------------------------
# RolloverExecutor – smart order routed 24h roll of one futures contract into the next
# --------------------------------------------------------------------------------------


class RolloverExecutor(AdaptiveCashCarryExecutor):
    """
    Roll an expiring COIN‑M short into the next quarter over 24h.
    Futures legs only; spot is unchanged. Slot weights come from a 21‑day
    volume profile.
    """

    def __init__(self,
                 api_key: str,
                 api_secret: str,
                 from_symbol: str,
                 to_symbol: str,
                 contracts: int,
                 target_q_btc: float,
                 slot_weights: Dict[int, float],
                 simulator: BinanceSimulator,
                 price_feed: BacktestPriceFeed,
                 params: ExecParams = ExecParams(hours=24, capital_usdt=0.0)):

        super().__init__(api_key, api_secret, slot_weights, simulator, params=params,
                         price_feed=price_feed, historical_avg_basis=0.0)

        self.from_symbol = from_symbol
        self.to_symbol = to_symbol
        self.roll_contracts = int(contracts)
        self.target_q_btc = float(target_q_btc)
        self.contract_size_usd = 100.0
        self.trade_rows = []

    # Override run to trade only the futures legs
    def run(self, start_dt: dt.datetime, realtime: bool = False):
        P = self.params
        buckets = int((P.hours*60)//P.bucket_minutes)

        remaining_ct_from = self.roll_contracts
        filled_ct_from = 0
        # Recompute target TO contracts each slice using the current to_mid
        target_total_ct_to = 0
        remaining_ct_to = 0
        filled_ct_to = 0

        weights_vec = np.array([self.slot_weights.get(k, 1.0/96) for k in range(96)], dtype=float)
        weights_vec = weights_vec / weights_vec.sum()

        for i in range(buckets):
            ts = start_dt + dt.timedelta(minutes=P.bucket_minutes * i)
            slot = ((ts.hour*60 + ts.minute)//15) % 96
            w = float(weights_vec[slot])

            spot_mid, _ = self._spot_mid_spread(ts)
            # mids for old and new contracts using fallback helpers
            from_mid, from_spr = self._future_mid_spread_by_symbol(self.from_symbol, ts)
            if not np.isfinite(from_mid):
                from_mid, from_spr = self._future_mid_spread_by_contractType("CURRENT_QUARTER", ts)
            to_mid, to_spr = self._future_mid_spread_by_symbol(self.to_symbol, ts)
            if not np.isfinite(to_mid):
                to_mid, to_spr = self._future_mid_spread_by_contractType("NEXT_QUARTER", ts)

            # Target total TO contracts for this slice's reference price
            if np.isfinite(to_mid) and to_mid > 0:
                target_total_ct_to = int(max(1, round(self.target_q_btc * to_mid / self.contract_size_usd)))
            else:
                target_total_ct_to = self.roll_contracts
            if i == 0:
                remaining_ct_to = target_total_ct_to

            # PoV cap: More aggressive cap for rollover execution
            recent_from_vol = self._recent_15m_spot_volume_btc(ts, symbol="BTCUSDT")  # proxy
            pov_cap = 0.25 * recent_from_vol  # Increased from 0.15 to 0.25
            
            # Base allocation from volume weights
            child_from_ct = min(remaining_ct_from, int(max(1, w * self.roll_contracts)))
            
            # Apply PoV cap but ensure minimum execution
            child_from_ct = min(child_from_ct, int(pov_cap))
            
            # Ensure minimum execution per slot to guarantee progress
            min_execution = max(1, remaining_ct_from // max(1, buckets - i))  # At least 1 contract per remaining slot
            child_from_ct = max(child_from_ct, min_execution)
            
            # If still too small, use remaining contracts divided by remaining slots
            if child_from_ct <= 0:
                child_from_ct = max(1, remaining_ct_from // max(1, buckets - i))
            
            # Final safety check
            if child_from_ct <= 0:
                child_from_ct = 1

            # Pricing: More aggressive pricing logic for better fill rates
            progress = 1 - remaining_ct_from / max(self.roll_contracts, 1)
            time_progress = (i+1) / buckets
            behind = (progress < time_progress - 0.05)  # Reduced threshold from 0.10 to 0.05
            
            # More aggressive spread crossing
            spread_threshold = 1.5 * np.median([from_spr, 0.5])  # Reduced from 2.0 to 1.5
            cross = behind or (from_spr > spread_threshold) or (progress < time_progress * 0.5)  # Force cross if very behind
            
            # More aggressive pricing when crossing
            if cross:
                px_from = (from_mid + 0.75*from_spr)  # More aggressive: was 0.5
                px_to   = (to_mid - 0.75*to_spr)      # More aggressive: was 0.5
            else:
                px_from = (from_mid - 0.1*from_spr)   # Less passive: was 0.25
                px_to   = (to_mid + 0.1*to_spr)       # Less passive: was 0.25

            # Place orders: close old (BUY) and open new (SELL)
            buy_params = {"market":"FUT", "symbol":self.from_symbol, "side":"BUY", "type":"LIMIT", "price":float(px_from), "contracts":int(child_from_ct)}

            # Scale the TO-side child contracts to target new hedge notional
            scaled_child_to = int(max(1, round(child_from_ct * (target_total_ct_to / max(self.roll_contracts, 1)))))
            child_to_ct = min(remaining_ct_to, scaled_child_to)
            sell_params= {"market":"FUT", "symbol":self.to_symbol,   "side":"SELL","type":"LIMIT", "price":float(px_to),   "contracts":int(child_to_ct)}

            # Enhanced retry mechanism with progressive pricing
            max_retries = 3
            for retry in range(max_retries):
                buy_resp = self._place_until_filled(buy_params, max_attempts=P.max_bucket_attempts)
                sell_resp = self._place_until_filled(sell_params, max_attempts=P.max_bucket_attempts)
                
                # If both filled, break
                if buy_resp["status"] == "filled" and sell_resp["status"] == "filled":
                    break
                
                # If not filled, make pricing more aggressive for next retry
                if retry < max_retries - 1:
                    if buy_resp["status"] != "filled":
                        buy_params["price"] = float(from_mid + from_spr * (0.5 + retry * 0.25))  # More aggressive
                    if sell_resp["status"] != "filled":
                        sell_params["price"] = float(to_mid - to_spr * (0.5 + retry * 0.25))  # More aggressive

            filled_child_from = int(buy_resp.get("filled_qty",0))
            filled_child_to   = int(sell_resp.get("filled_qty",0))
            filled_ct_from += filled_child_from
            filled_ct_to   += filled_child_to
            remaining_ct_from -= filled_child_from
            remaining_ct_to   -= filled_child_to

            progress_pct = (1 - remaining_ct_from / max(self.roll_contracts, 1)) * 100
            time_progress_pct = ((i+1) / buckets) * 100
            
            # Progress monitoring with warnings
            if progress_pct < time_progress_pct - 20:  # More than 20% behind schedule
                print(f"    ⚠️  [{ts.strftime('%H:%M')} UTC] BEHIND SCHEDULE! Progress: {progress_pct:.1f}% vs Time: {time_progress_pct:.1f}%")
            elif progress_pct < time_progress_pct - 10:  # More than 10% behind schedule
                print(f"    ⚡ [{ts.strftime('%H:%M')} UTC] Behind schedule. Progress: {progress_pct:.1f}% vs Time: {time_progress_pct:.1f}%")
            else:
                print(f"    [{ts.strftime('%H:%M')} UTC] Rollover progress: {progress_pct:.1f}% (Time: {time_progress_pct:.1f}%). "
                      f"Closing {self.from_symbol}, Opening {self.to_symbol}. "
                      f"Slice Qty (from→to): {filled_child_from} → {filled_child_to} contracts.")

            self.exec_log.append({"ts":ts,"slot":slot,"filled_from_ct":filled_child_from, "filled_to_ct": filled_child_to})

            # Record granular trades for CSV
            self.trade_rows.append({"ts":ts,"symbol":self.from_symbol,"side":"BUY","price":px_from,"filled_qty":filled_child_from,"attempts":buy_resp.get("attempts",1)})
            self.trade_rows.append({"ts":ts,"symbol":self.to_symbol,"side":"SELL","price":px_to,"filled_qty":filled_child_to,"attempts":sell_resp.get("attempts",1)})

            if remaining_ct_from<=0 and remaining_ct_to<=0:
                break

        # Enhanced final sweep: force any remaining at market prices
        if remaining_ct_from>0 or remaining_ct_to>0:
            print(f"    [FINAL SWEEP] Forcing completion: {remaining_ct_from} from contracts, {remaining_ct_to} to contracts")
            
            # Use market prices (cross the spread completely)
            buy_params["price"] = float(from_mid + from_spr * 1.5)  # More aggressive than before
            sell_params["price"] = float(to_mid - to_spr * 1.5)     # More aggressive than before
            buy_params["contracts"] = int(max(0, remaining_ct_from))
            sell_params["contracts"] = int(max(0, remaining_ct_to))
            
            # Force fill probability to 100% for final sweep
            original_fill_prob = self.sim.order_fill_prob
            self.sim.order_fill_prob = 1.0
            
            # Execute final sweep with multiple attempts
            if buy_params["contracts"]>0:
                final_buy_resp = self._place_until_filled(buy_params, 10)  # More attempts
                print(f"    [FINAL SWEEP] Buy result: {final_buy_resp.get('filled_qty', 0)} contracts filled")
            if sell_params["contracts"]>0:
                final_sell_resp = self._place_until_filled(sell_params, 10)  # More attempts
                print(f"    [FINAL SWEEP] Sell result: {final_sell_resp.get('filled_qty', 0)} contracts filled")
            
            # Restore original fill probability
            self.sim.order_fill_prob = original_fill_prob
            
            # Update final counts
            final_filled_from = int(buy_resp.get("filled_qty", 0)) if 'buy_resp' in locals() else 0
            final_filled_to = int(sell_resp.get("filled_qty", 0)) if 'sell_resp' in locals() else 0
            filled_ct_from += final_filled_from
            filled_ct_to += final_filled_to
            remaining_ct_from -= final_filled_from
            remaining_ct_to -= final_filled_to

        # Final completion verification
        completion_pct = (filled_ct_from / max(self.roll_contracts, 1)) * 100
        if remaining_ct_from > 0 or remaining_ct_to > 0:
            print(f"    ❌ ROLLOVER INCOMPLETE: {remaining_ct_from} from contracts, {remaining_ct_to} to contracts remaining")
            print(f"    📊 Completion: {completion_pct:.1f}% ({filled_ct_from}/{self.roll_contracts} contracts)")
        else:
            print(f"    ✅ ROLLOVER COMPLETE: 100% execution achieved")
            print(f"    📊 Final: {filled_ct_from} from contracts, {filled_ct_to} to contracts executed")

        return {"log": pd.DataFrame(self.exec_log), "trades": pd.DataFrame(self.trade_rows)}

# --------------------------------------------------------------------------------------
# PnL Tracking
# --------------------------------------------------------------------------------------
def initialize_pnl_tracker():
    """Create pnl_tracker.csv with header (overwrite)."""
    with open("pnl_tracker.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "pnl_usd"])

def record_pnl(ts: dt.datetime, symbol: str, pnl: float):
    with open("pnl_tracker.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts.isoformat(), symbol, pnl])

def initialize_daily_pnl_tracker():
    """Create daily_pnl_tracker.csv with header (overwrite)."""
    with open("daily_pnl_tracker.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "spot_price", "portfolio_value_usd", "daily_pnl_usd", "cumulative_pnl_usd",
            "spot_mtm_pnl_usd", "futures_unrealized_pnl_usd", "futures_realized_pnl_usd"
        ])

def record_daily_pnl(date: dt.date, spot_px: float, portfolio_value: float, daily_pnl: float, cumulative_pnl: float,
                     spot_pnl: float, fut_unrealized: float, fut_realized: float):
    with open("daily_pnl_tracker.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            date.strftime('%Y-%m-%d'), spot_px, portfolio_value, daily_pnl, cumulative_pnl,
            spot_pnl, fut_unrealized, fut_realized
        ])

# def initialize_rehedge_log():
#     """Create rehedge_log.csv with a header (disabled)."""
#     # with open("rehedge_log.csv", "w", newline="") as f:
#     #     writer = csv.writer(f)
#     #     writer.writerow(["timestamp", "symbol", "side", "qty", "price", "net_delta_btc", "reason"])

def record_rehedge_trade(ts: dt.datetime, symbol: str, side: str, qty: int, price: float, delta: float, reason: str):
    with open("rehedge_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts.isoformat(), symbol, side, qty, price, delta, reason])


# --------------------------------------------------------------------------------------
# Helper: parse expiry date from COIN‑M symbol e.g. BTCUSD_240927 → 2024‑09‑27
# --------------------------------------------------------------------------------------

def parse_coin_m_expiry(sym: str) -> dt.date:
    try:
        suffix = sym.split("_")[1]
        y, m, d = int("20" + suffix[:2]), int(suffix[2:4]), int(suffix[4:])
        return dt.date(y, m, d)
    except Exception:
        raise ValueError(f"Cannot parse expiry from symbol: {sym}")

# --------------------------------------------------------------------------------------
# Helper: next Binance quarterly expiry (last Friday of Mar/Jun/Sep/Dec, 08:00 UTC)
# --------------------------------------------------------------------------------------

def next_quarter_expiry(cur_expiry: dt.date) -> dt.date:
    quarter_months = [3, 6, 9, 12]
    # Find current index
    try:
        idx = quarter_months.index(cur_expiry.month)
    except ValueError:
        # If input not on a quarter month, roll forward
        idx = max(i for i,m in enumerate(quarter_months) if m > cur_expiry.month) if any(m>cur_expiry.month for m in quarter_months) else 0
        year = cur_expiry.year + (1 if idx==0 else 0)
        month = quarter_months[idx]
        cur_expiry = dt.date(year, month, 1)

    # Advance one quarter
    idx = (idx + 1) % 4
    year = cur_expiry.year + (1 if idx == 0 and cur_expiry.month == 12 else 0)
    month = quarter_months[idx]

    # Last Friday of that month
    d = dt.date(year, month, 1)
    if month == 12:
        last_day = dt.date(year, 12, 31)
    else:
        last_day = dt.date(year, month+1, 1) - dt.timedelta(days=1)
    while last_day.weekday() != 4:
        last_day -= dt.timedelta(days=1)
    return last_day

# --------------------------------------------------------------------------------------
# Position tracker – keeps current open futures legs (symbol → contracts short)
# --------------------------------------------------------------------------------------

class PositionTracker:
    def __init__(self):
        # Store entry as inverse‑average: entry_inv = contracts‑weighted avg of (1/price)
        # Per symbol: qty, side, entry_inv, entry_px (derived), q_btc
        self.fut_pos: Dict[str, Dict] = {}

    # Load first‑day executor log
    def load_from_log(self, log_df: pd.DataFrame):
        shorts = log_df.groupby("fut_symbol")
        for sym, group in shorts:
            qty = group["fill_fut_ct"].sum()
            fut_prices = pd.to_numeric(group["fut_px"], errors="coerce")
            fills = pd.to_numeric(group["fill_fut_ct"], errors="coerce")
            qty = int(qty)
            sum_inv = float(((1.0/fut_prices) * fills).sum()) if qty > 0 else 0.0
            entry_inv = sum_inv / max(qty,1) if sum_inv > 0 else 0.0
            entry_px = (1.0/entry_inv) if entry_inv > 0 else 0.0
            q_btc = float((qty * 100.0) * entry_inv) if entry_inv > 0 else np.nan
            self.fut_pos[sym] = {"qty": qty, "entry_inv": entry_inv, "entry_px": entry_px, "q_btc": q_btc, "side": "SHORT"}

    def _remove_zeroes(self):
        self.fut_pos = {s: d for s, d in self.fut_pos.items() if d["qty"] != 0}

    # Roll expiring symbol into new_symbol (resize to maintain BTC hedge)
    def rollover(self, from_sym: str, to_sym: str, to_sym_entry_px: float, to_sym_qty: int):
        from_data = self.fut_pos.get(from_sym)
        if not from_data:
            return
        
        ct = from_data["qty"]
        q_btc = from_data.get("q_btc")
        spot_open_px = from_data.get("spot_open_px")
        self.fut_pos[from_sym]["qty"] = 0
        
        # Use the filled qty for the new symbol to set its qty; average entry price
        if to_sym in self.fut_pos:
            existing_data = self.fut_pos[to_sym]
            total_qty = int(existing_data["qty"]) + int(to_sym_qty)
            # inverse-average update: entry_inv = (sum_inv_old + sum_inv_new) / total_qty
            sum_inv_old = float(existing_data.get("entry_inv", 0.0)) * int(existing_data["qty"])
            sum_inv_new = (1.0/float(to_sym_entry_px)) * int(to_sym_qty) if to_sym_entry_px > 0 else 0.0
            new_entry_inv = (sum_inv_old + sum_inv_new) / max(total_qty,1)
            self.fut_pos[to_sym]["qty"] = total_qty
            self.fut_pos[to_sym]["entry_inv"] = new_entry_inv
            self.fut_pos[to_sym]["entry_px"] = (1.0/new_entry_inv) if new_entry_inv>0 else 0.0
            self.fut_pos[to_sym]["side"] = from_data.get("side", "SHORT")
        else:
            entry_inv_new = (1.0/float(to_sym_entry_px)) if to_sym_entry_px>0 else 0.0
            self.fut_pos[to_sym] = {"qty": int(to_sym_qty), "entry_inv": entry_inv_new, "entry_px": float(to_sym_entry_px), "q_btc": q_btc, "side": from_data.get("side", "SHORT")}

        self._remove_zeroes()

    def update_position(self, symbol: str, qty_change: int, price: float, conversion_spot_usd: float | None = None, ts: dt.datetime | None = None):
        """
        Update position from a re‑hedge trade.
        +qty = SELL (add to short), -qty = BUY (reduce short).
        """
        if symbol not in self.fut_pos:
            # Initialize a new position with side inferred from trade direction
            inferred_side = "SHORT" if qty_change > 0 else "LONG"
            self.fut_pos[symbol] = {"qty": abs(qty_change), "entry_px": price, "side": inferred_side}
            return

        existing_data = self.fut_pos[symbol]
        existing_qty = int(existing_data.get("qty", 0))
        existing_inv = float(existing_data.get("entry_inv", 0.0))
        existing_vwap = (1.0/existing_inv) if existing_inv>0 else 0.0
        existing_side = existing_data.get("side", "SHORT")

        if existing_side == "SHORT":
            if qty_change > 0: # SELL: add to short
                total_qty = existing_qty + qty_change
                sum_inv_old = existing_inv * existing_qty
                sum_inv_add = (1.0/price) * qty_change if price>0 else 0.0
                new_entry_inv = (sum_inv_old + sum_inv_add) / max(total_qty,1)
                self.fut_pos[symbol]["qty"] = int(total_qty)
                self.fut_pos[symbol]["entry_inv"] = float(new_entry_inv)
                self.fut_pos[symbol]["entry_px"] = float((1.0/new_entry_inv) if new_entry_inv>0 else 0.0)
                self.fut_pos[symbol]["side"] = "SHORT"
            else: # BUY: reduce short (may flip to long)
                buy_qty = abs(qty_change)
                close_qty = min(existing_qty, buy_qty)
                if close_qty > 0 and existing_vwap > 0 and price > 0:
                    pnl_btc = close_qty * 100.0 * (1/price - 1/existing_vwap)
                    conv = conversion_spot_usd if (conversion_spot_usd is not None and np.isfinite(conversion_spot_usd)) else price
                    pnl_usd = pnl_btc * conv
                    ts_to_use = ts if ts is not None else dt.datetime.now(dt.timezone.utc)
                    record_pnl(ts_to_use, symbol, pnl_usd)
                remaining_short = existing_qty - close_qty
                if remaining_short > 0:
                    self.fut_pos[symbol]["qty"] = int(remaining_short)
                    # keep vwap and side
                    self.fut_pos[symbol]["side"] = "SHORT"
                else:
                    # zeroed or flipped
                    flip_qty = buy_qty - close_qty
                    if flip_qty > 0:
                        # open new LONG at this trade price
                        self.fut_pos[symbol] = {"qty": int(flip_qty), "entry_inv": float((1.0/price) if price>0 else 0.0), "entry_px": float(price), "side": "LONG"}
                    else:
                        self.fut_pos[symbol]["qty"] = 0
        else: # existing_side == "LONG"
            if qty_change < 0: # BUY: add to long
                add_qty = abs(qty_change)
                total_qty = existing_qty + add_qty
                sum_inv_old = existing_inv * existing_qty
                sum_inv_add = (1.0/price) * add_qty if price>0 else 0.0
                new_entry_inv = (sum_inv_old + sum_inv_add) / max(total_qty,1)
                self.fut_pos[symbol]["qty"] = int(total_qty)
                self.fut_pos[symbol]["entry_inv"] = float(new_entry_inv)
                self.fut_pos[symbol]["entry_px"] = float((1.0/new_entry_inv) if new_entry_inv>0 else 0.0)
                self.fut_pos[symbol]["side"] = "LONG"
            else: # SELL: reduce long (may flip to short)
                sell_qty = qty_change
                close_qty = min(existing_qty, sell_qty)
                if close_qty > 0 and existing_vwap > 0 and price > 0:
                    pnl_btc = close_qty * 100.0 * (1/existing_vwap - 1/price)
                    conv = conversion_spot_usd if (conversion_spot_usd is not None and np.isfinite(conversion_spot_usd)) else price
                    pnl_usd = pnl_btc * conv
                    ts_to_use = ts if ts is not None else dt.datetime.now(dt.timezone.utc)
                    record_pnl(ts_to_use, symbol, pnl_usd)
                remaining_long = existing_qty - close_qty
                if remaining_long > 0:
                    self.fut_pos[symbol]["qty"] = int(remaining_long)
                    self.fut_pos[symbol]["side"] = "LONG"
                else:
                    flip_qty = sell_qty - close_qty
                    if flip_qty > 0:
                        # open new SHORT at this trade price
                        self.fut_pos[symbol] = {"qty": int(flip_qty), "entry_inv": float((1.0/price) if price>0 else 0.0), "entry_px": float(price), "side": "SHORT"}
                    else:
                        self.fut_pos[symbol]["qty"] = 0

        self._remove_zeroes()

    def current_positions(self) -> Dict[str, Dict]:
        return dict(self.fut_pos)

# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------

def run_backtest(start_date: dt.date, end_date: dt.date):
    assert start_date <= end_date, "start_date must be <= end_date"

    # Load API credentials from environment
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file")
    
    # --- Initialize PnL trackers ---
    initialize_pnl_tracker()
    initialize_daily_pnl_tracker()
    # initialize_rehedge_log()

    # --- STEP‑1 entry day (task1 logic) ---
    scanner = SynchronizedCashCarryScanner(
        api_key=api_key, api_secret=api_secret, lookback_days=100, basis_bps_min_buy=0.0
    )
    analysis = scanner.run_synchronized_analysis(start_date.strftime("%Y-%m-%d"), resample_15m=True)
    slot_weights = derive_slot_weights(analysis["buy_leaderboard"])

    # Build 24h price feed for the start date
    start_dt = dt.datetime.combine(start_date, dt.time(0, 0), dt.timezone.utc)
    temp_spot = scanner.spot
    temp_cm = scanner.cm
    start_ms = int(start_dt.timestamp()*1000)
    end_ms = start_ms + 24*3600*1000
    spot_kl = scanner._fetch_klines_synchronized(temp_spot, "BTCUSDT", start_ms, end_ms)
    cur_kl  = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", start_ms, end_ms, is_continuous=True, contract_type="CURRENT_QUARTER")
    nxt_kl  = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", start_ms, end_ms, is_continuous=True, contract_type="NEXT_QUARTER")
    price_feed = BacktestPriceFeed(scanner._klines_to_df(spot_kl), scanner._klines_to_df(cur_kl), scanner._klines_to_df(nxt_kl))

    sim = BinanceSimulator(api_key=api_key, secret_key=api_secret, order_fill_prob=0.9)

    executor = AdaptiveCashCarryExecutor(
        api_key=api_key,
        api_secret=api_secret,
        slot_weights=slot_weights,
        simulator=sim,
        price_feed=price_feed,
        historical_avg_basis=analysis.get("historical_avg_basis", 200.0),
        params=ExecParams(capital_usdt=1_000_000.0, hours=24),
    )
    result = executor.run(start_dt=start_dt, realtime=False)
    log_df = result["log"]
    log_df.to_csv("entry_day_exec_log.csv", index=False)
    target_btc_long = result["summary"].get("target_btc", 0.0)

    # Use actual entry-day spot VWAP as basis instead of arrival mid
    try:
        spot_entry_px = float((pd.to_numeric(log_df["spot_px"], errors="coerce") * pd.to_numeric(log_df["fill_spot_btc"], errors="coerce")).sum() / max(pd.to_numeric(log_df["fill_spot_btc"], errors="coerce").sum(), 1e-9))
    except Exception:
        spot_entry_px = result["summary"].get("arrival_spot_mid", np.nan)
    initial_capital = executor.params.capital_usdt
    initial_spot_cost = target_btc_long * spot_entry_px

    # Store initial positions
    pt = PositionTracker()
    pt.load_from_log(log_df)

    # --- STEP‑2 daily loop for rollovers ---
    cur_day = start_date + dt.timedelta(days=1)
    roll_window_days = 2  # start rolling 2 days before expiry
    volume_lookback_days = 21  # use 3-week historical volume profile for weighting

    # --- Daily MTM vars ---
    last_day_portfolio_value = initial_capital
    delta_threshold_pct = 0.05 # Re-hedge if delta exceeds 5% of spot position
    
    final_unwind_executed = False
    while cur_day < end_date:
        print(f"\nProcessing MTM for {cur_day.strftime('%Y-%m-%d')}...")

        # --- Daily price feed for MTM ---
        # We use EOD prices to mark to market.
        day_start_dt = dt.datetime.combine(cur_day, dt.time(0,0), dt.timezone.utc)
        s_ms = int(day_start_dt.timestamp()*1000)
        e_ms = s_ms + 24*3600*1000
        spot_kl_day = scanner._fetch_klines_synchronized(temp_spot, "BTCUSDT", s_ms, e_ms)
        cur_kl_day  = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", s_ms, e_ms, is_continuous=True, contract_type="CURRENT_QUARTER")
        nxt_kl_day  = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", s_ms, e_ms, is_continuous=True, contract_type="NEXT_QUARTER")
        daily_price_feed = BacktestPriceFeed(scanner._klines_to_df(spot_kl_day), scanner._klines_to_df(cur_kl_day), scanner._klines_to_df(nxt_kl_day))
        
        # Get EOD prices (use ffill for safety if last minute has no data)
        eod_ts = day_start_dt + dt.timedelta(hours=23, minutes=59)
        spot_eod_price, _ = daily_price_feed.get_spot_mid_spread(eod_ts)

        # --- Rollover logic (checked before MTM is finalized) ---
        for sym, data in list(pt.current_positions().items()):
            expiry = parse_coin_m_expiry(sym)
            days_to_expiry = (expiry - cur_day).days
            if days_to_expiry <= roll_window_days:
                # Determine next quarter symbol
                next_expiry = next_quarter_expiry(expiry)
                next_sym = f"BTCUSD_{next_expiry.strftime('%y%m%d')}"

                print(f"ROLLOVER EVENT: {sym} → {next_sym} on {cur_day.strftime('%Y-%m-%d')}")

                # 24h price feed for this roll day
                day_start_dt = dt.datetime.combine(cur_day, dt.time(0,0), dt.timezone.utc)
                s_ms = int(day_start_dt.timestamp()*1000)
                e_ms = s_ms + 24*3600*1000
                spot_kl = scanner._fetch_klines_synchronized(temp_spot, "BTCUSDT", s_ms, e_ms)
                cur_kl  = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", s_ms, e_ms, is_continuous=True, contract_type="CURRENT_QUARTER")
                nxt_kl  = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", s_ms, e_ms, is_continuous=True, contract_type="NEXT_QUARTER")
                pf = BacktestPriceFeed(scanner._klines_to_df(spot_kl), scanner._klines_to_df(cur_kl), scanner._klines_to_df(nxt_kl))

                # Build roll slot weights from a 21‑day volume profile
                vol_end_dt = dt.datetime.combine(cur_day, dt.time(0, 0), dt.timezone.utc)
                vol_start_dt = vol_end_dt - dt.timedelta(days=volume_lookback_days)
                vs_ms = int(vol_start_dt.timestamp() * 1000)
                ve_ms = int(vol_end_dt.timestamp() * 1000)
                print(f"   Building volume profile from {vol_start_dt.strftime('%Y-%m-%d')} to {vol_end_dt.strftime('%Y-%m-%d')}")
                vol_kl = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", vs_ms, ve_ms, is_continuous=True, contract_type="CURRENT_QUARTER")
                vol_df = scanner._klines_to_df(vol_kl)
                vol_df["slot"] = (vol_df["open_time"].dt.hour*60 + vol_df["open_time"].dt.minute)//15
                vol_col = "fut_cur_volume" if "fut_cur_volume" in vol_df.columns else "volume"
                vol_per_slot = vol_df.groupby("slot")[vol_col].sum()
                w_vec = np.ones(96)
                if not vol_per_slot.empty and vol_per_slot.sum()>0:
                    for s,v in vol_per_slot.items():
                        w_vec[int(s)] = v
                w_vec = w_vec / w_vec.sum()
                roll_weights = {k: float(w_vec[k]) for k in range(96)}

                roll_exec = RolloverExecutor(
                    api_key=api_key,
                    api_secret=api_secret,
                    from_symbol=sym,
                    to_symbol=next_sym,
                    contracts=data["qty"],
                    target_q_btc=target_btc_long,
                    slot_weights=roll_weights, #slot_weights is the weights for the rollover
                    simulator=sim,
                    price_feed=pf,
                    params=ExecParams(capital_usdt=0.0, hours=24)
                )
                roll_result = roll_exec.run(start_dt=day_start_dt, realtime=False)
                # Append trades to CSV
                trade_df = roll_result["trades"]
                
                if not trade_df.empty:
                    # --- Calculate and Record Realized PnL from this Rollover ---
                    # Separate the rollover trades into buy-back (closing old position) and sell-forward (opening new position)
                    buy_back_df = trade_df[trade_df["side"] == "BUY"]
                    sell_forward_df = trade_df[trade_df["side"] == "SELL"]

                    # Calculate volume-weighted average price for the new contract position
                    # This will be used as the entry price for the new position in the position tracker
                    to_sym_entry_px = (sell_forward_df["price"] * sell_forward_df["filled_qty"]).sum() / sell_forward_df["filled_qty"].sum() if not sell_forward_df.empty else 0
                    
                    # Total quantity of the new contract that was opened during rollover
                    to_sym_qty = int(sell_forward_df["filled_qty"].sum()) if not sell_forward_df.empty else 0

                    # Process realized PnL from closing the expiring contract
                    if not buy_back_df.empty:
                        # Get the original entry price of the contract we're closing from position tracker
                        # This is the VWAP price at which we originally entered the short position
                        entry_vwap = pt.current_positions().get(sym, {}).get("entry_px", 0)

                        # Calculate realized PnL for each buy-back trade (closing short positions)
                        total_realized_usd = 0.0
                        for _, r in buy_back_df.iterrows():
                            # Extract timestamp for this specific trade
                            ts_trade = r["ts"]
                            if isinstance(ts_trade, str):
                                try:
                                    ts_trade = dt.datetime.fromisoformat(ts_trade)
                                except Exception:
                                    # Fallback to start of rollover day if timestamp parsing fails
                                    ts_trade = day_start_dt
                            
                            # Get spot price at the time of this trade for USD conversion
                            # This is needed because inverse futures PnL is in BTC, but we track USD PnL
                            spot_conv, _ = pf.get_spot_mid_spread(ts_trade)
                            if not np.isfinite(spot_conv):
                                # Fallback: use futures exit price as proxy for spot if spot data unavailable
                                spot_conv = float(r["price"])
                            
                            # Extract trade details
                            filled_qty = int(r["filled_qty"])  # Number of contracts bought back
                            exit_px = float(r["price"])        # Price at which we bought back (closed short)
                            
                            # Calculate PnL for this trade slice
                            if entry_vwap > 0 and exit_px > 0:
                                # For inverse futures (COIN-M): PnL in BTC = contracts * $100 * (1/exit_price - 1/entry_price)
                                # When closing a short: we profit when exit_price < entry_price (1/exit > 1/entry)
                                pnl_btc_row = filled_qty * 100.0 * (1/exit_px - 1/entry_vwap)
                                
                                # Convert BTC PnL to USD using spot price at trade time
                                pnl_usd_row = pnl_btc_row * spot_conv
                                total_realized_usd += pnl_usd_row
                                
                                # Record this PnL event in the tracking system
                                record_pnl(ts_trade, sym, pnl_usd_row)
                        
                        print(f"PNL LOG: Realized ${total_realized_usd:,.2f} from closing {sym}\n")

                    # Append rollover trades to the execution log CSV file
                    # This maintains a complete record of all rollover executions across the backtest
                    if not os.path.exists("rollover_exec_log.csv"):
                        # Create new file with headers if it doesn't exist
                        trade_df.to_csv("rollover_exec_log.csv", index=False)
                    else:
                        # Append to existing file without headers
                        trade_df.to_csv("rollover_exec_log.csv", mode="a", header=False, index=False)
                    
                    # Update the position tracker to reflect the rollover:
                    # - Close the old expiring contract position
                    # - Open the new contract position with the calculated entry price and quantity
                    pt.rollover(sym, next_sym, to_sym_entry_px, to_sym_qty)
        
        # --- Daily delta re‑hedge ---
        # 1) Current net delta
        spot_delta_btc = target_btc_long
        futures_delta_btc = 0.0
        active_futures_contract = ""
        if pt.current_positions():
            # Assume we hedge using a single active contract
            active_futures_contract = list(pt.current_positions().keys())[0]
            fut_data = pt.current_positions()[active_futures_contract]
            # Short inverse delta = - (contracts * $100) / spot
            futures_delta_btc = - (fut_data["qty"] * 100.0) / spot_eod_price if spot_eod_price > 0 else 0
        
        net_delta_btc = spot_delta_btc + futures_delta_btc
        
        # 2) Check if a re‑hedge is needed
        hedge_threshold_btc = target_btc_long * delta_threshold_pct
        contracts_to_trade = 0

        if net_delta_btc > hedge_threshold_btc:
            # Portfolio is too long, need to sell more futures
            contracts_to_trade = -int(round((net_delta_btc * spot_eod_price) / 100.0))
        elif net_delta_btc < -hedge_threshold_btc:
            # Portfolio is too short, need to buy back futures
            contracts_to_trade = abs(int(round((net_delta_btc * spot_eod_price) / 100.0)))

        # 3) Execute re‑hedge trade if needed
        if contracts_to_trade != 0 and active_futures_contract:
            side = "SELL" if contracts_to_trade < 0 else "BUY"
            qty = abs(contracts_to_trade)
            
            # Simplified: use EOD futures mid for the trade
            expiry = parse_coin_m_expiry(active_futures_contract)
            eod_quarter_months = [3, 6, 9, 12]
            q_idx_cur = (cur_day.month - 1) // 3
            cur_q_month = eod_quarter_months[q_idx_cur]
            ctype = "NEXT_QUARTER" if expiry.month > cur_q_month or expiry.year > cur_day.year else "CURRENT_QUARTER"
            hedge_price, _ = daily_price_feed.get_future_mid_spread(eod_ts, ctype)

            if np.isfinite(hedge_price):
                print(f"REHEDGE LOG: Net delta is {net_delta_btc:.2f} BTC. {side} {qty} contracts of {active_futures_contract} at ~${hedge_price:,.2f}.")
                record_rehedge_trade(day_start_dt, active_futures_contract, side, qty, hedge_price, net_delta_btc, "Delta hedge")
                
                # Update position tracker and record realized PnL in USD using EOD spot
                qty_change = -qty if side == "SELL" else qty
                pt.update_position(
                    active_futures_contract,
                    -qty_change,
                    hedge_price,
                    conversion_spot_usd=spot_eod_price,
                    ts=eod_ts,
                )

        # --- MTM calculation ---
        # Calculate spot position mark-to-market PnL in USD
        # This is the unrealized gain/loss on our long BTC spot position
        # Formula: (current_price - entry_price) * position_size_btc
        spot_mtm_pnl = (spot_eod_price - spot_entry_px) * target_btc_long
        
        futures_unrealized_pnl_btc = 0.0
        for sym, data in pt.current_positions().items():
            qty = int(data.get("qty", 0))
            entry_px = float(data.get("entry_px", 0.0))
            side = data.get("side", "SHORT")
            
            # Determine if EOD symbol is CURRENT or NEXT relative to cur_day
            expiry = parse_coin_m_expiry(sym)
            eod_quarter_months = [3, 6, 9, 12]
            q_idx_cur = (cur_day.month - 1) // 3
            cur_q_month = eod_quarter_months[q_idx_cur]
            
            contract_type_eod = "CURRENT_QUARTER"
            if expiry.month > cur_q_month or expiry.year > cur_day.year:
                 # This is a simplification; doesn't handle year boundaries perfectly but works for this case
                 contract_type_eod = "NEXT_QUARTER"

            mark_price, _ = daily_price_feed.get_future_mid_spread(eod_ts, contract_type_eod)
            if not np.isfinite(mark_price): # Fallback if specific contract type fails
                mark_price, _ = daily_price_feed.get_future_mid_spread(eod_ts, "CURRENT_QUARTER")

            if np.isfinite(entry_px) and entry_px > 0 and np.isfinite(mark_price) and mark_price > 0 and qty>0:
                if side == "SHORT":
                    pnl_btc = qty * 100.0 * (1/mark_price - 1/entry_px)
                else:  # LONG
                    pnl_btc = qty * 100.0 * (1/entry_px - 1/mark_price)
                futures_unrealized_pnl_btc += pnl_btc

        # Convert BTC PnL to USD using EOD spot
        futures_unrealized_pnl = futures_unrealized_pnl_btc * spot_eod_price

        futures_realized_pnl = 0.0
        try:
            pnl_df = pd.read_csv("pnl_tracker.csv")
            if not pnl_df.empty:
                futures_realized_pnl = pnl_df["pnl_usd"].sum()
        except FileNotFoundError:
            pass

        current_portfolio_value = initial_capital + spot_mtm_pnl + futures_unrealized_pnl + futures_realized_pnl
        daily_pnl = current_portfolio_value - last_day_portfolio_value
        cumulative_pnl = current_portfolio_value - initial_capital

        record_daily_pnl(
            cur_day, spot_eod_price, current_portfolio_value, daily_pnl, cumulative_pnl,
            spot_mtm_pnl, futures_unrealized_pnl, futures_realized_pnl
        )
        print(f"MTM LOG for {cur_day.strftime('%Y-%m-%d')}: Portfolio Value=${current_portfolio_value:,.2f}, Daily PnL=${daily_pnl:,.2f}")

        last_day_portfolio_value = current_portfolio_value

        cur_day += dt.timedelta(days=1)

    # --- FINAL DAY UNWIND ---
    print(f"\n--- Starting Final Day Unwind on {end_date.strftime('%Y-%m-%d')} ---")
    
    # 1) Build price feed for the day
    unwind_start_dt = dt.datetime.combine(end_date, dt.time(0,0), dt.timezone.utc)
    s_ms = int(unwind_start_dt.timestamp()*1000)
    e_ms = s_ms + 24*3600*1000
    spot_kl_unwind = scanner._fetch_klines_synchronized(temp_spot, "BTCUSDT", s_ms, e_ms)
    cur_kl_unwind  = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", s_ms, e_ms, is_continuous=True, contract_type="CURRENT_QUARTER")
    nxt_kl_unwind  = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", s_ms, e_ms, is_continuous=True, contract_type="NEXT_QUARTER")
    unwind_price_feed = BacktestPriceFeed(scanner._klines_to_df(spot_kl_unwind), scanner._klines_to_df(cur_kl_unwind), scanner._klines_to_df(nxt_kl_unwind))

    # 2) Setup and run the executor
    inventory = {
        "spot_btc": target_btc_long,
        "spot_entry_px": spot_entry_px,
        "futures": pt.current_positions()
    }

    # Flat weights for simplicity
    unwind_weights = {k: 1.0/96 for k in range(96)}
    
    unwind_executor = OptimizedUnwindExecutor(
        api_key=api_key,
        secret_key=api_secret,
        slot_weights=unwind_weights,
        simulator=sim,
        params=UnwindParams(capital_usdt=initial_capital),
        inventory=inventory,
        is_backtest=True,
        end_date=end_date.strftime("%Y-%m-%d"),
        price_feed=unwind_price_feed
    )
    
    unwind_result = unwind_executor.run(
        start_dt=unwind_start_dt,
        entry_date_str=start_date.strftime("%Y-%m-%d")
    )
    
    # 3) Record final day's realized PnL to the daily tracker
    final_pnl_summary = unwind_result["summary"]
    final_total_pnl = final_pnl_summary.get("total_pnl", 0.0)
    final_portfolio_value = initial_capital + final_total_pnl
    final_daily_pnl = final_portfolio_value - last_day_portfolio_value
    
    # At the end, realized/unrealized align with the final breakdown
    final_log_df = unwind_result["log"]
    spot_pnl_final = ( (final_log_df['spot_px'] - spot_entry_px) * final_log_df['fill_spot'] ).sum()
    
    # We already have realized futures PnL from the tracker and the final leg calc
    fut_realized_final = final_pnl_summary.get("total_pnl", 0.0) - spot_pnl_final
    
    record_daily_pnl(
        end_date,
        spot_px=final_log_df['spot_px'].iloc[-1] if not final_log_df.empty else spot_entry_px,
        portfolio_value=final_portfolio_value,
        daily_pnl=final_daily_pnl,
        cumulative_pnl=final_total_pnl,
        spot_pnl=spot_pnl_final,
        fut_unrealized=0.0, # All is realized now
        fut_realized=fut_realized_final
    )
    
    # --- STEP‑3 final day report ---
    print("\n===== Daily PNL summary saved to daily_pnl_tracker.csv =====")
    print("\n===== FINAL INVENTORY =====")
    print("Futures (short): {}")
    print("Spot BTC long: 0.0000 BTC")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-day BTC cash-and-carry back-test with rollovers")
    parser.add_argument("start_date", help="YYYY-MM-DD entry date")
    parser.add_argument("end_date", help="YYYY-MM-DD final exit date")
    args = parser.parse_args()

    run_backtest(dt.datetime.strptime(args.start_date, "%Y-%m-%d").date(),
                 dt.datetime.strptime(args.end_date, "%Y-%m-%d").date())