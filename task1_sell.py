import time, datetime as dt
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
# Import Binance clients
from binance.spot import Spot
from binance.cm_futures import CMFutures
from task1 import SynchronizedCashCarryScanner, BacktestPriceFeed, AdaptiveCashCarryExecutor, ExecParams, BinanceSimulator

# --- Load inventory produced by backtest_with_rollovers.py ---
import csv, sys, os

# def load_inventory(path: str) -> Dict:
#     """Load inventory CSV from the multi‑day backtest.

#     Expected columns: leg ("spot" or futures), symbol, qty, entryPx.
#     Returns: {"spot_btc", "spot_entry_px", "futures": symbol → {qty, entry_px}}.
#     """
#     spot_btc = 0.0; spot_px = np.nan; futs = {}
#     with open(path, newline="") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             if row["leg"]=="spot":
#                 spot_btc = float(row["qty"]); spot_px = float(row["entryPx"])
#             else:
#                 futs[row["symbol"]] = {
#                     "qty": int(row["qty"]),
#                     "entry_px": float(row["entryPx"]) if row["entryPx"] else 0.0
#                 }
#     return {"spot_btc":spot_btc,"spot_entry_px":spot_px,"futures":futs}

# # --- Optimized Unwind Weight Calculation ---
# def derive_hybrid_unwind_weights(spot_klines: list, fut_klines: list, unwind_date: str) -> Dict[int, float]:
#     """
#     Compute 96 unwind weights from spot/futures for one day:
#     - 50% volume (VWAP tilt)
#     - 30% basis convergence (closer to 0 → higher)
#     - 20% liquidity (tighter spreads → higher)
#     """
#     # Shape raw klines into DataFrames
#     spot_df = pd.DataFrame(spot_klines, columns=["open_time","open","high","low","close","volume",
#                                                  "close_time","quote_asset_volume","num_trades","taker_base","taker_quote","ignore"])
#     fut_df = pd.DataFrame(fut_klines, columns=["open_time","open","high","low","close","volume",
#                                               "close_time","quote_asset_volume","num_trades","taker_base","taker_quote","ignore"])
    
#     # Convert timestamps and numeric data
#     spot_df["open_time"] = pd.to_datetime(spot_df["open_time"], unit="ms", utc=True)
#     fut_df["open_time"] = pd.to_datetime(fut_df["open_time"], unit="ms", utc=True)
    
#     for df in [spot_df, fut_df]:
#         for c in ["close", "volume", "high", "low"]:
#             df[c] = pd.to_numeric(df[c], errors="coerce")
    
#     # 15-minute slots (96 per day)
#     spot_df["slot"] = (spot_df["open_time"].dt.hour * 60 + spot_df["open_time"].dt.minute) // 15
#     fut_df["slot"] = (fut_df["open_time"].dt.hour * 60 + fut_df["open_time"].dt.minute) // 15
    
#     # Volume weights (50%): heavier where traded volume is higher

#     #this is wrong as the volume is taken till T
#     volume_by_slot = spot_df.groupby("slot")["volume"].sum()
#     volume_sum = volume_by_slot.sum()
#     if volume_sum > 0 and not pd.isna(volume_sum):
#         volume_weights = volume_by_slot / volume_sum
#     else:
#         volume_weights = pd.Series({slot: 1/96 for slot in range(96)})
    
#     # Basis convergence (30%): weight slots where basis is nearer 0
#     basis_by_slot = {}
#     for slot in range(96):
#         spot_data = spot_df[spot_df["slot"] == slot]
#         fut_data = fut_df[fut_df["slot"] == slot]
        
#         if not spot_data.empty and not fut_data.empty:
#             avg_spot = spot_data["close"].mean()
#             avg_fut = fut_data["close"].mean()
#             basis_bps = (avg_fut - avg_spot) / avg_spot * 1e4
            
#             # Lower absolute basis → higher weight
#             convergence_score = max(0, 100 - abs(basis_bps)) / 100  # 0..1
#             basis_by_slot[slot] = convergence_score
#         else:
#             basis_by_slot[slot] = 0.1  # Default low weight
    
#     basis_weights = pd.Series(basis_by_slot)
#     basis_sum = basis_weights.sum()
#     if basis_sum > 0 and not pd.isna(basis_sum):
#         basis_weights = basis_weights / basis_sum
#     else:
#         basis_weights = pd.Series({slot: 1/96 for slot in range(96)})
    
#     # Liquidity (20%): spread as inverse proxy for depth
#     liquidity_by_slot = {}
#     for slot in range(96):
#         slot_data = spot_df[spot_df["slot"] == slot]
#         if not slot_data.empty:
#             # Use spread as inverse liquidity proxy
#             avg_spread = (slot_data["high"] - slot_data["low"]).mean()
#             liquidity_score = 1 / (1 + avg_spread)  # Lower spread = higher liquidity
#             liquidity_by_slot[slot] = liquidity_score
#         else:
#             liquidity_by_slot[slot] = 0.1
    
#     liquidity_weights = pd.Series(liquidity_by_slot)
#     liquidity_sum = liquidity_weights.sum()
#     if liquidity_sum > 0 and not pd.isna(liquidity_sum):
#         liquidity_weights = liquidity_weights / liquidity_sum
#     else:
#         liquidity_weights = pd.Series({slot: 1/96 for slot in range(96)})
    
#     # Combine 50% volume, 30% convergence, 20% liquidity
#     final_weights = {}
#     for slot in range(96):
#         vol_w = volume_weights.get(slot, 1/96)
#         basis_w = basis_weights.get(slot, 1/96)
#         liq_w = liquidity_weights.get(slot, 1/96)
        
#         # Safety check for NaN values
#         if pd.isna(vol_w): vol_w = 1/96
#         if pd.isna(basis_w): basis_w = 1/96
#         if pd.isna(liq_w): liq_w = 1/96
        
#         combined_weight = 0.5 * vol_w + 0.3 * basis_w + 0.2 * liq_w
#         final_weights[slot] = max(0.001, combined_weight)  # Minimum weight to avoid zeros
    
#     # Normalize to sum to 1 (safety check)
#     total_weight = sum(final_weights.values())
#     if total_weight > 0 and not pd.isna(total_weight):
#         final_weights = {k: v/total_weight for k, v in final_weights.items()}
#     else:
#         final_weights = {k: 1/96 for k in range(96)}  # Equal weights fallback
    
#     return final_weights

# def get_live_volume_adjustment(timestamp: dt.datetime, recent_hours: int = 4) -> float:
#     """Simple stub: multiplier to tilt for live volume (VWAP)."""
#     # Simplified - in practice would fetch live volume data
#     # Returns multiplier: >1 if current volume > historical, <1 if lower
#     current_hour = timestamp.hour
    
#     # Peak hours get higher multiplier (simulate high volume times)
#     peak_hours = [8, 9, 13, 14, 15, 16]  # Asia morning + US/EU overlap
#     if current_hour in peak_hours:
#         return 1.5
#     elif 6 <= current_hour <= 18:  # Business hours
#         return 1.2
#     else:  # Off hours
#         return 0.8

# def estimate_basis_convergence_score(current_basis: float, target_basis: float = 5.0) -> float:
#     """Score how close the futures-spot basis is to convergence (0-1 scale)."""
#     if abs(current_basis) <= target_basis:
#         return 1.0  # Perfect convergence
#     elif abs(current_basis) <= target_basis * 2:
#         return 0.7  # Close to convergence
#     elif abs(current_basis) <= target_basis * 4:
#         return 0.4  # Moderate
#     else:
#         return 0.1  # Far from convergence

# # --- Cost and Signal Helpers ---
# def estimate_slippage(notional: float, depth: float, vol_factor: float = 1.0) -> float:
#     """Estimate slippage in basis points (bps) based on notional and L2 depth."""
#     return (notional / depth) * vol_factor * 1e4 if depth > 0 else 10.0  # Fallback 10 bps

# def fit_ar1_basis(basis_series: pd.Series, lag: int = 1) -> Dict:
    # """Fit an AR(1) (first-order autoregression) model on the basis series for mean-reversion signals."""
    # from statsmodels.tsa.ar_model import AutoReg
    # model = AutoReg(basis_series, lags=lag).fit()
    # return {"coef": model.params.iloc[1], "intercept": model.params.iloc[0], "residuals": model.resid}

# --- Core Classes and Functions (Mirroring task1.py) ---
@dataclass
class UnwindParams(ExecParams):
    reserve_pct: float = 0.05  # 5% reserve for late-day opportunism

class OptimizedUnwindExecutor(AdaptiveCashCarryExecutor):
    """Unwind the cash‑and‑carry over 24h.

    Sells spot BTC and buys back the short inverse futures.
    Reuses the parent's scheduling and price‑feed utilities.
    """
    def __init__(self, api_key: str, secret_key: str, slot_weights: Dict[int, float], simulator: BinanceSimulator, 
                 params: UnwindParams, inventory: Dict, is_backtest: bool = False, end_date: str = None,
                 price_feed: BacktestPriceFeed = None):
        
        # Init parent executor
        #super is used to call the parent class
        super().__init__(
            # api_key=api_key,
            # api_secret=secret_key,
            api_key="B9qHPC4CkJ8gvQz9q5t7M09YUJ1VDDVNsNOmdb7zFud8dPAFqEF30gvEX6OPTebo",
            api_secret="LGL1Qo4Gx53HBUYlBy0oEJujH4uUwp2papB4ecxg0OprFrimW8sofmksfrWfmk5U",
            slot_weights=slot_weights,
            simulator=simulator,
            params=params,
            price_feed=price_feed,
            historical_avg_basis=200.0  # Default, can be adjusted
        )
        
        self.inventory = inventory
        self.is_backtest = is_backtest
        self.end_date = end_date
        self.exec_log: List[Dict] = []
        self.arrival_spot_mid = inventory["spot_entry_px"]
        self.target_basis_bps = 2.0  # More aggressive threshold for basis ~0
        np.random.seed(params.rng_seed)

    def run(self, start_dt: Optional[dt.datetime] = None, realtime: bool = False, entry_date_str: str = "") -> Dict:
        """Run the 24h unwind and return summary/log.

        Time‑weighted schedule with PoV sizing; pricing adapts to schedule.
        """
        P = self.params
        start_dt = start_dt or dt.datetime.now(dt.timezone.utc)
        buckets = int((P.hours * 60) // P.bucket_minutes)
        is_last_day = self.is_backtest and start_dt.date() == dt.datetime.strptime(self.end_date, "%Y-%m-%d").date()
        
        # Positions from entry
        spot_btc_to_sell = self.inventory["spot_btc"]
        fut_positions = self.inventory["futures"]
        fut_ct_to_buy_back = sum(v['qty'] for v in fut_positions.values())

        if spot_btc_to_sell <= 0 or fut_ct_to_buy_back <= 0:
            print("⚠️ No inventory to unwind.")
            return {
                "summary": {"r_ann": 0, "filled_btc": 0, "filled_ct": 0},
                "log": pd.DataFrame()
            }
        
        # Create a mapping of contracts to unwind, ordered by entry price (cheapest first for FIFO)
        contracts_to_unwind = []
        for symbol, data in fut_positions.items():
            if data['qty'] > 0:
                contracts_to_unwind.append({
                    'symbol': symbol,
                    'qty': data['qty'],
                    'entry_px': data.get('entry_px', 0.0),
                    'entry_inv': data.get('entry_inv', 0.0),
                    'side': data.get('side', 'SHORT')
                })
        
        # Sort by entry price (cheapest first for FIFO unwinding)
        contracts_to_unwind.sort(key=lambda x: x['entry_px'])
        
        print(f"Contracts to unwind: {len(contracts_to_unwind)}")
        for i, contract in enumerate(contracts_to_unwind):
            print(f"  {i+1}. {contract['symbol']}: {contract['qty']} contracts @ ${contract['entry_px']:.2f} (inv: {contract['entry_inv']:.6f})")
        
        if not contracts_to_unwind:
            print("⚠️ No valid contracts found in inventory to unwind.")
            return {
                "summary": {"r_ann": 0, "filled_btc": 0, "filled_ct": 0},
                "log": pd.DataFrame()
            }
            
        remaining_btc = spot_btc_to_sell
        remaining_ct = fut_ct_to_buy_back
        filled_btc_unwind = 0.0
        filled_ct_unwind = 0
        nav_start = P.capital_usdt
        
        # Use existing weights
        unwind_weights = self.slot_weights
        
        # Execution Mode Indicator
        mode_str = "Backtest mode" if self.is_backtest else "Live trading mode"
        print(f"Starting unwind execution")
        print(f"Mode: {mode_str}")
        print(f"Start Date: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Target to Unwind: {spot_btc_to_sell:.4f} BTC, {fut_ct_to_buy_back} contracts")
        print(f"Historical Avg Basis: {self.historical_avg_basis:.1f} bps")
        print(f"Target Basis (≈0): ≤{self.target_basis_bps:.1f} bps")
        print("=" * 80)
        
        # Using Pure TWAP
        weights_vec = np.array([unwind_weights.get(k, 1.0/96) for k in range(96)], dtype=float)
        weights_vec = weights_vec / weights_vec.sum()
        
        for i in range(buckets):
            ts = start_dt + dt.timedelta(minutes=P.bucket_minutes * i)
            utc_slot = ((ts.hour * 60 + ts.minute) // 15) % 96
            w = float(weights_vec[utc_slot])
            
            # Fetch live data using parent's methods (which use price_feed)
            spot_mid, spot_spread = self._spot_mid_spread(ts)
            
            # Use one futures mid for basis; fall back CURRENT→NEXT.
            # In production, map to the actual expiring contract.
            fut_mid, fut_spread = self._future_mid_spread_by_contractType("CURRENT_QUARTER", ts)
            if not np.isfinite(fut_mid):
                fut_mid, fut_spread = self._future_mid_spread_by_contractType("NEXT_QUARTER", ts)
            if not np.isfinite(fut_mid): # Final fallback
                fut_mid = spot_mid * 1.0005
                fut_spread = 5.0

            cur_bps = (fut_mid - spot_mid) / spot_mid * 1e4 if spot_mid > 0 else 0
            
            # Participation of Volume (PoV) sizing
            recent_vol_btc = self._recent_15m_spot_volume_btc(ts)
            time_frac = (i + 1) / buckets
            progress = (filled_btc_unwind / spot_btc_to_sell) if spot_btc_to_sell > 0 else 1.0
            sched_dev = progress - time_frac # > 0 is ahead of schedule
            
            pov = P.pov_base * (1 + 0.8 * (-sched_dev)) # -sched_dev is the deviation from the schedule
            pov = max(P.pov_min, min(P.pov_max, pov))
            
            # Child sizing: combine schedule weight and PoV cap
            child_from_weight = remaining_btc * w #remaining_btc is the remaining BTC to unwind
            child_from_pov = pov * recent_vol_btc #recent_vol_btc is the recent 15m spot volume in BTC
            child_btc = min(remaining_btc, min(child_from_weight, child_from_pov))
            
            if child_btc <= 1e-6:
                continue

            # Select the contract to unwind (FIFO - first in, first out)
            # Find the first contract that still has remaining quantity
            current_contract = None
            for contract in contracts_to_unwind:
                if contract['qty'] > 0:
                    current_contract = contract
                    break
            
            if current_contract is None:
                print("No more contracts to unwind")
                break
                
            fut_symbol_to_buy = current_contract['symbol']

            # Calculate equivalent contracts to unwind based on BTC ratio
            # This ensures we unwind contracts proportionally to the BTC being sold
            child_ct = int(child_btc / spot_btc_to_sell * fut_ct_to_buy_back) if spot_btc_to_sell > 0 else 0
            child_ct = min(remaining_ct, max(1, child_ct))
            
            # Ensure we don't try to unwind more contracts than available for the current contract
            child_ct = min(child_ct, current_contract['qty'])

            # Pricing: go more aggressive if we're behind schedule
            be_aggressive = sched_dev < -P.behind_sched_thresh

            # For a SELL order: passive is placing on the ASK, aggressive is hitting the BID.
            spot_px = float(spot_mid - 0.5 * spot_spread) if be_aggressive else float(spot_mid + 0.5 * spot_spread)
            # For a BUY order: passive is placing on the BID, aggressive is hitting the ASK.
            fut_px  = float(fut_mid  + 0.5 * fut_spread) if be_aggressive else float(fut_mid - 0.5 * fut_spread)

            spot_order = {"market": "SPOT", "symbol": "BTCUSDT", "side": "SELL", "type": "LIMIT", 
                          "price": spot_px, "quantity": round(child_btc, 6)}
            fut_order = {"market": "FUT", "symbol": fut_symbol_to_buy, "side": "BUY", "type": "LIMIT", 
                         "price": fut_px, "contracts": child_ct}
            
            spot_resp = self._place_until_filled(spot_order, P.max_bucket_attempts)
            fut_resp = self._place_until_filled(fut_order, P.max_bucket_attempts)
            
            # Accumulate and log
            fill_spot = spot_resp.get("filled_qty", 0.0)
            fill_ct = fut_resp.get("filled_qty", 0)
            
            filled_btc_unwind += fill_spot
            filled_ct_unwind += fill_ct
            remaining_btc = max(0, spot_btc_to_sell - filled_btc_unwind)
            remaining_ct = max(0, fut_ct_to_buy_back - filled_ct_unwind)
            
            # Update the current contract's remaining quantity
            if current_contract and fill_ct > 0:
                current_contract['qty'] = max(0, current_contract['qty'] - fill_ct)
            
            self.exec_log.append({
                "ts": ts, "slot": utc_slot, "weight": w, "pov": pov, "sched_dev": sched_dev,
                "basis_bps": cur_bps,
                "child_btc": child_btc, "child_ct": child_ct, "spot_px": spot_px, "fut_px": fut_px,
                "fill_spot": fill_spot, "fill_ct": fill_ct,
                "remaining_btc": remaining_btc, "remaining_ct": remaining_ct,
                "fut_symbol": fut_symbol_to_buy,
                "fut_entry_px": current_contract['entry_px'] if current_contract else 0.0,
                "fut_entry_inv": current_contract['entry_inv'] if current_contract else 0.0,
            })
            
            print(f"[{ts.strftime('%H:%M')} UTC] Unwound BTC {fill_spot:.4f}, CT {fill_ct} ({fut_symbol_to_buy}), Progress: {(filled_btc_unwind / spot_btc_to_sell)*100:.1f}%, Basis: {cur_bps:.1f}bps")
            
            if remaining_btc <= 1e-6 or remaining_ct <= 0:
                break
        
        # Final sweep: guarantee completion if anything remains
        if remaining_btc > 0.0 or remaining_ct > 0:
            final_ts = start_dt + dt.timedelta(minutes=P.bucket_minutes * buckets)
            print(f"Forcing 100% completion - Remaining: {remaining_btc:.4f} BTC, {remaining_ct} contracts")
            
            original_prob = self.sim.order_fill_prob
            try:
                self.sim.order_fill_prob = 1.0 # Ensure fill

                spot_mid, spot_spread = self._spot_mid_spread(final_ts)
                
                # Find the first contract that still has remaining quantity
                final_contract = None
                for contract in contracts_to_unwind:
                    if contract['qty'] > 0:
                        final_contract = contract
                        break
                
                if final_contract:
                    fut_symbol_to_buy = final_contract['symbol']
                    fut_mid, fut_spread = self._future_mid_spread_by_symbol(fut_symbol_to_buy, final_ts)
                    if not np.isfinite(fut_mid):
                        fut_mid, fut_spread = self._future_mid_spread_by_contractType("CURRENT_QUARTER", final_ts)
                else:
                    # Fallback if no contracts found
                    fut_symbol_to_buy = "BTCUSD_240927"  # Default fallback
                    fut_mid, fut_spread = self._future_mid_spread_by_contractType("CURRENT_QUARTER", final_ts)
                
                forced_fill_spot = 0.0
                forced_fill_ct = 0.0
                forced_spot_px = np.nan
                forced_fut_px = np.nan

                if remaining_btc > 0:
                    # Final sweep is always aggressive to guarantee completion
                    spot_px = float(spot_mid - 0.5 * spot_spread)
                    spot_order = {"market": "SPOT", "symbol": "BTCUSDT", "side": "SELL", "type": "LIMIT", 
                                  "price": spot_px, "quantity": round(remaining_btc, 6)}
                    spot_resp = self._place_until_filled(spot_order, 1)
                    fill_spot = float(spot_resp.get("filled_qty", 0.0))
                    filled_btc_unwind += fill_spot
                    forced_fill_spot = fill_spot
                    forced_spot_px = spot_px
                
                if remaining_ct > 0:
                    # Final sweep is always aggressive
                    fut_px = float(fut_mid + 0.5 * fut_spread)
                    fut_order = {"market": "FUT", "symbol": fut_symbol_to_buy, "side": "BUY", "type": "LIMIT", 
                                 "price": fut_px, "contracts": remaining_ct}
                    fut_resp = self._place_until_filled(fut_order, 1)
                    fill_ct = float(fut_resp.get("filled_qty", 0.0))
                    filled_ct_unwind += fill_ct
                    forced_fill_ct = fill_ct
                    forced_fut_px = fut_px

            finally:
                self.sim.order_fill_prob = original_prob

            # Add a synthetic row for the forced completion so reporting includes it
            try:
                utc_slot = ((final_ts.hour * 60 + final_ts.minute) // 15) % 96
                cur_bps = (fut_mid - spot_mid) / spot_mid * 1e4 if spot_mid and spot_mid > 0 else 0.0
                self.exec_log.append({
                    "ts": final_ts,
                    "slot": int(utc_slot),
                    "weight": 0.0,
                    "pov": 0.0,
                    "sched_dev": 0.0,
                    "basis_bps": float(cur_bps),
                    "child_btc": float(remaining_btc),
                    "child_ct": float(remaining_ct),
                    "spot_px": float(forced_spot_px if np.isfinite(forced_spot_px) else spot_mid),
                    "fut_px": float(forced_fut_px if np.isfinite(forced_fut_px) else fut_mid),
                    "fill_spot": float(forced_fill_spot),
                    "fill_ct": float(forced_fill_ct),
                    "remaining_btc": 0.0,
                    "remaining_ct": 0.0,
                })
            except Exception:
                pass

        remaining_btc = max(0, spot_btc_to_sell - filled_btc_unwind)
        remaining_ct = max(0, fut_ct_to_buy_back - filled_ct_unwind)

        # Optional: write detailed unwind log (disabled)
        # pd.DataFrame(self.exec_log).to_csv("unwind_exec_log.csv", index=False)

        # Final Status
        unwind_completion = (filled_btc_unwind / spot_btc_to_sell) * 100 if spot_btc_to_sell > 0 else 100
        
        print("=" * 80)
        print(f"Unwind execution completed")
        print(f"Mode: {mode_str}")
        print(f"Completion Rate: {unwind_completion:.1f}%")
        print(f"Remaining BTC: {remaining_btc:.6f}")
        print(f"Remaining Contracts: {remaining_ct}")
        print(f"Total Execution Buckets: {len(self.exec_log)}")
        print("=" * 80)
        
        # Compute PnL from detailed fills
        exec_df = pd.DataFrame(self.exec_log)
        if exec_df.empty:
             return {"summary": {"r_ann": 0.0, "total_pnl": 0.0, "filled_btc": 0, "filled_ct": 0}, "log": exec_df}

        # Spot PnL
        spot_fills = exec_df["fill_spot"].astype(float)
        spot_prices = exec_df["spot_px"].astype(float)
        btc_unwound = float(spot_fills.sum())
        spot_entry = float(self.inventory["spot_entry_px"]) #vwap price at which we entered the spot position
        spot_pnl = float(((spot_prices - spot_entry) * spot_fills).sum())

        # Calculate futures PnL for each contract that was unwound
        contract_multiplier = 100.0
        fut_pnl = 0.0
        ct_unwound = 0.0
        
        # Group execution log by futures symbol to calculate PnL per contract
        if 'fut_symbol' in exec_df.columns:
            for symbol in exec_df['fut_symbol'].unique():
                if pd.isna(symbol):
                    continue
                    
                symbol_fills = exec_df[exec_df['fut_symbol'] == symbol]
                if symbol_fills.empty:
                    continue
                
                # Get entry price for this symbol from inventory
                symbol_meta = fut_positions.get(symbol, {})
                entry_px = float(symbol_meta.get('entry_px', 0.0))
                entry_inv = float(symbol_meta.get('entry_inv', 0.0))
                
                if entry_px <= 0 and entry_inv <= 0:
                    print(f"Warning: No valid entry price for {symbol}, skipping PnL calculation")
                    continue
                
                # Calculate PnL for this symbol
                fut_prices = symbol_fills["fut_px"].astype(float)
                fut_fills = symbol_fills["fill_ct"].astype(float)
                spot_at_fill = symbol_fills["spot_px"].astype(float)
                
                # For inverse futures (COIN-M): PnL in BTC = contracts * $100 * (1/exit_price - 1/entry_price)
                # When closing a short: we profit when exit_price < entry_price (1/exit > 1/entry)
                if entry_inv > 0:
                    # Use entry_inv directly (it's already 1/price)
                    fut_pnl_btc = contract_multiplier * fut_fills * ((1.0/fut_prices) - entry_inv)
                else:
                    # Calculate 1/entry_price from entry_px
                    fut_pnl_btc = contract_multiplier * fut_fills * ((1.0/fut_prices) - (1.0/entry_px))
                
                # Convert BTC PnL to USD using spot price at fill time
                fut_pnl_series_usd = fut_pnl_btc * spot_at_fill
                symbol_pnl = float(fut_pnl_series_usd.sum())
                symbol_ct = float(fut_fills.sum())
                
                fut_pnl += symbol_pnl
                ct_unwound += symbol_ct
                
                print(f"   {symbol}: {symbol_ct:.0f} contracts, PnL: ${symbol_pnl:,.2f}")
        else:
            # Fallback: calculate PnL using the old method if fut_symbol column doesn't exist
            print("Warning: fut_symbol column not found, using fallback PnL calculation")
            fut_prices = exec_df["fut_px"].astype(float)
            fut_fills = exec_df["fill_ct"].astype(float)
            spot_at_fill = exec_df["spot_px"].astype(float)
            ct_unwound = float(fut_fills.sum())
            
            # Use average entry price from all contracts
            total_entry_px = 0.0
            total_weight = 0.0
            for symbol, data in fut_positions.items():
                if data['qty'] > 0:
                    entry_px = float(data.get('entry_px', 0.0))
                    if entry_px > 0:
                        total_entry_px += entry_px * data['qty']
                        total_weight += data['qty']
            
            avg_entry_px = total_entry_px / max(total_weight, 1.0) if total_weight > 0 else 0.0
            
            if avg_entry_px > 0:
                fut_pnl_btc = contract_multiplier * fut_fills * ((1.0/fut_prices) - (1.0/avg_entry_px))
                fut_pnl_series_usd = fut_pnl_btc * spot_at_fill
                fut_pnl = float(fut_pnl_series_usd.sum())
            else:
                fut_pnl = 0.0

        # Include realized PnL from rollovers
        realized_rollover_pnl = 0.0
        try:
            pnl_df = pd.read_csv("pnl_tracker.csv")
            if not pnl_df.empty:
                realized_rollover_pnl = pnl_df["pnl_usd"].sum()
                print(f"   Rollover PnL: ${realized_rollover_pnl:,.2f} (from {len(pnl_df)} rolls)")
        except FileNotFoundError:
            print("   Rollover PnL: $0.00 (no rolls)")

        # Total PnL
        total_pnl = spot_pnl + fut_pnl + realized_rollover_pnl
        nav_end = nav_start + total_pnl
        
        # Time calculation from entry to unwind date
        entry_date = dt.datetime.strptime(entry_date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        unwind_date = dt.datetime.strptime(self.end_date or "2021-10-01", "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        days_held = (unwind_date - entry_date).days
        
        r_ann = (nav_end / nav_start) ** (365 / max(days_held, 1)) - 1 if days_held > 0 and nav_start > 0 else 0
        
        # PnL calculation summary
        print(f"PnL breakdown:")
        print(f"   BTC Unwound: {btc_unwound:.4f} | Spot PnL: ${spot_pnl:,.2f}")
        print(f"   Contracts Unwound: {ct_unwound:.2f} | Futures PnL (Final Leg): ${fut_pnl:,.2f}")
        print(f"   Futures PnL (Rollovers): ${realized_rollover_pnl:,.2f}")
        print(f"   Total PnL: ${total_pnl:,.2f}")
        print(f"   NAV: ${nav_start:,.0f} → ${nav_end:,.0f}")
        print(f"   Days Held: {days_held}")
        
        return {
            "summary": {"r_ann": r_ann, "total_pnl": total_pnl, "filled_btc": btc_unwound, "filled_ct": ct_unwound, "spot_pnl": spot_pnl, "fut_pnl": float(fut_pnl)},
            "log": pd.DataFrame(self.exec_log)
        }