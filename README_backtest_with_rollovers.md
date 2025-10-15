## README — @backtest_with_rollovers.py: Multi‑day Backtest with Rollovers

What this is
- I run a full entry→hold→roll→unwind backtest between two dates. I open the cash‑and‑carry on the start date (via `task1.py`), step day‑by‑day, roll near‑expiry COIN‑M shorts, keep the book roughly delta‑neutral, and unwind everything on the end date. I track daily and realized PnL along the way.

Inputs
- Dates: `start_date`, `end_date` (YYYY‑MM‑DD)
- Capital: 1,000,000 USDT on entry day
- Roll settings: `roll_window_days = 2`, `volume_lookback_days = 21`
- Hedge setting: `delta_threshold_pct = 0.05` (re‑hedge if net delta > 5% of spot)

High‑level flow
1) Entry day (via `task1.py`)
   - I run the synchronized analysis, derive 96 weights, and execute the 24‑hour entry (long spot, short COIN‑M). I save `entry_day_exec_log.csv` and the target BTC.

2) Daily loop (for each day until the end date)
   - Build a daily price feed (spot, CURRENT, NEXT) aligned to the same minute window.
   - Rollover check per active short:
     - If days to expiry ≤ `roll_window_days`, I roll the expiring symbol into the next quarter. I build 96 roll weights from a 21‑day volume profile (liquidity‑tilt), then run a 24‑hour roll using `RolloverExecutor`.
     - I record realized rollover PnL (inverse COIN‑M). Per fill: BTC PnL = `100 × qty × (1/buyback_vwap_old − 1/entry_vwap_old)`; USD PnL = BTC PnL × `spot_at_fill`. Opening the new short sets a fresh entry_vwap.
     - I append roll trades to `rollover_exec_log.csv` and update the `PositionTracker` with the new symbol/entry.
   - Delta re‑hedge at EOD (simple, threshold‑based):
     - Net delta = spot_BTC − (contracts × 100 / spot_price). If |net_delta| > threshold, I trade EOD futures to rebalance and log to `rehedge_log.csv`. I update `PositionTracker` and realize inverse PnL for any buy‑backs.
   - Daily MTM + PnL tracking:
     - Spot MTM PnL = (EOD spot − spot_entry) × BTC.
     - Futures unrealized PnL (inverse, short): `qty × 100 × (1/mark_price − 1/entry_px)`.
     - Futures realized PnL: sum of `pnl_tracker.csv` (rolls + re‑hedges recorded as they happen).
     - I write a daily row to `daily_pnl_tracker.csv` with portfolio value, daily and cumulative PnL, plus spot/unrealized/realized splits.

3) Final day unwind (via `task1_sell.py`)
   - I build a 24‑hour price feed for the end date and run `OptimizedUnwindExecutor` with flat weights. This sells spot and buys back the remaining shorts in paced slices.
   - The executor computes spot PnL and the final futures realized PnL per fill as: BTC PnL = `100 × fill_ct × (1/exit_px − 1/entry_vwap)`; USD PnL = BTC PnL × `spot_at_fill`.
   - I fold the final realized PnL into the daily tracker and print a summary. Writing a final fill log to `unwind_exec_log.csv` is disabled by default.

How the rollover execution is paced (TWAP + VWAP + PoV)
- TWAP backbone: I spread the roll over 96 equal slots to finish in 24 hours.
- VWAP tilt: I tilt slices using a 21‑day volume profile (liquidity), so more size goes into busier windows.
- PoV cap: Each slice is capped at 15% of recent 15‑minute spot BTC volume to respect liquidity.
- Pricing: passive by default; if I am behind schedule or spreads widen, I cross more aggressively.

How the final‑day unwind is paced (TWAP + PoV)
- TWAP backbone: I distribute the unwind across 96 equal 15‑minute slots (flat weights: 1/96 each) to fully exit within 24 hours.
- PoV cap: Each slice is capped by a Participation of Volume limit based on recent 15‑minute spot BTC volume and execution parameters (`pov_base` clipped to `[pov_min, pov_max]`).
- Schedule tracking: I track schedule deviation (`progress − time_fraction`). If behind more than a threshold, I increase aggressiveness.
- Sizing: For each slice, `child_btc = min(remaining_btc × slot_weight, pov × recent_15m_spot_volume_btc)`. Futures contracts are sized proportionally to the remaining hedge.
- Pricing: Behind schedule → more aggressive (sell spot toward bid; buy futures toward ask). On schedule → passive (sell spot toward ask; buy futures toward bid).
- Execution: I place paired orders (SELL spot, BUY futures) and retry within the slice until filled (bounded attempts).
- Final sweep: After the last slot, if anything remains, I force completion at marketable prices to guarantee a flat book.

Outputs and logs
- `entry_day_exec_log.csv`: fills for the initial 24‑hour entry
- `rollover_exec_log.csv`: detailed roll trades (BUY close old, SELL open new)
- `pnl_tracker.csv`: realized PnL entries from rollovers/re‑hedges (inverse COIN‑M formula)
- `rehedge_log.csv`: EOD delta hedges
- `daily_pnl_tracker.csv`: date, portfolio value, daily/cumulative PnL, spot MTM, futures unrealized/realized

How to run
```bash
python backtest_with_rollovers.py 2024-01-01 2024-09-30
```
I fetch synchronized historical data via the same connectors used in `task1.py` and run everything locally. Expect it to take a bit longer for wider date ranges.

Notes on PnL signs (inverse COIN‑M)
- Closing a short higher than entry is negative: `1/exit < 1/entry` → negative PnL.
- Closing a short lower than entry is positive: `1/exit > 1/entry` → positive PnL.
- Total return combines spot MTM, futures PnL (unrealized + realized), and the basis harvested by maintaining the hedge.

