## README — @task1_sell.py: Unwinding the Positions

What this is
- I unwind the cash‑and‑carry position over 24 hours: SELL spot BTC and BUY back COIN‑M inverse futures in paced slices.
- I follow the same discipline as entry: time‑of‑day weighting, PoV caps, passive‑first pricing, and a final must‑complete sweep.

Inputs
- Inventory: `{ "spot_btc": float, "spot_entry_px": float, "futures": { symbol: { qty: int, entry_px: float } } }`
- Weights: 96×15‑minute slot weights (dict: slot → weight)
- UnwindParams: `capital_usdt` (for PnL context)
- Price feed: `BacktestPriceFeed` (historical) or live connectors
- Flags: `is_backtest`, `end_date` (for days‑held calc)

How the 24‑hour unwind works:
1) I split the day into 96×15‑minute buckets and compute each slot’s weight `w`, plus the recent 15‑minute spot BTC volume.
2) At each timestamp `ts`:
   - I read spot mid/spread and a futures mid/spread (CURRENT by default, NEXT as fallback).
   - PoV: `pov = clip(pov_base × (1 + 0.8×(-sched_dev)), pov_min, pov_max)`, where `sched_dev = progress − time_fraction`.
   - Sizing: `child_btc = min(remaining_btc × w, pov × recent_15m_spot_volume_btc)`, then I convert to `child_ct` (≥ 1, ≤ remaining).
   - Pricing: if I’m behind schedule, I get more aggressive (sell spot closer to bid; buy futures closer to ask). Otherwise I stay passive.
   - I place paired orders (SELL spot, BUY futures), retry a canceled leg inside the bucket, and log the fills.
3) Final sweep: after the last slot, if anything is left, I force completion at marketable prices.

Combined TWAP + VWAP (how I blend them)
- TWAP backbone: I distribute the unwind across 96 equal time slots so the portfolio is fully closed within 24 hours.
- VWAP tilt: I tilt slices steering more size into liquid, favorable windows.
- PoV cap: I cap each child by a fraction of recent 15‑minute spot volume to respect live liquidity.

How my weights differ from task1.py:
- task1.py: I compute 96 weights from a synchronized historical leaderboard that blends annualized edge and liquidity + I also apply an opportunity factor vs historical basis to size up/down per slice.
- task1_sell.py: Here the goal is clean risk‑off and I do not apply any basis/opportunity scaling. Sizing depends only on (remaining × weight) and a PoV cap from recent 15‑minute volume.
- Rationale: entry optimizes to harvest basis; exit prioritizes completion with minimal footprint and timing risk. 


- Execution: this script is meant to be driven by the multi‑day backtest so it can use the exact open positions. I didn’t create this as a standalone use because we need an inventory to run it.

