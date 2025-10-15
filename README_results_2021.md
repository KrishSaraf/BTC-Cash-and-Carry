## Backtest Results — 2021-01-01 to 2021-09-30

### Headline
- Start → End: 2021-01-02 → 2021-09-30 (272 days)
Please check @daily_pnl_tracker.csv 

### Notes
- In a bull run, the short COIN‑M futures leg typically realizes negative PnL on closes; spot gains dominate. The basis harvested offsets a portion of futures losses.
- Rollover PnL uses the inverse formula: 100 × qty × (1/exit − 1/entry) in BTC (USD‑converted using spot at the fill). Final day unwind uses the same per‑fill logic.

### How the backtest ran (step-by-step)

1. Initialize portfolio and parameters
   - Start: 2021-01-01; End: 2021-09-30; Initial NAV: $1,000,000.
   - Instruments: Spot `BTCUSDT` and inverse COIN‑M futures `BTCUSD` (current and next quarter).
   - Target: Remain long spot and short futures, harvesting the basis while staying near delta‑neutral.

2. For each trading day
   - Download minute bars
     - Fetch 1‑minute candles for `BTCUSDT`, `BTCUSD (CURRENT_QUARTER)`, and `BTCUSD (NEXT_QUARTER)`.
     - Uses chunking when needed (e.g., “Batch 1: 1000 bars”, “Batch 2: 440 bars”). Futures often report 1441 bars due to inclusive boundaries.
   - Compute basis and exposures
     - Compute intraday basis in bps and track spot/futures exposures and net delta.
   - Execute rollovers near expiry
     - Gradually close the current quarter and open the next quarter in small slices every 15 minutes.
     - Track progress (e.g., “Rollover progress: 45.8%”) and realized PnL per slice using the inverse formula.
   - Re‑hedge when delta drifts
     - If net delta deviates, buy/sell the futures to re‑align. Actions are logged (e.g., “REHEDGE LOG: SELL 994 contracts...”).
   - End‑of‑day mark‑to‑market (MTM)
     - Value the portfolio and record daily PnL (e.g., “MTM LOG for 2021‑09‑24: ... Daily PnL=$‑70,482.96”).

3. Final day unwind (2021‑09‑30)
   - Gradually flatten both legs throughout the day, targeting basis ≈ 0 bps.
   - If inventory remains near the close, force completion so the portfolio finishes flat.
   - Record per‑bucket progress and the final PnL breakdown.

### What each part is doing

- Spot leg (`BTCUSDT`)
  - Holds the long BTC exposure. In a bull run this drives most of the positive MTM PnL.

- Futures leg (`BTCUSD` COIN‑M inverse)
  - Provides the short hedge and carries the basis. Realized PnL on closes (rolls/hedges) can be negative during uptrends.

- Rollover engine
  - When the current quarter approaches expiry, closes it and opens the next quarter in small slices, logging progress and per‑fill PnL.

- Re‑hedge engine
  - Monitors net delta and issues BUY/SELL futures to re‑center exposure. Each re‑hedge is logged with contracts, direction, and prices.

- MTM engine
  - Computes end‑of‑day portfolio value and daily PnL and writes the results to `daily_pnl_tracker.csv`.

- Data fetcher
  - Retrieves minute‑level history in batches, ensuring continuity (e.g., 1000 + 440 bars for a full UTC day).

### Key outputs (CSV logs)

- `daily_pnl_tracker.csv`: Daily portfolio value and PnL summary across the backtest window.
- `pnl_tracker.csv`: Realized PnL entries from rollovers and re‑hedges.
- `rollover_exec_log.csv`: Per‑slice rollover fills, quantities, prices, and realized PnL.
- `rehedge_log.csv`: Re‑hedge actions with direction, size, and execution prices.
- `unwind_exec_log.csv`: Final‑day unwind progress by bucket, with basis and quantities (currently disabled in code).
- `entry_day_exec_log.csv`: Entry allocations/fills on the first trading day.


### Reproduce these results (2021‑01‑01 → 2021‑09‑30)

1. Install dependencies
   - `pip install -r requirements.txt`
2. Run the multi‑day backtest
   - `python backtest_with_rollovers.py 2021-01-02 2021-09-30`
3. Inspect outputs
   - Daily summary in `daily_pnl_tracker.csv`; realized events in `pnl_tracker.csv`; entry/rollover/re‑hedge details in their respective logs.


### Implementation details and assumptions

- Basis and selection
  - Basis is computed as (F − S) / S × 10,000 bps; positive basis favors shorting futures vs long spot.
  - For each 15‑minute slice, CURRENT vs NEXT quarter is chosen by higher positive basis.

- Entry/execution engine (Day 1)
  - 24‑hour schedule in 96 × 15‑minute buckets using weights derived from historical analysis.
  - Sizing combines time‑weights with a Participation‑of‑Volume cap (base PoV ~10%, bounded 2%–25%).
  - Price aggression increases if behind schedule or if basis compresses; a final sweep forces completion.
  - Orders are simulated (no live trading). A simple simulator fills ~90% prob. Final sweep assumes fills.

- Rollover engine
  - Triggers when a contract is ≤2 days from expiry. Rolls from expiring symbol to the next quarter.
  - Uses a 21‑day futures volume profile to weight the day; PoV‑capped per slice.
  - Realized PnL on buy‑to‑close fills is recorded using the inverse formula and converted to USD at spot.

- Re‑hedge engine
  - Checks end‑of‑day net delta; if |delta| > 5% of spot position, executes a hedge using EOD prices.
  - Trades are logged to `rehedge_log.csv`, and realized PnL is appended to `pnl_tracker.csv`.

- MTM calculation
  - End‑of‑day portfolio value = initial capital + spot MTM + futures unrealized + realized (rolls, hedges).
  - Futures unrealized PnL uses the inverse convention (short: qty×100×(1/mark − 1/entry)).

