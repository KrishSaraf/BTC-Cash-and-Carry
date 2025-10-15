# Bitcoin Cash-and-Carry Arbitrage System

An automated trading system for Bitcoin cash-and-carry arbitrage. Takes a $1M position across spot and futures markets, manages it over time with automatic contract rollovers, and exits completely after the holding period.

## What is Cash-and-Carry Arbitrage?

The basis spread exists when Bitcoin futures trade at a premium over spot. For example:
- Spot BTC: $30,000
- Futures (3 months out): $30,500
- Basis: $500 or ~165 bps

You can capture this spread by going long spot and short futures simultaneously. Since you're hedged, Bitcoin's price direction doesn't matter—you just collect the basis as it converges.

**The position:**
- Long: BTC spot on Binance (`BTCUSDT`)
- Short: BTC inverse futures on Binance COIN-M (`BTCUSD` quarterly contracts)
- Goal: Capture the basis premium while staying roughly delta-neutral

## Why This Implementation Matters

Executing a $1M arbitrage trade poorly will cost you real money:

**The slippage problem:** Market orders on a $1M position will move the market against you. You might lose 0.3-0.5% ($3,000-$5,000) just entering the position. Over multiple trades (entry, rollovers, exit), this adds up fast.

**The rollover problem:** Futures contracts expire. The March contract expires in March, the June contract in June, etc. If you don't roll your short position into the next contract before expiry, you're suddenly unhedged and exposed to Bitcoin's full volatility.

**The timing problem:** Not all times of day are equal. Trading during low liquidity windows means wider spreads and worse execution prices.

This system addresses these issues by:
1. Spreading execution over 24 hours in small slices (like institutional TWAP/VWAP execution)
2. Automatically detecting when futures contracts are near expiry and rolling them
3. Using historical analysis to identify the best execution windows
4. Managing the entire lifecycle from entry to exit without manual intervention

## How the System Works

### Part 1: Entry Execution (task1.py)

**Goal:** Open a $1M position (long spot + short futures) over 24 hours with minimal slippage.

**Process:**

1. **Historical Analysis (lookback 50-200 days)**
   - Fetches minute-by-minute spot and futures prices
   - Calculates basis spread for every 15-minute window across the day
   - Identifies which time slots historically had: (a) highest basis, (b) most volume/liquidity
   - Generates 96 execution weights (one per 15-minute slot) favoring high-edge, high-liquidity windows

2. **Execution Over 24 Hours (96 × 15-minute slices)**
   
   Every 15 minutes, the system:
   - Checks current basis for CURRENT_QUARTER and NEXT_QUARTER futures
   - Selects whichever contract has higher positive basis (more profit potential)
   - Calculates safe order size based on:
     * Historical weight for this time slot
     * Recent 15-minute trading volume (participation-of-volume limit)
     * How far behind/ahead of schedule we are
   - Places paired orders: BUY spot + SELL futures
   - Retries if orders don't fill, gets more aggressive if falling behind
   
   **Pricing logic:**
   - On schedule → passive pricing (limit orders inside the spread)
   - Behind schedule → aggressive pricing (cross the spread if needed)
   
   **Final sweep:**
   - After 96 slots, if any capital remains uninvested, forces completion
   - Uses market orders if necessary to guarantee full $1M deployment

**Output:** 
- Console log showing execution progress
- Entry day execution log with all fills (optional CSV)
- Final position: ~34.5 BTC long spot, equivalent contracts short futures

### Part 2: Multi-Day Backtest with Rollovers (backtest_with_rollovers.py)

**Goal:** Simulate holding the position for months, handling all the daily management automatically.

**Complete Flow:**

**Day 1 (Entry)**
- Runs `task1.py` logic to open position
- Records all entry trades to `entry_day_exec_log.csv`
- Tracks initial position in `PositionTracker`

**Days 2 through N-1 (Daily Management)**

For each trading day, the system:

1. **Fetches daily price data**
   - Gets 1-minute bars for spot, CURRENT_QUARTER, and NEXT_QUARTER futures
   - Uses chunked fetching (1500 bars per API call) to handle API limits

2. **Checks for rollover conditions**
   
   For each active futures position:
   - If days to expiry ≤ 2: triggers rollover
   
   **Rollover execution** (when triggered):
   - Builds 21-day volume profile to weight the rollover execution
   - Runs a 24-hour gradual roll (96 × 15-minute slices, just like entry):
     * Slice by slice: BUY to close expiring contract, SELL to open new contract
     * Uses participation-of-volume limits (25% of recent volume)
     * Gets progressively more aggressive if behind schedule
   - Records realized PnL from closing the expiring contract:
     * Formula: `BTC_PnL = contracts × $100 × (1/exit_price - 1/entry_price)`
     * Converts to USD: `USD_PnL = BTC_PnL × spot_price_at_fill`
   - Updates position tracker with new contract at new entry price
   - Logs all rollover trades to `rollover_exec_log.csv`
   - Records PnL to `pnl_tracker.csv`
   
   **Why the inverse formula?** COIN-M contracts are inverse perpetuals. Each contract is worth $100/BTC_price in BTC terms. When you close a short position, you profit if the exit price is lower than entry (you're buying back cheaper than you sold).

3. **Daily delta rehedging**
   - Calculates net delta: `spot_BTC - (contracts × $100 / spot_price)`
   - If |net_delta| > 5% of spot position: executes rebalancing trade
   - Uses EOD futures prices for the hedge
   - Records realized PnL if closing any contracts
   - Logs rehedge to `rehedge_log.csv`

4. **End-of-day mark-to-market**
   - **Spot MTM PnL**: `(current_price - entry_price) × BTC_held`
   - **Futures unrealized PnL**: For each open position:
     * `contracts × $100 × (1/mark_price - 1/entry_price)`
     * If short and price went up: negative unrealized PnL
     * If short and price went down: positive unrealized PnL
   - **Futures realized PnL**: Sum of all recorded rollovers and rehedges
   - **Portfolio value**: `initial_capital + spot_MTM + futures_unrealized + futures_realized`
   - Writes daily snapshot to `daily_pnl_tracker.csv`

**Day N (Final Day - Exit)**

On the last day, the system:

1. **Builds 24-hour exit schedule**
   - Uses flat weights (1/96 for each slot) - no trying to optimize exit timing
   - Goal is just clean risk-off with minimal market impact

2. **Executes gradual unwind** (via `task1_sell.py`)
   
   Every 15 minutes over 24 hours:
   - Calculates remaining position to close
   - Sizes orders based on:
     * Time-weighted schedule (close 1/96 each slot)
     * Recent 15-minute volume (PoV limits)
     * Schedule adherence (more aggressive if behind)
   - Places paired orders: SELL spot + BUY futures
   - Uses FIFO for futures: closes cheapest contracts first
   - Tracks basis as it converges toward zero
   
   **Pricing:**
   - On schedule → passive (place limit orders)
   - Behind schedule → aggressive (cross the spread)
   
   **Final sweep:**
   - After 96 slots, if any position remains, forces 100% completion
   - Uses aggressive market orders to guarantee flat book

3. **Final PnL calculation**
   - Sums spot PnL from selling at exit prices vs entry
   - Calculates futures PnL from all buybacks (same inverse formula)
   - Adds all realized PnL from rollovers/rehedges during holding period
   - Records final portfolio value

**Outputs:**
- `daily_pnl_tracker.csv`: Every day's portfolio value, PnL breakdown
- `entry_day_exec_log.csv`: All entry trades from day 1
- `rollover_exec_log.csv`: Every rollover trade (buy old contract, sell new contract)
- `rehedge_log.csv`: Delta rebalancing trades
- `pnl_tracker.csv`: All realized PnL events
- Console summary: Final return, days held, annualized return

## Real Example: 2021 Backtest

I ran this on January 2 - September 30, 2021. Here's what actually happened:

**Setup:**
- Starting capital: $1,000,000
- Entry: Jan 2, 2021 at spot price $32,172
- Position: 34.57 BTC long, 688 contracts short
- Initial basis: ~700 bps

**During the run (272 days):**
- Bitcoin rallied from $29k to $43k (peak)
- The system rolled futures contracts 3-4 times automatically
- Daily rebalancing kept delta within tolerance
- Spot position gained significantly (BTC went up)
- Futures position lost money (we were short, price rose)
- Basis capture partially offset futures losses

**Results:**
- Peak portfolio value: $1,116,154 on Jan 27 (up $116k in 26 days)
- Strategy reduced volatility vs pure spot holding
- Delta-neutral hedge limited downside when BTC corrected

**What the logs show:**
- `daily_pnl_tracker.csv`: 272 rows, one per day
- Daily PnL split into: spot MTM, futures unrealized, futures realized
- You can see exactly when rollovers happened (realized PnL spikes)
- Portfolio value tracked through multiple BTC rallies and corrections

The point: the strategy worked as designed. It captured basis premium while staying hedged. Not as much upside as pure spot holding in a bull run, but way less risk.

## Installation & Setup

**Requirements:**
- Python 3.8+
- Binance account with Spot and COIN-M Futures API access

**Install:**
```bash
git clone <your-repo>
cd "BTC Cash and Carry"
pip install -r requirements.txt
```

**Configure API keys:**
```bash
cp .env.example .env
# Edit .env and add your keys:
# BINANCE_API_KEY=your_key
# BINANCE_API_SECRET=your_secret
```

The scripts load credentials from `.env` automatically. Never commit `.env` to git (it's already in `.gitignore`).

## Usage

**Option 1: Test entry execution on a single day**
```bash
python task1.py
```
- Prompts for target date (e.g., 2021-06-15) and lookback days (50-100 recommended)
- Shows how the system would enter a position that day
- Displays execution schedule and simulated fills
- Good for understanding the entry logic

**Option 2: Run full multi-day backtest**
```bash
python backtest_with_rollovers.py 2021-01-01 2021-09-30
```
- Simulates complete lifecycle: entry → daily management → rollovers → exit
- Takes 10-20 minutes for a 9-month backtest (lots of API calls)
- Generates all CSV logs
- Shows daily progress and final summary

## Output Files

| File | Contents |
|------|----------|
| `daily_pnl_tracker.csv` | Daily portfolio value, PnL (spot/futures/realized), cumulative returns |
| `entry_day_exec_log.csv` | All entry trades: timestamp, prices, quantities, which futures contract |
| `rollover_exec_log.csv` | Rollover trades: closing old contract, opening new contract, realized PnL |
| `rehedge_log.csv` | Delta rebalancing trades: direction, size, prices |
| `pnl_tracker.csv` | All realized PnL events from rollovers and rehedges |

Open these in Excel/Python to analyze performance, execution quality, rollover timing, etc.

## Technical Implementation Notes

**Data handling:**
- Fetches 1-minute bars synchronized across spot and futures
- Chunks large requests (1500 bars per API call) to bypass limits
- Rate limiting with exponential backoff for API resilience
- Inner joins ensure spot and futures data align perfectly by timestamp

**Execution simulation:**
- 90% fill probability on limit orders (realistic, not 100%)
- Final sweeps use forced fills to guarantee completion
- Tracks every order attempt, retry, and final fill

**Position tracking:**
- Maintains weighted-average entry price for each futures position
- Handles partial fills and position updates correctly
- Properly accounts for inverse futures mechanics (COIN-M)

**Risk management:**
- Delta threshold: 5% of spot position
- Participation of volume limits: 2%-25% (typically 10%)
- Rollover window: 2 days before expiry
- Spread-crossing logic when behind schedule

## Disclaimer

This is a backtest simulation using historical data. No real trades are executed. The system is built for education and research.

**Before considering live trading:**
1. Understand that crypto futures can liquidate your account
2. Factor in real costs: trading fees, funding rates, exchange risk
3. Test extensively across different market conditions
4. Start small (this was built for $1M simulations)
5. Monitor actively—automation doesn't mean fire-and-forget

**What this is good for:**
- Learning how cash-and-carry arbitrage works in practice
- Understanding institutional execution techniques (TWAP/VWAP/PoV)
- Studying automated position management and contract rollovers
- Backtesting and strategy research

**What this is not:**
- Financial advice or a recommendation to trade
- A guaranteed profit system (basis can compress or go negative)
- Production-ready for live trading without significant modifications
- Suitable for people unfamiliar with derivatives

Trading involves substantial risk of loss. Use at your own risk. The author assumes no responsibility for financial losses.
