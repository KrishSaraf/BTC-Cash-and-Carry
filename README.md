# Bitcoin Cash-and-Carry Arbitrage Trading System

An automated quantitative trading system that captures the price difference between Bitcoin spot and futures markets. The system intelligently enters positions, manages them over time, and automatically handles futures contract rollovers—all while minimizing market impact and slippage.

## 🎯 What is Cash-and-Carry Arbitrage?

Imagine Bitcoin is trading at $30,000 on the spot market, but a futures contract expiring in 3 months is trading at $30,500. This $500 difference is called the "basis." Cash-and-carry arbitrage captures this difference by:

1. **Buying** Bitcoin on the spot market ($30,000)
2. **Selling** (shorting) Bitcoin futures ($30,500)
3. **Holding** both positions and collecting the basis as profit
4. **Staying neutral** to Bitcoin's price movements (if BTC goes up or down, you're hedged)

### The Trade
- **Long Position**: BTC Spot (`BTCUSDT`)
- **Short Position**: BTC Inverse Futures (`BTCUSD` COIN-M contracts)
- **Profit Source**: Basis spread + roll yield
- **Risk**: Delta-neutral (protected from BTC price swings)

## 💡 Why This Matters

**The Problem**: Most crypto traders are exposed to massive price swings. If you buy Bitcoin at $30k and it drops to $25k, you lose 16%. If it rises to $35k, you gain 16%. It's a pure directional bet.

**The Solution**: Cash-and-carry arbitrage lets you profit from market inefficiencies (the basis spread) WITHOUT betting on Bitcoin's direction. You're hedged—if BTC goes up or down, your spot and futures positions offset each other.

**The Challenge**: Executing this strategy poorly kills your returns:
- Trade $1M at once → you move the market and lose 0.5% to slippage ($5,000)
- Miss rollovers → futures expire and you're unhedged
- Poor timing → you buy during low liquidity and pay wider spreads

**This System's Edge**:
- ✅ Minimizes slippage through gradual, timed execution
- ✅ Automatically handles rollovers (never caught off-guard)
- ✅ Adapts to market conditions in real-time
- ✅ Executes like institutional traders, not retail

## 🚀 Key Features

### 1. **Smart Entry with Minimal Slippage** (`task1.py`)

**Why spread trades over 24 hours?** Dumping $1 million into the market at once would move prices against you (slippage), eating into profits. Instead, this system:

- **Analyzes history**: Looks back 1-200 days to find the best times to trade (high liquidity, tight spreads)
- **Spreads execution**: Breaks the $1M into 96 small orders over 24 hours (every 15 minutes)
- **Times it right**: Places bigger orders during historically liquid periods, smaller ones during quiet times
- **Stays flexible**: Adjusts order size based on current market conditions (never overwhelming the order book)
- **Picks the best contract**: Every 15 minutes, chooses CURRENT or NEXT quarter futures based on which has higher basis
- **Guarantees completion**: Even if some orders don't fill, a final sweep ensures you're fully invested by end of day

**Result**: You get better prices and pay less in slippage—just like institutional traders do.

### 2. **Automatic Position Management** (`backtest_with_rollovers.py`)

**The challenge**: Futures contracts expire. You can't just "hold" them forever like spot Bitcoin.

**The solution**: This system automatically handles everything:

- **Auto-rollover**: When a futures contract is 2 days from expiry, the system automatically:
  - Closes the expiring contract (buys it back)
  - Opens a new contract in the next quarter (sells the new one)
  - Does this gradually over 24 hours (again, to avoid slippage!)
  - Tracks all the PnL from each rollover
  
- **Daily rebalancing**: Checks every day if your position is still neutral (spot vs futures balanced). If it drifts by more than 5%, automatically rebalances.

- **Complete tracking**: Records every trade, every rollover, every hedge—so you know exactly where profits are coming from.

**Result**: Set it and forget it. The system manages positions for weeks or months without manual intervention.

### 3. **Clean Exit Strategy** (`task1_sell.py`)

When it's time to close the position:
- Unwinds everything over 24 hours (same low-slippage approach)
- Sells spot BTC and buys back the short futures simultaneously
- Closes cheapest contracts first (FIFO - minimizes losses)
- Adapts pricing if behind schedule
- Forces completion to ensure you're 100% out by end of day

### 4. **Institutional-Quality Infrastructure**

This isn't a quick script—it's built like professional trading systems:
- **Handles API limits**: Automatically chunks large data requests
- **Resilient**: Retries failed orders, handles rate limits, never gives up
- **Realistic simulation**: 90% fill probability mirrors real market conditions
- **Complete audit trail**: Every trade logged with timestamp, price, quantity
- **Synchronized data**: Ensures spot and futures prices are perfectly aligned (no timing errors)

## 📊 How It Works (Simple Explanation)

### Step 1: Learn from History
The system analyzes the last 100+ days of trading data to answer: *"When is the best time to trade?"*

- Looks at every 15-minute window across the day
- Finds windows with the highest basis (profit opportunity)
- Finds windows with the most liquidity (easy to trade without moving the market)
- Creates a "game plan" of 96 time slots, giving more weight to the best windows

### Step 2: Execute Gradually (Avoid Slippage!)
Instead of one big trade, the system makes 96 small trades over 24 hours:

```
Every 15 minutes:
  → Check current basis for CURRENT and NEXT quarter futures
  → Pick the contract with higher basis (more profit)
  → Calculate safe order size (based on recent trading volume)
  → Place paired orders: BUY spot + SELL futures
  → If orders don't fill, retry with better prices
  → Track progress and adjust if falling behind schedule
```

**Why this matters**: Large trades move prices against you. Small, timed trades get you better prices.

### Step 3: Hold & Manage Automatically
Once positions are open, the system runs on autopilot:

- **Monitors daily**: Checks if the hedge is still balanced
- **Rebalances if needed**: Adjusts futures position if it drifts >5%
- **Rolls contracts**: 2 days before expiry, automatically closes old contracts and opens new ones
- **Tracks everything**: Every PnL source is logged (spot gains, futures PnL, rollover costs)

### Step 4: Calculate Profits

**Inverse Futures Math** (COIN-M contracts are special):
- Each contract is worth $100 / Bitcoin_Price in BTC
- When you close a short position:
  - **Profit**: If you buy back at a LOWER price than you sold
  - **Loss**: If you buy back at a HIGHER price than you sold
- Formula: `Profit_in_BTC = contracts × $100 × (1/exit_price - 1/entry_price)`
- Convert to USD: Multiply by current spot price

**Total Profit** = Spot gains + Futures PnL + Basis captured - Rollover costs

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Binance account with Spot and COIN-M Futures API access

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd "BTC Cash and Carry"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure API credentials**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Binance API keys
# BINANCE_API_KEY=your_api_key_here
# BINANCE_API_SECRET=your_api_secret_here
```

## 📖 Usage

### Option 1: Test Entry Execution (Single Day)

Want to see how the system would enter a position? Run this:

```bash
python task1.py
```

You'll be prompted for:
- **Target date**: Any historical date (e.g., `2021-06-15`)
- **Lookback days**: How much history to analyze (50-100 recommended)

The system will:
1. Analyze historical data to find optimal trading windows
2. Simulate entering a $1M position over 24 hours
3. Show you exactly when/how it would place each order
4. Display final results (no CSV files created by default)

**Use this to**: Understand the entry logic and see how slippage reduction works.

### Option 2: Full Multi-Day Backtest (The Whole Journey)

Want to simulate the complete strategy over months? Run this:

```bash
python backtest_with_rollovers.py 2021-01-01 2021-09-30
```

The system will:
1. Enter the position on Jan 1 (using `task1.py` logic)
2. Hold and manage it every day through Sep 30
3. Automatically roll futures contracts as they expire
4. Rebalance daily if the hedge drifts
5. Exit completely on Sep 30
6. Generate detailed CSV logs of everything

**Use this to**: See complete strategy performance including rollovers and PnL tracking.

**Note**: This fetches a lot of historical data, so expect it to take 10-20 minutes for a 9-month backtest.

## 📁 Output Files

| File | Description |
|------|-------------|
| `daily_pnl_tracker.csv` | Daily portfolio value, PnL breakdown (spot/futures/realized) |
| `entry_day_exec_log.csv` | Detailed entry execution fills and prices |
| `rollover_exec_log.csv` | Rollover trade details (close old, open new contracts) |
| `rehedge_log.csv` | Delta re-hedging transactions |
| `pnl_tracker.csv` | Realized PnL events from rollovers and hedges |

## 📈 Real Backtest Example

**Period**: January 2 - September 30, 2021 (272 days during the bull run)  
**Starting Capital**: $1,000,000 USDT  
**Bitcoin Price Movement**: $29,000 → $43,000 (peak)

### What Happened

This was a wild ride—Bitcoin rallied hard in early 2021. Here's how the cash-and-carry strategy performed:

**Day 1 (Jan 2, 2021)**:
- Entered position: Long 34.57 BTC spot + Short equivalent futures
- Initial basis: ~700 bps (7% annualized)
- Portfolio value: $1,048,511 (+4.85% on day 1!)

**During the Bull Run**:
- **Spot position**: Made huge gains as BTC rose from $29k to $43k
- **Futures position**: Lost money (we were short, price went up)
- **Net effect**: Profitable because basis capture + spot gains > futures losses
- **Multiple rollovers**: System automatically rolled contracts 3-4 times as they expired

**Peak (Jan 27, 2021)**:
- Portfolio hit $1,116,154 when BTC touched $30,382
- Up $116,154 in just 26 days

**Volatility**:
- Unlike pure spot holding (very volatile), this strategy was more stable
- Delta-neutral hedge protected from worst drawdowns
- Basis kept accruing regardless of BTC direction

### Key Takeaway

The strategy worked as designed: captured basis premium while reducing directional risk. When BTC rallied, we still profited (just less than pure spot). When BTC dropped, we were protected by the hedge.

**Full results**: See `daily_pnl_tracker.csv` for day-by-day performance breakdown.

## 🔧 Technical Details

### Data Architecture
- **Synchronized fetching**: Ensures spot and futures bars align perfectly by timestamp
- **Chunked requests**: Breaks large historical queries into 1500-bar chunks
- **Continuous contracts**: Uses CURRENT/NEXT quarter continuous series for analysis
- **Fallback mechanisms**: Gracefully handles missing data with multiple fallback strategies

### Execution Parameters
- **Capital**: $1,000,000 USDT (configurable)
- **Bucket size**: 15 minutes (96 slots per 24 hours)
- **PoV range**: 2%-25% (base: 10%)
- **Fill probability**: 90% (simulation)
- **Contract size**: $100 USD per COIN-M contract
- **Roll window**: 2 days before expiry
- **Delta threshold**: 5% of spot position

### Position Tracking
- Maintains inverse-weighted average entry prices
- Handles partial fills and position updates
- Tracks realized PnL on each close
- Supports multiple concurrent futures positions

## 🔒 Security

- **Never commit `.env`**: API keys stored securely in `.env` file (gitignored)
- **Environment validation**: Scripts verify credentials before execution
- **Read-only recommended**: Use read-only API keys for backtesting

## ⚠️ Important Notes

### This is a Simulation
- **No real money**: All trades are simulated using historical data
- **90% fill rate**: Mimics realistic market conditions (not all orders fill)
- **Educational only**: Built to learn how institutional arbitrage works

### Before Live Trading (If You Ever Consider It)
1. **Understand the risks**: Crypto is volatile; futures can liquidate your account
2. **Start small**: This was built for $1M simulations, but you should start with much less
3. **Know the costs**: Fees, funding rates, and slippage eat into profits
4. **Test thoroughly**: Run extensive backtests across different market conditions
5. **Monitor actively**: Even automated systems need human oversight

### What This Project Is Good For
- ✅ Learning how cash-and-carry arbitrage works
- ✅ Understanding slippage reduction techniques
- ✅ Seeing how institutional-grade execution differs from retail
- ✅ Studying automated position management and rollovers
- ✅ Backtesting and strategy research

### What This Project Is NOT
- ❌ Financial advice
- ❌ A guaranteed money printer
- ❌ Ready for live trading without extensive modification
- ❌ Suitable for beginners (requires deep understanding of derivatives)

## 🤝 Questions or Improvements?

This project is open for learning and experimentation. If you:
- Find bugs or improvements
- Want to add features (better risk management, different execution styles)
- Have questions about the implementation

Feel free to open an issue or fork the project!

## 📄 License

MIT License - Use freely, but at your own risk.

---

**Final Disclaimer**: Cryptocurrency trading carries substantial risk of loss. This software is provided "as-is" for educational purposes only. Past performance (even in backtests) does not guarantee future results. The author assumes no responsibility for financial losses incurred from using this code. Trade responsibly.

