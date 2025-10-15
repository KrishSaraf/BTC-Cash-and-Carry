## README — @task1.py: Opening the Positions

- I run a synchronized historical analysis over the last (1-200) days to find the best 15‑minute windows to trade.
- I execute over 24 hours using a Participation of Volume (PoV) schedule with opportunistic sizing vs. the historical basis.
- At each slice I choose CURRENT or NEXT quarter (COIN‑M inverse) by whichever has the higher positive basis(but its obvious that the NEXT QUARTER will have higher basis)

How the synchronized analysis works:
1) I align time to midnight UTC, fetch the same 1‑minute window for spot `BTCUSDT` and futures `BTCUSD` (continuous CURRENT and NEXT).  
2) I inner‑join on minute timestamps so spot and futures are one‑to‑one.  
3) For each minute, I compute basis bps: (F − S) / S × 10,000.  
4) I estimate days to the quarter expiry to compute annualized return: ((F − S) / S) × (365 / days_to_expiry).  
5) I convert COIN‑M volume (contracts) into BTC: contracts × $100 / F, and aggregate by 15‑minute slot (96 slots per day).  
6) For each slot I keep mean/p75 annualized return, mean/p75 basis, and total BTC volume; I derive 96 weights using inverse rank blended with robust z‑scores of return and volume (missing slots fall back to uniform).

The historical analysis aims to identify optimal 15-minute trading windows by examining the last 100 days of market data. This helps us discover time slots that consistently offered the highest basis spreads and deepest liquidity, allowing us to target execution during periods with the greatest historical edge and market depth.

Once the historical analysis is done and the then we do the the 24‑hour execution:
1) Target sizing: from 1,000,000 USDT (minus a small fee buffer), I compute target BTC and target contracts (COIN‑M is $100/contract).  
2) Every 15 minutes (96 slices):  
   - I get spot and both futures mids/spreads for the current timestamp, compute basis for CURRENT and NEXT, and pick the higher positive basis to short.  
   - PoV sizing: `pov = pov_base × (1 + 0.8×(-sched_dev)) × (1 + 0.005×|basis_bps|)` clipped to [pov_min, pov_max].  
   - Weight sizing: `weight_child = target_btc × slot_weight × opportunity_score`, where `opportunity_score = clip(cur_bps / historical_avg_basis, 0.5, 2.0)`.  
   - Child size (spot) is `min(weight_child, pov × recent_15m_spot_volume_btc)`.  
   - Futures child contracts are allocated by weight toward the precomputed target contracts (≥ 1), capped by remaining.  
   - Pricing: passive by default but if I’m behind schedule or the basis compresses vs. arrival I cross more aggressively. I place paired orders (buy spot, sell futures) and retry any canceled leg inside the slice.  
3) Final sweep: after the last slice if anything remains, I force completion so the full 1,000,000 USDT is invested by 24h.

Combined TWAP + VWAP scheduling :
- TWAP backbone: I spread the target over 96 equal 15‑minute slots so I always finish in 24 hours (the schedule provides time discipline and reduces slippage risk).
- VWAP tilt: I tilt each slot by historical volume/edge via the 96 weights (derived from the leaderboard), so heavier weights land in time windows that historically had more liquidity and better basis.
- PoV cap: I cap each child by a % of recent 15‑minute spot volume (dynamic liquidity control) to avoid over‑impacting the book.
- Net effect: TWAP guarantees completion and smooth pacing; the VWAP tilt and PoV cap adapt size to where liquidity and expected edge are, slice by slice.

Live vs. historical (what changes):
- Historical day: I pre-fetch that day’s 24h price feed and run fast. (Optional CSV export is disabled by default.)  
- Live day: I re-normalize weights to the remaining slots in the day, start at the current bucket, and print a short progress line per slice.  
The analysis/weights logic is identical in both.

task1.py exclusively focuses on the 24‑hour entry execution. 

How to run :
```bash
pip install -r requirements.txt
python task1.py
```
Then enter:  
- Date for analysis (YYYY‑MM‑DD)  
- Lookback days (1–200): recommended 50+ for robust stats 

Outputs: console summary + full execution log printed. No CSV is written by default.

