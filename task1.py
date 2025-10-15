import time, datetime as dt
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

from binance.spot import Spot
from binance.cm_futures import CMFutures

# Load environment variables from .env file
load_dotenv()

class SynchronizedCashCarryScanner:
    """
    Run a synchronized cash-and-carry scan with precise timing alignment.
    - Use midnight UTC boundaries for start/end timestamps
    - Fetch spot and futures data over identical time windows
    - Ensure spot and futures bars are aligned one-to-one
    - Cash-and-carry context: long spot, short futures to capture the basis
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        lookback_days: int = 200,
        spot_symbol: str = "BTCUSDT",
        cm_pair: str = "BTCUSD",
        interval: str = "1m",
        contract_usd: float = 100.0,
        kl_limit: int = 1500,
        sleep_s: float = 0.3,
        basis_bps_min_buy: float = 0.0,
        basis_bps_max_sell: float = -0.0
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.lookback_days = lookback_days
        self.spot_symbol = spot_symbol
        self.cm_pair = cm_pair
        self.interval = interval
        self.contract_usd = float(contract_usd)
        self.kl_limit = kl_limit
        self.sleep_s = sleep_s
        self.basis_bps_min_buy = float(basis_bps_min_buy)
        self.basis_bps_max_sell = float(basis_bps_max_sell)

        self.spot = Spot(api_key=api_key, api_secret=api_secret)
        self.cm = CMFutures(key=api_key, secret=api_secret)

        self.last_buy_leaderboard: Optional[pd.DataFrame] = None
        self.last_sell_leaderboard: Optional[pd.DataFrame] = None

    @staticmethod
    def _get_precise_time_boundaries(end_date_str: str, lookback_days: int) -> Tuple[int, int]:
        """
        Get precise start and end timestamps aligned to midnight UTC boundaries.
        
        Args:
            end_date_str: Date in "YYYY-MM-DD" format
            lookback_days: Number of days to look back
            
        Returns:
            Tuple of (start_ms, end_ms) both aligned to midnight UTC
        """
        # Parse end date and set to midnight UTC
        end_date = dt.datetime.strptime(end_date_str, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=dt.timezone.utc
        )
        
        # Calculate start date (lookback_days before end_date)
        start_date = end_date - dt.timedelta(days=lookback_days)
        
        # Convert to milliseconds
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        # Ensure we don't go beyond current time
        now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
        end_ms = min(end_ms, now_ms)
        
        #returns start_ms and end_ms which is the start and end time of the klines
        return start_ms, end_ms

    @staticmethod
    def _ms_to_dt(ms: int) -> dt.datetime:
        return dt.datetime.fromtimestamp(ms/1000, dt.timezone.utc)

    def _fetch_klines_synchronized(self, client, symbol: str, start_ms: int, end_ms: int, is_continuous: bool = False, contract_type: str = None) -> list:
        """
        Fetch klines with synchronized timing and robust error handling.
        - Automatically chunks requests to respect API limits (e.g., 1500 bars)
        - Applies exponential backoff for rate limits and transient failures
        - Returns all bars in chronological order for clean downstream alignment
        
        Args:
            client: Binance client (spot or futures)
            symbol: Symbol to fetch
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            is_continuous: Whether this is a continuous futures contract
            contract_type: Contract type for continuous futures
        """
        display_symbol = f"{symbol} ({contract_type})" if is_continuous and contract_type else symbol
        print(f"   Fetching {display_symbol} from {self._ms_to_dt(start_ms)} to {self._ms_to_dt(end_ms)}")
        
        all_klines = []
        current_start = start_ms
        batch_count = 0
        backoff = 1
        
        while current_start < end_ms:
            batch_count += 1
            
            # Calculate chunk end (don't exceed end_ms)
            chunk_end = min(current_start + (self.kl_limit * 60 * 1000), end_ms)
            
            try:
                print(f"      Batch {batch_count}: {self._ms_to_dt(current_start).strftime('%m/%d %H:%M')} to {self._ms_to_dt(chunk_end).strftime('%m/%d %H:%M')}")
                
                if is_continuous:
                    batch = client.continuous_klines(
                        pair=self.cm_pair,
                        contractType=contract_type,
                        interval=self.interval,
                        startTime=current_start,
                        endTime=chunk_end,
                        limit=self.kl_limit
                    )
                else:
                    batch = client.klines(
                        symbol=symbol,
                        interval=self.interval,
                        startTime=current_start,
                        endTime=chunk_end,
                        limit=self.kl_limit
                    )
                
                if not batch:
                    print(f"      No data returned for batch {batch_count}")
                    break
                
                all_klines.extend(batch)
                print(f"      Got {len(batch)} bars. Total: {len(all_klines)}")
                
                # Move to next chunk (start from last bar's close time + 1 minute)
                last_close_time = int(batch[-1][6])  # close_time
                current_start = last_close_time + 60000  # Add 1 minute
                
                # Reset backoff on success
                backoff = 1
                time.sleep(self.sleep_s)
                
            except Exception as e:
                error_msg = str(e)
                print(f"      Error in batch {batch_count}: {e}")
                
                if "429" in error_msg:  # Rate limit
                    print(f"      Rate limit hit, backing off {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                elif "418" in error_msg:  # IP ban
                    print(f"      IP banned, waiting 5 minutes...")
                    time.sleep(300)
                else:
                    time.sleep(5)
                    backoff += 2
                
                if backoff > 60:
                    print(f"      Too many failures for {symbol}, stopping")
                    break
        
        print(f"   Completed {display_symbol}: {len(all_klines)} total bars in {batch_count} batches")
        return all_klines

    @staticmethod
    def _klines_to_df(kl: list, symbol_prefix: str = "") -> pd.DataFrame:
        """Convert klines to a cleaned DataFrame with optional column prefix.

        - Parses timestamps and numeric fields
        - Drops invalid rows and de-duplicates on open_time
        - Optionally prefixes key columns (close, volume) for merging
        - Returns data sorted by open_time
        """
        if not kl:
            return pd.DataFrame()
        
        cols = ["open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","num_trades","taker_base","taker_quote","ignore"]
        df = pd.DataFrame(kl, columns=cols)
        
        # Convert timestamps
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        
        # Convert numeric columns
        for c in ["open","high","low","close","volume","quote_asset_volume","num_trades"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # Clean and deduplicate
        df = df.dropna(subset=["open_time","close","volume"]).drop_duplicates(subset=["open_time"])
        
        # Add prefix to columns if specified
        if symbol_prefix:
            rename_map = {
                "close": f"{symbol_prefix}_close",
                "volume": f"{symbol_prefix}_volume"
            }
            df = df.rename(columns=rename_map)
        
        #save the dataframe to a csv file
        # print(f"Saving {symbol_prefix}_klines.csv for {symbol_prefix}")
        # df.sort_values("open_time").reset_index(drop=True).to_csv(f"{symbol_prefix}_klines.csv", index=False)
        
        return df.sort_values("open_time").reset_index(drop=True)

    def _compute_basis_metrics(self, merged_df: pd.DataFrame, futures_col_prefix: str) -> pd.DataFrame:
        """Compute basis (bps), annualized return, and volume metrics.

        Returns a DataFrame including: open_time, minute_of_day, spot_close,
        futures_close, basis_bps, ann_return, total_vol_btc, days_to_expiry.
        """
        
        S = pd.to_numeric(merged_df["spot_close"], errors="coerce")
        F = pd.to_numeric(merged_df[f"{futures_col_prefix}_close"], errors="coerce")
        
        # Calculate basis in basis points
        basis_bps = (F - S) / S * 1e4
        
        # Estimate time to expiry (simplified quarterly logic)
        ts = pd.to_datetime(merged_df["open_time"], utc=True)
        
        def estimate_quarter_expiry(t: pd.Timestamp) -> float:
            """Estimate days to next quarter expiry."""
            # Simplified: assume quarterly contracts expire at end of quarters
            # March, June, September, December
            quarter_months = [3, 6, 9, 12]
            current_month = t.month
            current_year = t.year
            
            # Find next quarter month
            next_quarter_month = None
            for qm in quarter_months:
                if qm > current_month:
                    next_quarter_month = qm
                    break
            
            if next_quarter_month is None:
                next_quarter_month = 3
                current_year += 1
            
            # Last Friday of quarter month at 08:00 UTC (Binance convention)
            quarter_end = dt.datetime(current_year, next_quarter_month, 1, tzinfo=dt.timezone.utc)
            if next_quarter_month == 12:
                quarter_end = quarter_end.replace(day=31)
            else:
                quarter_end = (quarter_end.replace(month=next_quarter_month+1, day=1) - dt.timedelta(days=1))
            
            # Find last Friday
            while quarter_end.weekday() != 4:  # Friday = 4
                quarter_end -= dt.timedelta(days=1)
            
            quarter_end = quarter_end.replace(hour=8, minute=0, second=0)
            
            # Calculate days difference
            days_diff = (quarter_end - t.to_pydatetime()).total_seconds() / 86400
            return max(days_diff, 0.1)  # Minimum 0.1 days
        
        if "cur" in futures_col_prefix:
            dte_days = ts.apply(estimate_quarter_expiry)
        else:  # next quarter
            dte_days = ts.apply(lambda t: estimate_quarter_expiry(t) + 90)  # Roughly +3 months
        
        # Annualized return
        ann_return = ((F - S) / S) * (365.0 / dte_days)
        
        # Volume calculations (convert futures contracts to BTC equivalent)
        spot_vol_btc = pd.to_numeric(merged_df["spot_volume"], errors="coerce").fillna(0.0)
        fut_contracts = pd.to_numeric(merged_df[f"{futures_col_prefix}_volume"], errors="coerce").fillna(0.0)
        fut_vol_btc = (fut_contracts * self.contract_usd / F).fillna(0.0)
        total_vol_btc = spot_vol_btc + fut_vol_btc
        
        result_df = merged_df[["open_time"]].copy()
        result_df["minute_of_day"] = (ts.dt.hour * 60 + ts.dt.minute).astype(int)
        result_df["spot_close"] = S
        result_df[f"{futures_col_prefix}_close"] = F
        result_df["basis_bps"] = basis_bps
        result_df["ann_return"] = ann_return
        result_df["total_vol_btc"] = total_vol_btc
        result_df["days_to_expiry"] = dte_days
        
        return result_df.dropna(subset=["basis_bps", "ann_return", "total_vol_btc"])

    @staticmethod
    def _create_leaderboard(df: pd.DataFrame, resample_15m: bool = True) -> pd.DataFrame:
        """Create a leaderboard from basis metrics.

        Optionally resamples to 15-minute slots (96 per day), aggregates edge
        and liquidity metrics, ranks by annualized return and volume, and
        returns a sorted table with friendly time labels.
        """
        if df.empty:
            return df
        
        if resample_15m:
            df = df.copy()
            df["slot_15m"] = (df["minute_of_day"] // 15).astype(int)
            
            grp = df.groupby("slot_15m").agg(
                mean_ann_return=("ann_return", "mean"),
                p75_ann_return=("ann_return", lambda x: np.percentile(x, 75)),
                mean_basis_bps=("basis_bps", "mean"),
                p75_basis_bps=("basis_bps", lambda x: np.percentile(x, 75)),
                total_volume_btc=("total_vol_btc", "sum"),
                median_volume_btc=("total_vol_btc", "median"),
                avg_days_to_expiry=("days_to_expiry", "mean"),
                samples=("basis_bps", "count"),
            ).reset_index()
            
            # Create time labels
            grp["time_label"] = grp["slot_15m"].apply(
                lambda slot: f"{(slot*15)//60:02d}:{(slot*15)%60:02d}-{((slot+1)*15)//60%24:02d}:{((slot+1)*15)%60:02d} UTC"
            )
            
        else:
            grp = df.groupby("minute_of_day").agg(
                mean_ann_return=("ann_return", "mean"),
                p75_ann_return=("ann_return", lambda x: np.percentile(x, 75)),
                mean_basis_bps=("basis_bps", "mean"),
                p75_basis_bps=("basis_bps", lambda x: np.percentile(x, 75)),
                total_volume_btc=("total_vol_btc", "sum"),
                median_volume_btc=("total_vol_btc", "median"),
                avg_days_to_expiry=("days_to_expiry", "mean"),
                samples=("basis_bps", "count"),
            ).reset_index()
            
            grp["time_label"] = grp["minute_of_day"].apply(
                lambda m: f"{m//60:02d}:{m%60:02d} UTC"
            )
        
        # Ranking
        grp["rank_by_ann"] = grp["mean_ann_return"].rank(ascending=False, method="min")
        grp["rank_by_vol"] = grp["total_volume_btc"].rank(ascending=False, method="min")
        grp["combined_rank"] = grp["rank_by_ann"] + grp["rank_by_vol"]
        
        # Sort and return
        return grp.sort_values(["combined_rank", "mean_ann_return", "total_volume_btc"],
                              ascending=[True, False, False]).reset_index(drop=True)

    def run_synchronized_analysis(self, end_date: str, resample_15m: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run synchronized cash-and-carry analysis with precise timing alignment.
        
        Args:
            end_date: End date in "YYYY-MM-DD" format
            resample_15m: Whether to resample to 15-minute intervals
            
        Returns:
            Dictionary with DataFrames, e.g., buy_leaderboard. The SELL side is
            intentionally omitted in this workflow.
        """
        
        # Get precise time boundaries
        start_ms, end_ms = self._get_precise_time_boundaries(end_date, self.lookback_days)
        
        print(f"Synchronized analysis")
        print(f"Time window: {self._ms_to_dt(start_ms):%Y-%m-%d %H:%M:%S} → {self._ms_to_dt(end_ms):%Y-%m-%d %H:%M:%S} UTC")
        print(f"Duration: {self.lookback_days} days")
        print("=" * 80)
        
        # Fetch spot data
        print(f"Fetching SPOT {self.spot_symbol}...")
        spot_klines = self._fetch_klines_synchronized(self.spot, self.spot_symbol, start_ms, end_ms)
        spot_df = self._klines_to_df(spot_klines, "spot")
        
        if spot_df.empty:
            print("No spot data retrieved")
            return {"buy_leaderboard": pd.DataFrame(), "sell_leaderboard": pd.DataFrame()}
        
        print(f"Spot data: {len(spot_df)} bars")
        
        # Fetch futures data
        print(f"Fetching FUTURES {self.cm_pair}...")

        # Attempt to discover actual CURRENT/NEXT symbols for clearer logging
        cur_sym_display = "CURRENT_QUARTER"
        nxt_sym_display = "NEXT_QUARTER"
        try:
            info = self.cm.exchange_info()
            cur_sym = None; nxt_sym = None
            for s in info.get("symbols", []):
                ct = s.get("contractType")
                sym = s.get("symbol", "")
                if sym.startswith("BTCUSD_"):
                    if ct == "CURRENT_QUARTER" and cur_sym is None:
                        cur_sym = sym
                    elif ct == "NEXT_QUARTER" and nxt_sym is None:
                        nxt_sym = sym
            # Determine if requested end date is historical (before today minus 30 days)
            hist_mode = end_ms < (int(dt.datetime.now(dt.timezone.utc).timestamp()*1000) - 30*24*3600*1000)

            if (not cur_sym or not nxt_sym) and not hist_mode:
                raise RuntimeError("Could not retrieve exact CURRENT/NEXT BTCUSD futures symbols from Binance.")

            if hist_mode:
                # Derive expiry codes based on end date reference
                ref_dt = self._ms_to_dt(end_ms)
                quarter_months = [3,6,9,12]
                def _expiry(dt_ref, offset_q):
                    q_idx = (dt_ref.month-1)//3 + offset_q
                    year = dt_ref.year + q_idx//4
                    q_idx = q_idx%4
                    qm = quarter_months[q_idx]
                    q_end = dt.datetime(year, qm, 1, tzinfo=dt.timezone.utc)
                    if qm==12:
                        q_end=q_end.replace(month=12, day=31)
                    else:
                        q_end=(q_end.replace(month=qm+1, day=1)-dt.timedelta(days=1))
                    while q_end.weekday()!=4:
                        q_end-=dt.timedelta(days=1)
                    return q_end.strftime("%y%m%d")
                cur_sym_display = f"BTCUSD_{_expiry(ref_dt,0)}"
                nxt_sym_display = f"BTCUSD_{_expiry(ref_dt,1)}"
            else:
                cur_sym_display = cur_sym
                nxt_sym_display = nxt_sym
        except Exception:
            # On API failure fallback to derived codes
            ref_dt = self._ms_to_dt(end_ms)
            quarter_months = [3,6,9,12]
            def _expiry(dt_ref, offset_q):
                q_idx=(dt_ref.month-1)//3+offset_q
                year=dt_ref.year+q_idx//4
                q_idx=q_idx%4
                qm=quarter_months[q_idx]
                q_end=dt.datetime(year,qm,1,tzinfo=dt.timezone.utc)
                if qm==12:
                    q_end=q_end.replace(month=12,day=31)
                else:
                    q_end=(q_end.replace(month=qm+1,day=1)-dt.timedelta(days=1))
                while q_end.weekday()!=4:
                    q_end-=dt.timedelta(days=1)
                return q_end.strftime("%y%m%d")
            cur_sym_display=f"BTCUSD_{_expiry(ref_dt,0)}"
            nxt_sym_display=f"BTCUSD_{_expiry(ref_dt,1)}"

        # Current quarter
        print(f"   {cur_sym_display}...")
        cur_klines = self._fetch_klines_synchronized(
            self.cm, self.cm_pair, start_ms, end_ms, 
            is_continuous=True, contract_type="CURRENT_QUARTER"
        )
        cur_df = self._klines_to_df(cur_klines, "fut_cur")
        
        # Next quarter
        print(f"   {nxt_sym_display}...")
        nxt_klines = self._fetch_klines_synchronized(
            self.cm, self.cm_pair, start_ms, end_ms,
            is_continuous=True, contract_type="NEXT_QUARTER"
        )
        nxt_df = self._klines_to_df(nxt_klines, "fut_nxt")
        
        if cur_df.empty and nxt_df.empty:
            print("No futures data retrieved")
            return {"buy_leaderboard": pd.DataFrame(), "sell_leaderboard": pd.DataFrame()}
        
        print(f"Futures data: Current={len(cur_df)}, Next={len(nxt_df)}")
        
        # Synchronize and merge data
        print("Synchronizing and merging data...")
        
        all_basis_data = []
        
        # Process current quarter if available
        if not cur_df.empty:
            print("   Processing Current Quarter alignment...")
            merged_cur = spot_df.merge(cur_df, on="open_time", how="inner")
            print(f"   Current Quarter aligned bars: {len(merged_cur)}")
            
            if not merged_cur.empty:
                basis_cur = self._compute_basis_metrics(merged_cur, "fut_cur")
                all_basis_data.append(basis_cur)
        
        # Process next quarter if available
        if not nxt_df.empty:
            print("   Processing Next Quarter alignment...")
            merged_nxt = spot_df.merge(nxt_df, on="open_time", how="inner")
            print(f"   Next Quarter aligned bars: {len(merged_nxt)}")
            
            if not merged_nxt.empty:
                basis_nxt = self._compute_basis_metrics(merged_nxt, "fut_nxt")
                all_basis_data.append(basis_nxt)
        
        if not all_basis_data:
            print("No aligned data after synchronization")
            return {"buy_leaderboard": pd.DataFrame(), "sell_leaderboard": pd.DataFrame()}
        
        # Combine all basis data
        combined_basis = pd.concat(all_basis_data, ignore_index=True)
        print(f"Combined basis data: {len(combined_basis)} rows")

        # Save raw price series for verification later. (disabled)
        # try:
        #     price_df = spot_df[["open_time", "spot_close"]].copy()
        #     if not cur_df.empty:
        #         price_df = price_df.merge(cur_df[["open_time", "fut_cur_close"]], on="open_time", how="left")
        #     if not nxt_df.empty:
        #         price_df = price_df.merge(nxt_df[["open_time", "fut_nxt_close"]], on="open_time", how="left")
        #
        #     price_df.to_csv("extracted_prices.csv", index=False)
        #     print("Saved price series to extracted_prices.csv")
        # except Exception as e:
        #     print(f"Warning: failed to save price series data: {e}")

        # Also calculate a simple historical average basis for context.
        historical_avg_basis = combined_basis['basis_bps'].mean() if not combined_basis.empty else 200.0
        print(f"   Historical avg basis: {historical_avg_basis:.2f} bps")

        # Split into BUY and SELL based on basis thresholds
        buy_data = combined_basis[combined_basis["basis_bps"] > self.basis_bps_min_buy].copy()

        print(f"BUY opportunities: {len(buy_data)} samples")

        # Create leaderboard (SELL side removed per new requirement)
        buy_leaderboard = self._create_leaderboard(buy_data, resample_15m)

        # Display results
        def _display_leaderboard(df: pd.DataFrame, name: str):
            if df.empty:
                print(f"{name}: No opportunities found")
                return
            
            print(f"\nTop 10 — {name}:")
            head = df.head(10)
            for _, row in head.iterrows():
                time_label = row["time_label"]
                ann_ret = row["mean_ann_return"] * 100
                basis = row["mean_basis_bps"]
                volume = row["total_volume_btc"]
                samples = row["samples"]
                print(f"   {time_label:<20} | Ann: {ann_ret:6.2f}% | Basis: {basis:7.1f}bp | Vol: {volume:8.1f}₿ | Samples: {samples}")

        _display_leaderboard(buy_leaderboard, "BUY Leaderboard")

        # Store results
        self.last_buy_leaderboard = buy_leaderboard
        self.last_sell_leaderboard = None  # sell leaderboard removed

        return {
            "buy_leaderboard": buy_leaderboard,
            "historical_avg_basis": historical_avg_basis
        }


def derive_slot_weights(buy_df: pd.DataFrame) -> dict:
    """
    Build 96-slot (15m) weights from the BUY leaderboard.
    Uses inverse rank + robust z-scores of ann_return and volume.
    Handles empty inputs and missing slots gracefully.
    """
    nslots = 96
    if buy_df is None or buy_df.empty:
        return {k: 1.0/nslots for k in range(nslots)}

    df = buy_df.copy()

    # Ensure we have a 15m slot column
    if "slot_15m" not in df.columns:
        if "minute_of_day" in df.columns:
            df["slot_15m"] = (df["minute_of_day"] // 15).astype(int)
        else:
            # fallback: derive from time_label like "HH:MM-HH:MM UTC"
            def _tl_to_slot(tl: str) -> int:
                try:
                    hh, mm = map(int, tl.split(" ")[0].split("-")[0].split(":"))
                    return (hh*60 + mm) // 15
                except Exception:
                    return 0
            df["slot_15m"] = df["time_label"].astype(str).map(_tl_to_slot)

    # Clean numeric cols
    for c in ["mean_ann_return", "total_volume_btc", "combined_rank"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0

    df["combined_rank"] = df["combined_rank"].fillna(df["combined_rank"].max() + 1)
    df["mean_ann_return"] = df["mean_ann_return"].fillna(0.0)
    df["total_volume_btc"] = df["total_volume_btc"].fillna(0.0)

    inv_rank = 1.0 / df["combined_rank"].clip(lower=1)

    def robust_z(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        med = np.nanmedian(s)
        mad = np.nanmedian(np.abs(s - med))
        if not np.isfinite(mad) or mad < 1e-12:
            std = np.nanstd(s)
            std = std if (np.isfinite(std) and std > 1e-12) else 1.0
            return (s - med) / std
        return (s - med) / (mad + 1e-12)

    ann_z = robust_z(df["mean_ann_return"])
    vol_z = robust_z(df["total_volume_btc"])

    # Blend (heaviest weight on inverse rank)
    raw = (0.6*inv_rank +
           0.25*(ann_z - np.nanmin(ann_z)) + #shift the ann_z to the right by the minimum value of ann_z so that the minimum value of ann_z is 0
           0.15*(vol_z - np.nanmin(vol_z))) #shift the vol_z to the right by the minimum value of vol_z so that the minimum value of vol_z is 0
    raw = np.clip(raw, 1e-9, None)

    # Average raw score per 15m slot (in case multiple rows per slot)
    slot_raw = pd.DataFrame({"slot": df["slot_15m"].astype(int), "raw": raw}).groupby("slot")["raw"].mean()

    # Fill all 96 slots; missing → 0 then renormalize
    weights = np.zeros(nslots, dtype=float)
    for k, v in slot_raw.items():
        if 0 <= k < nslots:
            weights[k] = float(v)
    if weights.sum() <= 0:
        weights[:] = 1.0/nslots
    else:
        weights /= weights.sum()

    return {k: float(weights[k]) for k in range(nslots)}


# Task 1 execution (24h must-invest) using Step-1 BUY leader
# 
# board + a simulator
# Assumes the scanner class above is available in this module.

import os, math, time, random, datetime as dt
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
from dataclasses import dataclass

from binance.spot import Spot
from binance.cm_futures import CMFutures

# BacktestPriceFeed for historical runs
class BacktestPriceFeed:
    """
    Serve market data from a pre-fetched 24-hour slice for backtests instead of
    calling live APIs.

    Provides spot and futures (CURRENT/NEXT) mid-price and spread at arbitrary
    timestamps, plus a recent-15m spot volume proxy for PoV sizing.
    """
    def __init__(self, spot_df: pd.DataFrame, fut_cur_df: pd.DataFrame, fut_nxt_df: pd.DataFrame):
        self.spot_df = self._prepare_df(spot_df, "spot")
        self.fut_cur_df = self._prepare_df(fut_cur_df, "fut_cur")
        self.fut_nxt_df = self._prepare_df(fut_nxt_df, "fut_nxt")

    @staticmethod
    def _prepare_df(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        # Use raw column names before prefixing
        df['mid_price'] = (pd.to_numeric(df['high'], errors='coerce') + pd.to_numeric(df['low'], errors='coerce')) / 2.0
        df['spread'] = (pd.to_numeric(df['high'], errors='coerce') - pd.to_numeric(df['low'], errors='coerce')).clip(lower=1e-2)
        df['volume_numeric'] = pd.to_numeric(df.get(f"{prefix}_volume", df.get("volume")), errors='coerce')

        if 'open_time' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['open_time']):
             df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        return df.set_index('open_time').sort_index()

    def _get_row_at(self, df: pd.DataFrame, timestamp: dt.datetime) -> Optional[pd.Series]:
        if df.empty:
            return None
        try:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
            pos = df.index.get_indexer([timestamp], method='ffill')[0]
            if pos == -1: return None
            return df.iloc[pos]
        except Exception:
            return None

    def get_spot_mid_spread(self, timestamp: dt.datetime) -> Tuple[float, float]:
        row = self._get_row_at(self.spot_df, timestamp)
        if row is not None:
            return float(row['mid_price']), float(row['spread'])
        return np.nan, np.nan

    def get_future_mid_spread(self, timestamp: dt.datetime, contract_type: str) -> Tuple[float, float]:
        df = self.fut_cur_df if contract_type == "CURRENT_QUARTER" else self.fut_nxt_df
        row = self._get_row_at(df, timestamp)
        if row is not None:
            return float(row['mid_price']), float(row['spread'])
        return np.nan, np.nan
        
    #this function is used to get the recent 15m spot volume in BTC
    def get_recent_15m_spot_volume_btc(self, timestamp: dt.datetime) -> float:
        if self.spot_df.empty: return 100.0
        
        end_ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=dt.timezone.utc)
        start_ts = end_ts - dt.timedelta(minutes=15)
        
        subset = self.spot_df.loc[start_ts:end_ts]
        return max(subset['volume_numeric'].sum(), 1.0) if not subset.empty else 1.0

class BinanceSimulator(object):
    """Simple order simulator with a fixed fill probability.

    Mimics LIMIT order behavior by randomly filling or canceling based on a
    configured probability. Used to test execution logic without hitting
    external services.
    """
    def __init__(self,
                 api_key=None,
                 secret_key=None,
                 order_fill_prob: float = 0.9):
        # Load from environment if not provided
        api_key = api_key or os.getenv('BINANCE_API_KEY')
        secret_key = secret_key or os.getenv('BINANCE_API_SECRET')
        
        self.spot = Spot(api_key=api_key, api_secret=secret_key)
        self.cm_future = CMFutures(key=api_key, secret=secret_key)
        self.order_fill_prob = float(order_fill_prob)

    def place_order(self, order_params: dict):
        response = order_params.copy()
        x = random.randint(1, 100)
        response['status'] = 'filled' if x <= self.order_fill_prob*100 else 'canceled'
        return response

# ----- Helpers -----
def _round_down(n: float, decimals: int) -> float:
    f = 10 ** decimals
    return math.floor(n * f) / f

@dataclass
class ExecParams:
    capital_usdt: float = 1_000_000.0
    hours: int = 24
    bucket_minutes: int = 15            # 24h * 15m = 96 slices
    fee_buffer: float = 0.001           # 10 bps buffer
    pov_base: float = 0.10              # base PoV (Participation of Volume)
    pov_min: float = 0.02
    pov_max: float = 0.25
    max_bucket_attempts: int = 6        # retries inside a 15m bucket per leg
    max_final_attempts: int = 9999      # final forced attempts to guarantee fill
    behind_sched_thresh: float = 0.05   # 5% behind → aggressive
    basis_compress_bps: float = 20.0    # basis drop vs arrival → aggressive
    contract_usd_default: float = 100.0 # COIN-M BTCUSD contract value
    rng_seed: int = 42
    cm_pair: str = "BTCUSD"             # COIN-M pair for continuous klines

class AdaptiveCashCarryExecutor:
    """
    Execute a 24-hour cash-and-carry entry: long spot BTCUSDT and short COIN-M
    BTCUSD (CURRENT/NEXT), sliced into 15-minute buckets.

    - Uses a weight schedule derived from prior analysis (volume profile/edge)
    - Caps slice size by Participation of Volume (PoV)
    - Applies an implementation-shortfall overlay to toggle aggressiveness
    - Chooses CURRENT vs NEXT each slice based on the observed basis
    - Falls back to continuous-klines mids if individual symbols are unavailable
    - Forces completion in a final sweep to guarantee full notional within 24h

    Works in both live (forward-sim) and backtest (historical-sim) modes.
    """
    def __init__(self, api_key: str, api_secret: str, slot_weights: Dict[int,float], simulator: BinanceSimulator, params: ExecParams=ExecParams(), price_feed: Optional[BacktestPriceFeed] = None, historical_avg_basis: float = 200.0):
        self.params = params
        self.slot_weights = slot_weights if slot_weights and len(slot_weights)==96 else {k:1.0/96 for k in range(96)}
        self.price_feed = price_feed  # Used when backtesting
        self.historical_avg_basis = historical_avg_basis  # Carried for opportunistic sizing
        random.seed(params.rng_seed)
        # clients
        self.spot = Spot(api_key=api_key, api_secret=api_secret)
        self.cm   = CMFutures(key=api_key, secret=api_secret)
        self.sim  = simulator
        # state
        self.arrival_spot_mid: Optional[float] = None
        self.arrival_basis_bps: Optional[float] = None
        self.exec_log: List[Dict] = []
        self._sym_cache: Optional[Dict[str, Dict]] = None  # futures meta cache

    # ---------- order book / price helpers ----------
    @staticmethod
    def _mid_spread_from_depth(ob: Optional[Dict]) -> Tuple[float, float]:
        # Compute a best-quote mid price and spread from an order book snapshot (depth)
        if not ob or not ob.get("bids") or not ob.get("asks"):
            return (np.nan, np.nan)
        bid = float(ob["bids"][0][0]); ask = float(ob["asks"][0][0])
        return (bid+ask)/2.0, max(ask-bid, 1e-2) #Output the mid price and the spread

    def _spot_mid_spread(self, timestamp: dt.datetime) -> Tuple[float, float]:
        if self.price_feed:
            return self.price_feed.get_spot_mid_spread(timestamp) #Output the mid price and the spread
        
        try: #for live mode
            ob = self.spot.depth("BTCUSDT", limit=5)
            mid, spr = self._mid_spread_from_depth(ob)
            if not np.isfinite(mid):
                raise ValueError("no depth")
            return mid, spr

        except Exception:
            # If depth fails, I fall back to recent 1m klines and use last close; spread heuristic from last (high-low)
            end = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
            start = end - 2*60*1000
            kl = self.spot.klines(symbol="BTCUSDT", interval="1m", startTime=start, endTime=end, limit=2)
            if kl:
                last = kl[-1]
                close = float(last[4]); hi = float(last[2]); lo = float(last[3])
                spr = max((hi - lo) * 0.2, 1e-2)
                return close, spr
            # Worst-case static defaults
            return 30000.0, 5.0

    def _fetch_current_future_symbols(self) -> Dict[str, Dict]:
        if self._sym_cache is not None:
            return self._sym_cache
        out = {}
        try:
            info = self.cm.exchange_info()
            for s in info.get("symbols", []):
                sym = s.get("symbol","")
                if not sym.startswith("BTCUSD_"):
                    continue
                if s.get("status") != "TRADING":
                    continue
                ct = s.get("contractType","")
                if ct not in ("CURRENT_QUARTER","NEXT_QUARTER"):
                    continue
                out[sym] = {
                    "contractType": ct,
                    "contractSize": float(s.get("contractSize", self.params.contract_usd_default)),
                    "deliveryDate": int(s.get("deliveryDate", 0))
                }
            out = dict(sorted(out.items(), key=lambda kv: kv[1]["deliveryDate"]))
        except Exception:
            out = {}
        self._sym_cache = out
        return out

    def _future_mid_spread_by_symbol(self, symbol: str, timestamp: dt.datetime) -> Tuple[float, float]:
        if self.price_feed:
            meta = self._fetch_current_future_symbols().get(symbol, {}) #get the contract type
            contract_type = meta.get("contractType")
            if contract_type:
                return self.price_feed.get_future_mid_spread(timestamp, contract_type)
            return np.nan, np.nan

        try:
            ob = self.cm.depth(symbol, limit=5)
            mid, spr = self._mid_spread_from_depth(ob)
            if np.isfinite(mid):
                return mid, spr
            raise ValueError("no depth")
        except Exception:
            # Fall back to continuous klines if a tradable symbol cannot be inferred
            return np.nan, np.nan

    def _future_mid_spread_by_contractType(self, contract_type: str, timestamp: dt.datetime) -> Tuple[float, float]:
        """
        Fallback when no tradable symbol is visible.
        Uses continuous klines to estimate a mid-price and a spread proxy for
        the requested contract type (CURRENT/NEXT), enabling robust execution
        even if the per-expiry order book is unavailable.
        """
        if self.price_feed:
            return self.price_feed.get_future_mid_spread(timestamp, contract_type)

        try:
            end = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
            start = end - 3*60*1000
            kl = self.cm.continuous_klines(pair=self.params.cm_pair,
                                           contractType=contract_type,
                                           interval="1m",
                                           startTime=start, endTime=end, limit=3)
            if kl:
                last = kl[-1]
                close = float(last[4]); hi = float(last[2]); lo = float(last[3])
                spr = max((hi - lo) * 0.2, 0.5)  # small heuristic spread
                return close, spr
        except Exception:
            pass
        return np.nan, np.nan

    def _pick_best_future(self, spot_mid: float, timestamp: dt.datetime) -> Tuple[str, Dict, float, float, float]:
        """
        Choose CURRENT vs NEXT by higher positive basis (bps).
        Returns (identifier, meta, fut_mid, fut_spread, basis_bps).
        - identifier: real symbol if available, else synthetic like 'BTCUSD-CURRENT_QUARTER-SYN'
        - meta: contains contractSize (falls back to default if unknown)
        """
        # In backtests historical per-expiry symbols cannot be queried; build choices
        # from the price feed using CURRENT/NEXT continuous klines.
        if self.price_feed:
            choices = []
            for ct in ("CURRENT_QUARTER", "NEXT_QUARTER"):
                fut_mid, fut_spr = self.price_feed.get_future_mid_spread(timestamp, ct)
                if np.isfinite(fut_mid):
                    meta = {"contractType": ct, "contractSize": self.params.contract_usd_default}
                    # Derive historical expiry code (yyMMdd) for logging clarity
                    def _quarter_expiry(ts: dt.datetime, ct_str: str) -> str:
                        # Determine the base quarter (current or next) relative to timestamp ts
                        quarter_months = [3, 6, 9, 12]
                        month = ts.month
                        year = ts.year
                        # current quarter index 0-3
                        q_idx = (month-1)//3
                        if ct_str == "NEXT_QUARTER":
                            q_idx += 1
                        year += q_idx//4
                        q_idx = q_idx % 4
                        qm = quarter_months[q_idx]
                        # last Friday of quarter month
                        q_end = dt.datetime(year, qm, 1, tzinfo=dt.timezone.utc)
                        if qm == 12:
                            q_end = q_end.replace(month=12, day=31)
                        else:
                            q_end = (q_end.replace(month=qm+1, day=1) - dt.timedelta(days=1))
                        while q_end.weekday() != 4:
                            q_end -= dt.timedelta(days=1)
                        return q_end.strftime("%y%m%d")

                    expiry_suffix = _quarter_expiry(timestamp, ct)
                    ident = f"BTCUSD_{expiry_suffix}"
                    bps = (fut_mid - spot_mid) / spot_mid * 1e4
                    choices.append((ident, meta, fut_mid, fut_spr, bps))
            if not choices:
                raise RuntimeError("Price feed did not return valid futures prices for CURRENT/NEXT quarter.")
            choices.sort(key=lambda x: x[4], reverse=True)
            return choices[0]

        # Live mode
        syms = self._fetch_current_future_symbols()

        choices = []
        if syms:
            # In live mode, try real symbols 
            for sym, meta in syms.items():
                fut_mid, fut_spr = self._future_mid_spread_by_symbol(sym, timestamp)
                if not np.isfinite(fut_mid):
                    # If depth fails, fall back to the continuous series
                    fut_mid, fut_spr = self._future_mid_spread_by_contractType(meta["contractType"], timestamp)
                if np.isfinite(fut_mid):
                    bps = (fut_mid - spot_mid) / spot_mid * 1e4
                    choices.append((sym, meta, fut_mid, fut_spr, bps))
        # If no valid real futures symbols were found, abort.
        if not choices:
            raise RuntimeError("No tradable BTCUSD CURRENT/NEXT quarter futures symbols available from Binance.")

        # Pick the one with max basis bps
        choices.sort(key=lambda x: x[4], reverse=True)
        return choices[0]

    def _recent_15m_spot_volume_btc(self, timestamp: dt.datetime, symbol="BTCUSDT") -> float:
        if self.price_feed:
            return self.price_feed.get_recent_15m_spot_volume_btc(timestamp)

        end = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
        start = end - 15*60*1000
        try:
            kl = self.spot.klines(symbol=symbol, interval="1m", startTime=start, endTime=end, limit=15)
            if not kl: return 100.0
            return max(sum(float(x[5]) for x in kl), 1.0)
        except Exception:
            return 100.0

    def _btc_to_contracts(self, btc_qty: float, spot_mid: float, contract_size_usd: float) -> int:
        return max(1, int(math.floor((btc_qty * spot_mid) / contract_size_usd)))

    def _place_until_filled(self, order_params: dict, max_attempts: int) -> Dict:
        """
        Route through the simulator until an order fills or attempts are exhausted.

        Returns the last simulator response with additional fields:
        - attempts: number of tries within this bucket
        - filled_qty: the requested quantity/contracts when filled, else 0
        """
        attempts = 0
        last = None
        while attempts < max_attempts:
            attempts += 1
            last = self.sim.place_order(order_params)
            if last.get("status") == "filled":
                filled = order_params.get("quantity", order_params.get("contracts", 0.0))
                return {**last, "attempts": attempts, "filled_qty": float(filled)}
        return {**(last or {}), "attempts": attempts, "filled_qty": 0.0, "status": "canceled"}

    # ---------- core run ----------
    def run(self, start_dt: Optional[dt.datetime]=None, realtime: bool=False) -> Dict:
        P = self.params
        start_dt = start_dt or dt.datetime.now(dt.timezone.utc)
        buckets = int((P.hours*60)//P.bucket_minutes)

        # arrival mids & basis (use start_dt as timestamp)
        spot_mid, spot_spread = self._spot_mid_spread(start_dt)
        ident0, meta0, fut_mid0, fut_spread0, _ = self._pick_best_future(spot_mid, start_dt)
        self.arrival_spot_mid = spot_mid
        self.arrival_basis_bps = (fut_mid0 - spot_mid) / spot_mid * 1e4

        # target BTC and contracts
        target_btc = (P.capital_usdt * (1 - P.fee_buffer)) / spot_mid
        target_btc = _round_down(target_btc, 4)  # 1e-4 BTC lot
        contract_size = float(meta0.get("contractSize", P.contract_usd_default))
        # For inverse futures, hedge contracts should be sized with the futures price (not spot): N = round(Q * F / contract_size)
        target_ct  = self._btc_to_contracts(target_btc, fut_mid0, contract_size)

        remaining_btc = target_btc
        remaining_ct  = target_ct
        filled_btc = 0.0
        filled_ct  = 0

        weights_vec = np.array([self.slot_weights.get(k, 1.0/96) for k in range(96)], dtype=float)
        weights_vec = weights_vec / weights_vec.sum()

        for i in range(buckets):
            ts = start_dt + dt.timedelta(minutes=P.bucket_minutes * i)
            utc_slot = ((ts.hour*60 + ts.minute) // 15) % 96
            w = float(weights_vec[utc_slot])

            # Refresh mids & SOR (pass current simulation timestamp `ts`)
            spot_mid, spot_spread = self._spot_mid_spread(ts)
            ident, meta, fut_mid, fut_spread, cur_bps = self._pick_best_future(spot_mid, ts)

            # Size opportunistically versus the historical average basis
            opportunity_score = 1.0
            if self.historical_avg_basis > 1e-9:
                opportunity_score = cur_bps / self.historical_avg_basis
            capped_opportunity_score = np.clip(opportunity_score, 0.5, 2.0)

            # time_frac = the fraction of the day that has passed
            time_frac = (i+1)/buckets
            progress = (target_btc - remaining_btc) / max(target_btc, 1e-9)
            # sched_dev = the deviation from the schedule
            sched_dev = progress - time_frac  # <0 behind schedule

            # PoV = Participation of Volume sizing
            recent_vol_btc = self._recent_15m_spot_volume_btc(ts, "BTCUSDT")
            basis_delta = cur_bps - (self.arrival_basis_bps or cur_bps)

            pov = P.pov_base * (1 + 0.8 * (-sched_dev)) * (1 + 0.005*abs(cur_bps))
            pov = max(P.pov_min, min(P.pov_max, pov))

            # Child sizing: weight-based AND PoV-capped (child is the size of the order)
            child_from_weight = max(0.0005, target_btc * w * capped_opportunity_score)
            child_from_pov    = max(0.0005, pov * recent_vol_btc)
            child_btc = min(remaining_btc, min(child_from_weight, child_from_pov))
            if child_btc <= 0.0:
                continue

            # Aggression flag (if the schedule is behind or the basis is compressing, then we cross more aggressively)
            cross = (sched_dev < -P.behind_sched_thresh) or (basis_delta < -P.basis_compress_bps)

            #for pricing the spot/future
            px_spot = (spot_mid - 0.5*spot_spread) if not cross else (spot_mid + 0.5*spot_spread)
            px_fut  = (fut_mid + 0.5*fut_spread) if not cross else (fut_mid - 0.5*fut_spread)

            # Futures child contracts: allocate by weight toward the precomputed target_ct
            desired_ct_for_slot = int(max(1, round(w * target_ct)))
            child_ct = min(remaining_ct, desired_ct_for_slot)

            # Paired ordering with retry on cancellations
            # SPOT leg
            spot_params = {
                "market": "SPOT",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "price": float(px_spot),
                "quantity": float(round(child_btc, 6))
            }
            spot_resp = self._place_until_filled(spot_params, max_attempts=P.max_bucket_attempts)

            # FUTURES leg (identifier can be real symbol or synthetic)
            fut_symbol = ident
            fut_params = {
                "market": "FUT",
                "symbol": fut_symbol,
                "side": "SELL",
                "type": "LIMIT",
                "price": float(px_fut),
                "contracts": int(child_ct)
            }
            fut_resp = self._place_until_filled(fut_params, max_attempts=P.max_bucket_attempts)

            # If one leg canceled while the other filled, keep retrying the canceled leg to re-hedge in-bucket
            if spot_resp["status"] != "filled" and fut_resp["status"] == "filled":
                spot_resp = self._place_until_filled(spot_params, max_attempts=P.max_bucket_attempts*2)
            if fut_resp["status"] != "filled" and spot_resp["status"] == "filled":
                fut_resp = self._place_until_filled(fut_params, max_attempts=P.max_bucket_attempts*2)

            # Accumulate
            fill_spot = float(spot_resp.get("filled_qty", 0.0))
            fill_ct   = int(fut_resp.get("filled_qty", 0.0))
            filled_btc += fill_spot
            filled_ct  += fill_ct
            remaining_btc = max(0.0, target_btc - filled_btc)
            remaining_ct  = max(0,   target_ct  - filled_ct)

            self.exec_log.append({
                "ts": ts, "slot": int(utc_slot), "weight": w, "pov": pov, "sched_dev": sched_dev, "cross": bool(cross),
                "opportunity_score": float(capped_opportunity_score),
                "spot_mid": spot_mid, "fut_mid": fut_mid, "basis_bps": cur_bps,
                "spot_px": px_spot, "fut_px": px_fut,
                "req_spot_btc": child_btc, "req_fut_ct": child_ct,
                "spot_attempts": spot_resp.get("attempts", 0), "fut_attempts": fut_resp.get("attempts", 0),
                "spot_status": spot_resp.get("status","?"), "fut_status": fut_resp.get("status","?"),
                "fill_spot_btc": fill_spot, "fill_fut_ct": fill_ct,
                "fut_symbol": fut_symbol, "fut_contractType": meta.get("contractType", "UNKNOWN")
            })

            if remaining_btc < 1e-6 and remaining_ct <= 0:
                break

            if realtime:
                # In live mode, print a short progress line
                last_log = self.exec_log[-1]
                ts_utc = last_log['ts']
                ts_sgt = ts_utc + dt.timedelta(hours=8)

                print(
                    f"[{ts_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC | {ts_sgt.strftime('%H:%M:%S')} SGT] "
                    f"Slot {last_log['slot']:>2} | "
                    f"Fut: {last_log['fut_symbol']} | "
                    f"Basis: {last_log['basis_bps']:.1f}bp | "
                    f"Target: {last_log['req_spot_btc']:.4f} BTC | "
                    f"Filled: {last_log['fill_spot_btc']:.4f} BTC | "
                    f"Progress: {(filled_btc/target_btc)*100:.1f}%"
                )

                next_ts_utc = start_dt + dt.timedelta(minutes=P.bucket_minutes * (i+1))
                next_ts_sgt = next_ts_utc + dt.timedelta(hours=8)
                print(f"   Waiting... Next execution at {next_ts_utc.strftime('%H:%M:%S')} UTC ({next_ts_sgt.strftime('%H:%M:%S')} SGT)\n")
                sleep_s = max(0.0, (next_ts_utc - dt.datetime.now(dt.timezone.utc)).total_seconds())
                time.sleep(sleep_s)

        # Final sweep to ensure full completion
        if remaining_btc > 0.0 or remaining_ct > 0:
            final_ts = start_dt + dt.timedelta(minutes=P.bucket_minutes * buckets)
            original_prob = self.sim.order_fill_prob
            try:
                # Make final fills deterministic in the simulator
                self.sim.order_fill_prob = 1.0

                # refresh prices (use final timestamp)
                spot_mid, spot_spread = self._spot_mid_spread(final_ts)
                ident_f, meta_f, fut_mid_f, fut_spread_f, _ = self._pick_best_future(spot_mid, final_ts)

                # SPOT residue
                if remaining_btc > 0.0:
                    spot_params = {
                        "market": "SPOT",
                        "symbol": "BTCUSDT",
                        "side": "BUY",
                        "type": "LIMIT",
                        "price": float(spot_mid + 0.5*spot_spread),  # marketable
                        "quantity": float(round(remaining_btc, 6))
                    }
                    done = 0.0; attempts = 0
                    while done < remaining_btc and attempts < P.max_final_attempts:
                        attempts += 1
                        resp = self.sim.place_order(spot_params)
                        if resp.get("status") == "filled":
                            done = remaining_btc
                    filled_btc += remaining_btc
                    remaining_btc = 0.0

                # FUT residue
                if remaining_ct > 0:
                    fut_params = {
                        "market": "FUT",
                        "symbol": ident_f,
                        "side": "SELL",
                        "type": "LIMIT",
                        "price": float(fut_mid_f - 0.5*fut_spread_f),  # marketable
                        "contracts": int(remaining_ct)
                    }
                    done = 0; attempts = 0
                    while done < remaining_ct and attempts < P.max_final_attempts:
                        attempts += 1
                        resp = self.sim.place_order(fut_params)
                        if resp.get("status") == "filled":
                            done = remaining_ct
                    filled_ct += remaining_ct
                    remaining_ct = 0
            finally:
                self.sim.order_fill_prob = original_prob

        # Pack result
        log_df = pd.DataFrame(self.exec_log)
        return {
            "summary": {
                "target_btc": float(target_btc), "filled_btc": float(filled_btc),
                "target_ct":  int(target_ct),    "filled_ct":  int(filled_ct),
                "arrival_spot_mid": float(self.arrival_spot_mid),
                "arrival_basis_bps": float(self.arrival_basis_bps)
            },
            "log": log_df
        }

# ----- Run the executor (paper mode using a simulator) -----
def main():
    """Main function to run the interactive cash-and-carry analysis and execution."""
    
    # 1. Get user input
    print("--- Cash & Carry Arbitrage Tool ---")
    
    while True:
        try:
            date_str = input("Enter the date for analysis (YYYY-MM-DD): ")
            target_date_obj = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    while True:
        try:
            lookback_str = input("Enter lookback days for analysis (1-200): ")
            lookback_days = int(lookback_str)
            if 1 <= lookback_days <= 200:
                break
            else:
                print("Please enter a number between 1 and 200.")
        except ValueError:
            print("Please enter a valid number.")
            
    # 2. Determine mode
    today = dt.datetime.now(dt.timezone.utc).date()
    is_live_mode = (target_date_obj == today)
    
    # Load API credentials from environment
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in .env file")
    
    # 3. Run scanner for the plan
    print("\n--- [Step 1] Running Historical Analysis to Build Plan ---")
    scanner = SynchronizedCashCarryScanner(
        api_key=api_key,
        api_secret=api_secret,
        lookback_days=lookback_days,
        basis_bps_min_buy=0.0,
    )
    # The analysis always ends on the morning of the target date
    analysis_result = scanner.run_synchronized_analysis(date_str, resample_15m=True)
    buy_leader = analysis_result["buy_leaderboard"]
    historical_avg_basis = analysis_result.get("historical_avg_basis", 200.0)
    
    if buy_leader.empty:
        print("\nNo historical buy opportunities found. Using uniform weights.")
        slot_weights = {k: 1.0/96.0 for k in range(96)}
    else:
        print("\nAnalysis complete. Generating execution weights.")
        slot_weights = derive_slot_weights(buy_leader)
    
    # 4. Prepare for execution (Live or Backtest)
    price_feed = None
    start_execution_dt = dt.datetime.now(dt.timezone.utc)

    if is_live_mode:
        print("\n--- [Step 2] Starting LIVE 24-Hour Simulation ---")
        print("This will run slowly and print updates periodically. (Ctrl+C to stop)")
        
        # Re-normalize the remaining weights for the rest of the day
        now = dt.datetime.now(dt.timezone.utc)
        current_slot = ((now.hour * 60) + now.minute) // 15
        
        print(f"Live mode started at {now.strftime('%H:%M:%S')} UTC, beginning at slot {current_slot}.")

        # Filter for remaining slots
        remaining_weights = {s: w for s, w in slot_weights.items() if s >= current_slot}
        
        if remaining_weights:
            # Re-normalize the remaining weights so they sum to 1.0
            sum_remaining = sum(remaining_weights.values())
            if sum_remaining > 1e-9:
                live_slot_weights = {s: w / sum_remaining for s, w in remaining_weights.items()}
                print(f"Execution plan re-normalized for the {96 - current_slot} remaining slots.")
                slot_weights = live_slot_weights
        
        # Adjust start time to the beginning of the current bucket for a clean start
        start_of_bucket = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        start_execution_dt = start_of_bucket

    else:
        print("\n--- [Step 2] Preparing Historical Backtest ---")
        print("Fetching 24h of 1-minute data for backtest...")
        
        # We need the 24h data for the target date
        backtest_start_dt = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        start_ms = int(backtest_start_dt.timestamp() * 1000)
        end_ms = start_ms + (24 * 60 * 60 * 1000)
        
        # Use a client to fetch data for the price feed
        temp_spot = Spot(api_key=api_key, api_secret=api_secret)
        temp_cm = CMFutures(key=api_key, secret=api_secret)
        
        spot_kl = scanner._fetch_klines_synchronized(temp_spot, "BTCUSDT", start_ms, end_ms)
        cur_kl = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", start_ms, end_ms, is_continuous=True, contract_type="CURRENT_QUARTER")
        nxt_kl = scanner._fetch_klines_synchronized(temp_cm, "BTCUSD", start_ms, end_ms, is_continuous=True, contract_type="NEXT_QUARTER")

        # Convert to DFs without prefixes for the feed
        spot_df_feed = scanner._klines_to_df(spot_kl)
        cur_df_feed = scanner._klines_to_df(cur_kl)
        nxt_df_feed = scanner._klines_to_df(nxt_kl)
        
        price_feed = BacktestPriceFeed(spot_df_feed, cur_df_feed, nxt_df_feed)
        start_execution_dt = backtest_start_dt
        print("Data ready. Starting backtest simulation...")

    # 5. Run executor
    sim = BinanceSimulator(api_key=api_key, secret_key=api_secret, order_fill_prob=0.90)
    
    executor = AdaptiveCashCarryExecutor(
        api_key=api_key,
        api_secret=api_secret,
        slot_weights=slot_weights,
        simulator=sim,
        price_feed=price_feed, # Pass the price_feed here
        historical_avg_basis=historical_avg_basis,
        params=ExecParams(
            capital_usdt=1_000_000.0,
            hours=24
        ),
    )

    result = executor.run(start_dt=start_execution_dt, realtime=is_live_mode)

    summary = result["summary"]
    exec_log_df = result["log"]

    print("\n\n=== EXECUTION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")

    print("\nExec log (full):")
    print(exec_log_df.to_string())


if __name__ == "__main__":
    main()
