"""
Visualization script for Bitcoin Cash-and-Carry Backtest Results
Generates comprehensive charts from CSV logs
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for clean, professional charts
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_data():
    """Load all CSV files"""
    print("Loading data files...")
    
    # Daily PnL tracker
    daily = pd.read_csv('daily_pnl_tracker.csv')
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Entry execution log
    entry = pd.read_csv('entry_day_exec_log.csv')
    entry['ts'] = pd.to_datetime(entry['ts'])
    
    # PnL tracker (realized events)
    pnl = pd.read_csv('pnl_tracker.csv')
    pnl['timestamp'] = pd.to_datetime(pnl['timestamp'])
    
    # Rollover log
    try:
        rollover = pd.read_csv('rollover_exec_log.csv')
        rollover['ts'] = pd.to_datetime(rollover['ts'])
    except:
        rollover = None
    
    # Rehedge log
    try:
        rehedge = pd.read_csv('rehedge_log.csv')
        rehedge['timestamp'] = pd.to_datetime(rehedge['timestamp'])
    except:
        rehedge = None
    
    print(f"✓ Loaded daily PnL: {len(daily)} days")
    print(f"✓ Loaded entry log: {len(entry)} trades")
    print(f"✓ Loaded PnL events: {len(pnl)} events")
    if rollover is not None:
        print(f"✓ Loaded rollover log: {len(rollover)} trades")
    if rehedge is not None:
        print(f"✓ Loaded rehedge log: {len(rehedge)} trades")
    
    return daily, entry, pnl, rollover, rehedge

def create_main_dashboard(daily, entry, pnl, rollover):
    """Create main dashboard with 4 key charts"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Portfolio Value Over Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(daily['date'], daily['portfolio_value_usd'], 
             linewidth=2, color='#2E86AB', label='Portfolio Value')
    ax1.axhline(y=1000000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.fill_between(daily['date'], 1000000, daily['portfolio_value_usd'], 
                     where=(daily['portfolio_value_usd'] >= 1000000),
                     alpha=0.2, color='green', label='Profit')
    ax1.fill_between(daily['date'], 1000000, daily['portfolio_value_usd'], 
                     where=(daily['portfolio_value_usd'] < 1000000),
                     alpha=0.2, color='red', label='Loss')
    
    # Annotate key points
    max_idx = daily['portfolio_value_usd'].idxmax()
    min_idx = daily['portfolio_value_usd'].idxmin()
    ax1.scatter(daily.loc[max_idx, 'date'], daily.loc[max_idx, 'portfolio_value_usd'], 
               color='green', s=100, zorder=5)
    ax1.annotate(f"Peak: ${daily.loc[max_idx, 'portfolio_value_usd']:,.0f}\n{daily.loc[max_idx, 'date'].strftime('%Y-%m-%d')}", 
                xy=(daily.loc[max_idx, 'date'], daily.loc[max_idx, 'portfolio_value_usd']),
                xytext=(10, 20), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
    
    # 2. Cumulative PnL
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(daily['date'], daily['cumulative_pnl_usd'], 
             linewidth=2.5, color='#A23B72')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    ax2.fill_between(daily['date'], 0, daily['cumulative_pnl_usd'], 
                     alpha=0.3, color='#A23B72')
    
    final_pnl = daily['cumulative_pnl_usd'].iloc[-1]
    final_return = (final_pnl / 1000000) * 100
    days_held = len(daily)
    
    ax2.text(0.02, 0.98, f'Final PnL: ${final_pnl:,.0f}\nReturn: {final_return:.2f}%\nDays: {days_held}',
            transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    ax2.set_title('Cumulative Profit & Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative PnL (USD)')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    
    # 3. PnL Components Breakdown
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(daily['date'], daily['spot_mtm_pnl_usd'], 
             linewidth=2, label='Spot MTM', color='#F18F01', alpha=0.8)
    ax3.plot(daily['date'], daily['futures_unrealized_pnl_usd'], 
             linewidth=2, label='Futures Unrealized', color='#C73E1D', alpha=0.8)
    ax3.plot(daily['date'], daily['futures_realized_pnl_usd'], 
             linewidth=2, label='Futures Realized', color='#6A994E', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
    
    ax3.set_title('PnL Components Over Time', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('PnL (USD)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    
    # 4. Daily PnL Distribution
    ax4 = fig.add_subplot(gs[2, :])
    colors = ['green' if x > 0 else 'red' for x in daily['daily_pnl_usd']]
    ax4.bar(daily['date'], daily['daily_pnl_usd'], color=colors, alpha=0.6, width=1)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    avg_daily = daily['daily_pnl_usd'].mean()
    ax4.axhline(y=avg_daily, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, 
               label=f'Avg Daily PnL: ${avg_daily:,.0f}')
    
    ax4.set_title('Daily PnL Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Daily PnL (USD)')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    
    plt.suptitle('Bitcoin Cash-and-Carry Backtest Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    return fig

def create_entry_analysis(entry):
    """Analyze entry execution"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Entry Day Execution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Execution timeline
    ax1 = axes[0, 0]
    ax1.scatter(entry['ts'], entry['spot_px'], c=entry['basis_bps'], 
               cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_title('Entry Prices Over Time (colored by basis)')
    ax1.set_xlabel('Time (UTC)')
    ax1.set_ylabel('Spot Price (USD)')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    cbar = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar.set_label('Basis (bps)', rotation=270, labelpad=20)
    
    # 2. Cumulative execution
    ax2 = axes[0, 1]
    entry['cumulative_btc'] = entry['fill_spot_btc'].cumsum()
    entry['cumulative_contracts'] = entry['fill_fut_ct'].cumsum()
    
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(entry['ts'], entry['cumulative_btc'], 
                     linewidth=2.5, color='#2E86AB', label='Cumulative BTC')
    line2 = ax2_twin.plot(entry['ts'], entry['cumulative_contracts'], 
                          linewidth=2.5, color='#F18F01', label='Cumulative Contracts', linestyle='--')
    
    ax2.set_title('Cumulative Execution Progress')
    ax2.set_xlabel('Time (UTC)')
    ax2.set_ylabel('BTC Filled', color='#2E86AB')
    ax2_twin.set_ylabel('Futures Contracts Filled', color='#F18F01')
    ax2.tick_params(axis='y', labelcolor='#2E86AB')
    ax2_twin.tick_params(axis='y', labelcolor='#F18F01')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    # 3. Basis evolution
    ax3 = axes[1, 0]
    ax3.plot(entry['ts'], entry['basis_bps'], linewidth=2, color='#6A994E', marker='o', markersize=4)
    ax3.axhline(y=entry['basis_bps'].mean(), color='red', linestyle='--', 
               alpha=0.7, label=f"Avg: {entry['basis_bps'].mean():.1f} bps")
    ax3.fill_between(entry['ts'], entry['basis_bps'].min(), entry['basis_bps'], alpha=0.3)
    
    ax3.set_title('Basis Spread During Entry')
    ax3.set_xlabel('Time (UTC)')
    ax3.set_ylabel('Basis (bps)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 4. Execution summary stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_btc = entry['fill_spot_btc'].sum()
    total_contracts = entry['fill_fut_ct'].sum()
    avg_spot = (entry['spot_px'] * entry['fill_spot_btc']).sum() / total_btc
    avg_basis = entry['basis_bps'].mean()
    execution_time = (entry['ts'].max() - entry['ts'].min()).total_seconds() / 3600
    
    stats_text = f"""
    ENTRY EXECUTION SUMMARY
    ══════════════════════════════
    
    Total BTC Acquired:      {total_btc:.4f} BTC
    Total Contracts Shorted: {total_contracts:,} contracts
    
    Avg Spot Price:          ${avg_spot:,.2f}
    Avg Basis:               {avg_basis:.1f} bps
    Annualized Basis:        {avg_basis * 365 / 90:.1f}% (est)
    
    Execution Time:          {execution_time:.1f} hours
    Number of Slices:        {len(entry)} fills
    
    Price Range:             ${entry['spot_px'].min():,.0f} - ${entry['spot_px'].max():,.0f}
    Basis Range:             {entry['basis_bps'].min():.1f} - {entry['basis_bps'].max():.1f} bps
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_rollover_analysis(rollover, pnl):
    """Analyze rollover events"""
    if rollover is None or len(rollover) == 0:
        print("No rollover data available")
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Futures Contract Rollovers Analysis', fontsize=16, fontweight='bold')
    
    # Identify rollover dates (group by date)
    rollover['date'] = rollover['ts'].dt.date
    rollover_dates = rollover.groupby('date').size()
    
    # 1. Rollover timeline
    ax1 = axes[0]
    
    # Separate BUY (close old) and SELL (open new) trades
    buy_trades = rollover[rollover['side'] == 'BUY']
    sell_trades = rollover[rollover['side'] == 'SELL']
    
    ax1.scatter(buy_trades['ts'], buy_trades['price'], 
               label='Close Old Contract (BUY)', color='red', s=50, alpha=0.6, marker='v')
    ax1.scatter(sell_trades['ts'], sell_trades['price'], 
               label='Open New Contract (SELL)', color='green', s=50, alpha=0.6, marker='^')
    
    ax1.set_title('Rollover Execution Prices')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Futures Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Realized PnL from rollovers
    ax2 = axes[1]
    
    # Filter PnL events to show rollover-related realized PnL
    pnl['date'] = pnl['timestamp'].dt.date
    daily_realized = pnl.groupby('date')['pnl_usd'].sum().reset_index()
    daily_realized['date'] = pd.to_datetime(daily_realized['date'])
    
    colors = ['green' if x > 0 else 'red' for x in daily_realized['pnl_usd']]
    ax2.bar(daily_realized['date'], daily_realized['pnl_usd'], 
           color=colors, alpha=0.6, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Annotate major rollover events
    major_pnl = daily_realized[abs(daily_realized['pnl_usd']) > 5000]
    for _, row in major_pnl.iterrows():
        ax2.annotate(f"${row['pnl_usd']:,.0f}", 
                    xy=(row['date'], row['pnl_usd']),
                    xytext=(0, 10 if row['pnl_usd'] > 0 else -20), 
                    textcoords='offset points',
                    fontsize=8, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    ax2.set_title('Daily Realized PnL (Rollovers & Rehedges)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Realized PnL (USD)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    
    plt.tight_layout()
    return fig

def main():
    """Generate all visualizations"""
    print("\n" + "="*60)
    print("Bitcoin Cash-and-Carry Backtest Visualization")
    print("="*60 + "\n")
    
    # Load data
    daily, entry, pnl, rollover, rehedge = load_data()
    
    print("\nGenerating visualizations...")
    
    # 1. Main dashboard
    print("  → Creating main dashboard...")
    fig1 = create_main_dashboard(daily, entry, pnl, rollover)
    fig1.savefig('backtest_dashboard.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: backtest_dashboard.png")
    
    # 2. Entry analysis
    print("  → Creating entry analysis...")
    fig2 = create_entry_analysis(entry)
    fig2.savefig('entry_analysis.png', dpi=300, bbox_inches='tight')
    print("    ✓ Saved: entry_analysis.png")
    
    # 3. Rollover analysis
    print("  → Creating rollover analysis...")
    fig3 = create_rollover_analysis(rollover, pnl)
    if fig3:
        fig3.savefig('rollover_analysis.png', dpi=300, bbox_inches='tight')
        print("    ✓ Saved: rollover_analysis.png")
    
    print("\n" + "="*60)
    print("✓ All visualizations generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  • backtest_dashboard.png  - Main portfolio performance dashboard")
    print("  • entry_analysis.png      - Entry day execution details")
    print("  • rollover_analysis.png   - Futures rollover events")
    print("\n")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()


