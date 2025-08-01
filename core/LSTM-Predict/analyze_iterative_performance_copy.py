import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_simulation.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class Position:
    """Represents an active trading position"""
    entry_date: str
    signal_type: str  # 'LONG' or 'SHORT'
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float  # Dollar amount invested
    shares: float  # Number of shares/units
    predicted_return: float
    confidence: float
    dynamic_stop_pct: float
    days_held: int = 0
    max_days: int = 30

@dataclass
class Trade:
    """Represents a completed trade"""
    entry_date: str
    exit_date: str
    signal_type: str
    entry_price: float
    exit_price: float
    target_price: float
    stop_loss: float
    position_size: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'TP', 'SL', 'TIMEOUT'
    days_held: int
    predicted_return: float
    confidence: float
    dynamic_stop_pct: float

class TradingSimulator:
    def __init__(self, db_path: str = "../../db.sqlite3", initial_balance: float = 10000.0):
        self.db_path = db_path
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: List[Position] = []
        self.completed_trades: List[Trade] = []
        self.daily_balance: List[Dict] = []
        
        # Trading parameters
        self.transaction_cost_pct = 0.001  # 0.1% transaction cost
        self.max_position_pct = 0.3  # Maximum 30% of balance per position
        
        # Load data
        self.signals_df = None
        self.price_data = None
        self.load_data()
    
    def load_data(self):
        """Load trading signals and price data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load trading signals
            signals_query = """
                SELECT * FROM iterative_trading_signals 
                WHERE signal_type != 'HOLD'
                ORDER BY prediction_date ASC
            """
            self.signals_df = pd.read_sql_query(signals_query, conn)
            
            if len(self.signals_df) == 0:
                logging.error("No trading signals found in database")
                return
            
            logging.info(f"Loaded {len(self.signals_df)} trading signals")
            
            # Load price data
            price_query = """
                SELECT * FROM btc_daily_ohlcv 
                ORDER BY datetime ASC
            """
            self.price_data = pd.read_sql_query(price_query, conn)
            self.price_data.set_index('datetime', inplace=True)
            
            logging.info(f"Loaded {len(self.price_data)} price records")
            
            conn.close()
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
    
    def get_price_data_for_date(self, date: str) -> Optional[pd.Series]:
        """Get OHLCV data for a specific date"""
        try:
            if date in self.price_data.index:
                return self.price_data.loc[date]
            else:
                # Find the closest available date
                available_dates = pd.to_datetime(self.price_data.index)
                target_date = pd.to_datetime(date)
                closest_date_idx = np.argmin(np.abs(available_dates - target_date))
                closest_date = self.price_data.index[closest_date_idx]
                logging.warning(f"Date {date} not found, using closest date {closest_date}")
                return self.price_data.loc[closest_date]
        except Exception as e:
            logging.error(f"Error getting price data for {date}: {e}")
            return None
    
    def check_position_exit(self, position: Position, current_data: pd.Series) -> Tuple[bool, str, float]:
        """
        Check if position should be exited based on TP/SL or timeout
        Returns: (should_exit, exit_reason, exit_price)
        """
        try:
            high = current_data['high']
            low = current_data['low']
            close = current_data['close']
            
            if position.signal_type == 'LONG':
                # Check for take profit (high reached target)
                if high >= position.target_price:
                    return True, 'TP', position.target_price
                
                # Check for stop loss (low hit stop loss)
                if low <= position.stop_loss:
                    return True, 'SL', position.stop_loss
            
            elif position.signal_type == 'SHORT':
                # Check for take profit (low reached target)
                if low <= position.target_price:
                    return True, 'TP', position.target_price
                
                # Check for stop loss (high hit stop loss)
                if high >= position.stop_loss:
                    return True, 'SL', position.stop_loss
            
            # Check for timeout (30 days)
            if position.days_held >= position.max_days:
                return True, 'TIMEOUT', close
            
            return False, '', 0.0
            
        except Exception as e:
            logging.error(f"Error checking position exit: {e}")
            return False, '', 0.0
    
    def calculate_position_size(self, signal_row: pd.Series) -> float:
        """Calculate position size based on signal and available balance"""
        try:
            # Get suggested position size from signal (as fraction)
            suggested_size = signal_row['position_size']
            
            # Apply maximum position limit
            max_position_size = self.current_balance * self.max_position_pct
            suggested_dollar_amount = self.current_balance * suggested_size
            
            # Use the smaller of suggested size or max position size
            position_size = min(suggested_dollar_amount, max_position_size)
            
            # Ensure we have enough balance (including transaction costs)
            required_balance = position_size * (1 + self.transaction_cost_pct)
            
            if required_balance > self.current_balance:
                position_size = self.current_balance / (1 + self.transaction_cost_pct)
            
            return max(0, position_size)
            
        except Exception as e:
            logging.error(f"Error calculating position size: {e}")
            return 0.0
    
    def open_position(self, signal_row: pd.Series, entry_date: str) -> bool:
        """Open a new position based on signal"""
        try:
            # Calculate position size
            position_size = self.calculate_position_size(signal_row)
            
            if position_size <= 0:
                logging.warning(f"Insufficient balance for position on {entry_date}")
                return False
            
            # Get entry price (use next day's open price for realistic simulation)
            next_date = (pd.to_datetime(entry_date) + timedelta(days=1)).strftime('%Y-%m-%d')
            entry_price_data = self.get_price_data_for_date(next_date)
            
            if entry_price_data is None:
                logging.warning(f"No price data available for entry on {next_date}")
                return False
            
            entry_price = entry_price_data['open']  # Use opening price of next day
            
            # Calculate shares
            transaction_cost = position_size * self.transaction_cost_pct
            net_position_size = position_size - transaction_cost
            shares = net_position_size / entry_price
            
            # Create position
            position = Position(
                entry_date=entry_date,
                signal_type=signal_row['signal_type'],
                entry_price=entry_price,
                target_price=signal_row['target_price'],
                stop_loss=signal_row['stop_loss'],
                position_size=position_size,
                shares=shares,
                predicted_return=signal_row['predicted_return'],
                confidence=signal_row['confidence'],
                dynamic_stop_pct=signal_row['dynamic_stop_pct']
            )
            
            # Update balance
            self.current_balance -= position_size
            self.positions.append(position)
            
            logging.info(f"Opened {signal_row['signal_type']} position on {entry_date}: "
                        f"${position_size:.2f} at ${entry_price:.2f}, target: ${signal_row['target_price']:.2f}, "
                        f"stop: ${signal_row['stop_loss']:.2f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error opening position: {e}")
            return False
    
    def close_position(self, position: Position, exit_date: str, exit_reason: str, exit_price: float) -> Trade:
        """Close a position and calculate P&L"""
        try:
            # Calculate P&L
            if position.signal_type == 'LONG':
                gross_pnl = (exit_price - position.entry_price) * position.shares
            else:  # SHORT
                gross_pnl = (position.entry_price - exit_price) * position.shares
            
            # Account for exit transaction cost
            exit_transaction_cost = (exit_price * position.shares) * self.transaction_cost_pct
            net_pnl = gross_pnl - exit_transaction_cost
            
            # Calculate percentage return
            pnl_pct = net_pnl / position.position_size
            
            # Return proceeds to balance
            proceeds = position.position_size + net_pnl
            self.current_balance += proceeds
            
            # Create trade record
            trade = Trade(
                entry_date=position.entry_date,
                exit_date=exit_date,
                signal_type=position.signal_type,
                entry_price=position.entry_price,
                exit_price=exit_price,
                target_price=position.target_price,
                stop_loss=position.stop_loss,
                position_size=position.position_size,
                shares=position.shares,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                days_held=position.days_held,
                predicted_return=position.predicted_return,
                confidence=position.confidence,
                dynamic_stop_pct=position.dynamic_stop_pct
            )
            
            self.completed_trades.append(trade)
            
            logging.info(f"Closed {position.signal_type} position: "
                        f"P&L: ${net_pnl:.2f} ({pnl_pct*100:.2f}%), "
                        f"Exit reason: {exit_reason}, Days held: {position.days_held}")
            
            return trade
            
        except Exception as e:
            logging.error(f"Error closing position: {e}")
            return None
    
    def simulate_trading(self):
        """Run the complete trading simulation"""
        logging.info(f"Starting trading simulation with ${self.initial_balance:,.2f}")
        logging.info(f"Processing {len(self.signals_df)} signals...")
        
        # Get all unique dates for simulation
        start_date = self.signals_df['prediction_date'].min()
        end_date = max(self.signals_df['prediction_date'].max(), 
                      (pd.to_datetime(self.signals_df['prediction_date'].max()) + timedelta(days=30)).strftime('%Y-%m-%d'))
        
        logging.info(f"Simulation period: {start_date} to {end_date}")
        
        # Create date range for daily processing
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for current_date in date_range:
            current_date_str = current_date.strftime('%Y-%m-%d')
            
            # Get current day's price data
            current_price_data = self.get_price_data_for_date(current_date_str)
            if current_price_data is None:
                continue
            
            # Process new signals for this date
            new_signals = self.signals_df[self.signals_df['prediction_date'] == current_date_str]
            
            for _, signal_row in new_signals.iterrows():
                self.open_position(signal_row, current_date_str)
            
            # Update existing positions and check for exits
            positions_to_remove = []
            
            for i, position in enumerate(self.positions):
                position.days_held += 1
                
                # Check if position should be closed
                should_exit, exit_reason, exit_price = self.check_position_exit(position, current_price_data)
                
                if should_exit:
                    trade = self.close_position(position, current_date_str, exit_reason, exit_price)
                    positions_to_remove.append(i)
            
            # Remove closed positions
            for i in reversed(positions_to_remove):
                self.positions.pop(i)
            
            # Calculate current portfolio value
            portfolio_value = self.current_balance
            for position in self.positions:
                current_price = current_price_data['close']
                if position.signal_type == 'LONG':
                    position_value = position.shares * current_price
                else:  # SHORT
                    position_value = position.position_size + (position.entry_price - current_price) * position.shares
                portfolio_value += position_value
            
            # Record daily balance
            self.daily_balance.append({
                'date': current_date_str,
                'cash_balance': self.current_balance,
                'portfolio_value': portfolio_value,
                'active_positions': len(self.positions)
            })
        
        # Close any remaining positions at the end
        if self.positions:
            final_date = date_range[-1].strftime('%Y-%m-%d')
            final_price_data = self.get_price_data_for_date(final_date)
            
            for position in self.positions:
                self.close_position(position, final_date, 'END_OF_SIMULATION', final_price_data['close'])
            
            self.positions.clear()
        
        logging.info("Trading simulation completed!")
        self.print_performance_summary()
    
    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        if not self.completed_trades:
            logging.info("No completed trades to analyze")
            return
        
        # Calculate performance metrics
        trades_df = pd.DataFrame([
            {
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'signal_type': trade.signal_type,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'exit_reason': trade.exit_reason,
                'days_held': trade.days_held,
                'predicted_return': trade.predicted_return,
                'confidence': trade.confidence,
                'dynamic_stop_pct': trade.dynamic_stop_pct
            }
            for trade in self.completed_trades
        ])
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        final_balance = self.current_balance
        total_return = (final_balance / self.initial_balance - 1) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        returns = trades_df['pnl_pct'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown()
        
        # Performance by exit reason
        exit_reasons = trades_df['exit_reason'].value_counts()
        
        print("\n" + "="*80)
        print("TRADING SIMULATION PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${final_balance:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total P&L: ${total_pnl:,.2f}")
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.1%})")
        print(f"Losing Trades: {losing_trades}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else 'N/A'}")
        
        print(f"\n⚡ RISK METRICS:")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Average Days Held: {trades_df['days_held'].mean():.1f}")
        
        print(f"\n🎯 EXIT REASONS:")
        for reason, count in exit_reasons.items():
            pct = count / total_trades * 100
            avg_pnl = trades_df[trades_df['exit_reason'] == reason]['pnl'].mean()
            print(f"{reason}: {count} trades ({pct:.1f}%) - Avg P&L: ${avg_pnl:.2f}")
        
        print(f"\n📊 SIGNAL TYPE PERFORMANCE:")
        for signal_type in ['LONG', 'SHORT']:
            type_trades = trades_df[trades_df['signal_type'] == signal_type]
            if len(type_trades) > 0:
                type_pnl = type_trades['pnl'].sum()
                type_win_rate = len(type_trades[type_trades['pnl'] > 0]) / len(type_trades)
                print(f"{signal_type}: {len(type_trades)} trades, ${type_pnl:.2f} P&L, {type_win_rate:.1%} win rate")
        
        # Top and bottom trades
        best_trades = trades_df.nlargest(5, 'pnl')[['entry_date', 'signal_type', 'pnl', 'pnl_pct', 'exit_reason']]
        worst_trades = trades_df.nsmallest(5, 'pnl')[['entry_date', 'signal_type', 'pnl', 'pnl_pct', 'exit_reason']]
        
        print(f"\n🏆 TOP 5 TRADES:")
        for _, trade in best_trades.iterrows():
            print(f"  {trade['entry_date']}: {trade['signal_type']} - ${trade['pnl']:.2f} ({trade['pnl_pct']:.1%}) [{trade['exit_reason']}]")
        
        print(f"\n💸 BOTTOM 5 TRADES:")
        for _, trade in worst_trades.iterrows():
            print(f"  {trade['entry_date']}: {trade['signal_type']} - ${trade['pnl']:.2f} ({trade['pnl_pct']:.1%}) [{trade['exit_reason']}]")
        
        print("\n" + "="*80)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from daily balance data"""
        if not self.daily_balance:
            return 0.0
        
        balances = [day['portfolio_value'] for day in self.daily_balance]
        peak = balances[0]
        max_dd = 0.0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def save_results_to_database(self):
        """Save simulation results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Save completed trades
            trades_data = []
            for trade in self.completed_trades:
                trades_data.append({
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'signal_type': trade.signal_type,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'target_price': trade.target_price,
                    'stop_loss': trade.stop_loss,
                    'position_size': trade.position_size,
                    'shares': trade.shares,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'exit_reason': trade.exit_reason,
                    'days_held': trade.days_held,
                    'predicted_return': trade.predicted_return,
                    'confidence': trade.confidence,
                    'dynamic_stop_pct': trade.dynamic_stop_pct
                })
            
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_sql('simulation_trades', conn, if_exists='replace', index=False)
            
            # Save daily balance data
            daily_df = pd.DataFrame(self.daily_balance)
            daily_df.to_sql('simulation_daily_balance', conn, if_exists='replace', index=False)
            
            conn.close()
            logging.info("Simulation results saved to database")
            
        except Exception as e:
            logging.error(f"Error saving results to database: {e}")
    
    def plot_performance(self):
        """Create performance visualization plots"""
        if not self.daily_balance or not self.completed_trades:
            logging.warning("No data available for plotting")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Portfolio value over time
        daily_df = pd.DataFrame(self.daily_balance)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        ax1.plot(daily_df['date'], daily_df['portfolio_value'], linewidth=2, color='blue')
        ax1.axhline(y=self.initial_balance, color='red', linestyle='--', alpha=0.7, label='Initial Balance')
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative returns
        daily_df['returns'] = daily_df['portfolio_value'].pct_change().fillna(0)
        daily_df['cumulative_returns'] = (1 + daily_df['returns']).cumprod() - 1
        
        ax2.plot(daily_df['date'], daily_df['cumulative_returns'] * 100, linewidth=2, color='green')
        ax2.set_title('Cumulative Returns (%)')
        ax2.set_ylabel('Returns (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Trade P&L distribution
        trades_df = pd.DataFrame([{'pnl': trade.pnl} for trade in self.completed_trades])
        ax3.hist(trades_df['pnl'], bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Trade P&L Distribution')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Frequency')
        
        # 4. Win/Loss by exit reason
        trades_summary = pd.DataFrame([
            {
                'exit_reason': trade.exit_reason,
                'pnl': trade.pnl
            }
            for trade in self.completed_trades
        ])
        
        exit_reason_pnl = trades_summary.groupby('exit_reason')['pnl'].sum()
        colors = ['green' if pnl > 0 else 'red' for pnl in exit_reason_pnl.values]
        
        ax4.bar(exit_reason_pnl.index, exit_reason_pnl.values, color=colors, alpha=0.7)
        ax4.set_title('P&L by Exit Reason')
        ax4.set_ylabel('Total P&L ($)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('trading_simulation_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logging.info("Performance plots saved as 'trading_simulation_performance.png'")

def main():
    """Main function to run the trading simulation"""
    # Initialize simulator
    simulator = TradingSimulator(initial_balance=10000.0)
    
    if simulator.signals_df is None or simulator.price_data is None:
        logging.error("Failed to load required data. Exiting.")
        return
    
    # Run simulation
    simulator.simulate_trading()
    
    # Save results
    simulator.save_results_to_database()
    
    # Create visualizations
    try:
        simulator.plot_performance()
    except Exception as e:
        logging.error(f"Error creating plots: {e}")

if __name__ == "__main__":
    main()