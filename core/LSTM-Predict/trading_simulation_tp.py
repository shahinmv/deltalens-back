import numpy as np
import pandas as pd
import json
import sys, os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from main_ml import ImprovedBitcoinPredictor

# Import the required modules using relative imports
base_path = os.path.dirname(os.path.abspath(__file__))
dev_path = os.path.join(base_path, 'development')
if dev_path not in sys.path:
    sys.path.insert(0, dev_path)
if base_path not in sys.path:
    sys.path.insert(0, base_path)
# Now import
from data_loader import load_all_data
from sentiment import add_vader_sentiment, aggregate_daily_sentiment
from feature_engineering import engineer_features
from model import ImprovedBitcoinPredictor

class TradingSimulator:
    """
    Trading simulation functionality for model evaluation.
    """
    
    def __init__(self, predictor, min_acceptable_sharpe=0.5, max_acceptable_drawdown=0.2):
        self.predictor = predictor
        self.min_acceptable_sharpe = min_acceptable_sharpe
        self.max_acceptable_drawdown = max_acceptable_drawdown
    
    def run_trading_simulation(self, df, initial_capital=10000, transaction_cost=0.001):
        """
        Run only the trading simulation
        """
        print("="*60)
        print("TRADING SIMULATION")
        print("="*60)
        print(f"Data period: {df.index[0]} to {df.index[-1]}")
        print(f"Total days: {len(df)}")
        
        # Train the model
        print("Training model...")
        try:
            self.predictor.train_ensemble(df, validation_split=0.2, epochs=100, batch_size=32)
            print("✅ Model training completed successfully")
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            return None
        
        # Run trading simulation
        print("\nRunning Trading Simulation...")
        try:
            trading_results = self.simulate_trading(df, initial_capital=initial_capital, transaction_cost=transaction_cost)
            print("✅ Trading simulation completed successfully")
            return trading_results
        except Exception as e:
            print(f"❌ Trading simulation failed: {str(e)}")
            return None
    
    def simulate_trading(self, df, initial_capital=10000, transaction_cost=0.001):
        """
        Simulate realistic trading with transaction costs, position sizing, and stop losses
        """
        try:
            # Prepare data
            df_proc = self.predictor.engineer_30day_target(df)
            features, _ = self.predictor.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            
            X, y, _ = self.predictor.create_sequences(features, targets)
            
            if len(X) == 0:
                print("  Warning: No sequences created for trading simulation")
                return {
                    'error': 'No sequences created',
                    'profitable': False, 'meets_sharpe_threshold': False, 'meets_drawdown_threshold': False,
                    'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'n_trades': 0, 'win_rate': 0,
                    'initial_capital': initial_capital, 'final_capital': initial_capital, 
                    'annualized_return': 0, 'avg_trade_return': 0, 'trade_frequency': 0
                }
            
            # Split data
            split_idx = int(0.7 * len(X))
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            if len(X_test) == 0:
                print("  Warning: No test data for trading simulation")
                return {
                    'error': 'No test data',
                    'profitable': False, 'meets_sharpe_threshold': False, 'meets_drawdown_threshold': False,
                    'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'n_trades': 0, 'win_rate': 0,
                    'initial_capital': initial_capital, 'final_capital': initial_capital, 
                    'annualized_return': 0, 'avg_trade_return': 0, 'trade_frequency': 0
                }
            
            # Get predictions using pre-trained model
            ensemble_pred, _, _ = self.predictor.predict_ensemble(X_test)
            
            # Stop loss parameters
            stop_loss_pct = 0.05  # 5% stop loss
            
            def apply_stop_loss(predicted_return, actual_return, position_type):
                """Apply stop loss logic to limit downside"""
                if position_type == 'long' and actual_return < -stop_loss_pct:
                    return -stop_loss_pct  # Cap loss at stop loss level
                elif position_type == 'short' and actual_return > stop_loss_pct:
                    return -stop_loss_pct  # Cap loss for short position
                else:
                    return actual_return  # No stop triggered, use actual return
            
            # Simulate trading
            capital = initial_capital
            positions = []
            returns = []
            equity_curve = [capital]
            trades_log = []  # Track all individual trades
            
            # Get test data indices for timestamps and prices
            test_start_idx = split_idx
            test_dates = df_proc.index[test_start_idx:test_start_idx + len(ensemble_pred)]
            
            # Get BTC prices for the test period
            btc_prices_start = []
            btc_prices_end = []
            trade_start_dates = []
            trade_end_dates = []
            
            # Helper function to find when predicted price is reached
            def find_exit_point(start_date, start_price, predicted_return, max_days=30):
                """Find when predicted price target is reached or max days elapsed"""
                try:
                    start_pos = df.index.get_loc(start_date)
                    target_price = start_price * (1 + predicted_return)
                    
                    # Check each day up to max_days
                    for day_offset in range(1, min(max_days + 1, len(df) - start_pos)):
                        current_date = df.index[start_pos + day_offset]
                        current_price = df.loc[current_date, 'close'] if 'close' in df.columns else df.loc[current_date, 'Close']
                        
                        # Check if target is reached (within 1% tolerance)
                        if predicted_return > 0:  # Long position - check if price went up enough
                            if current_price >= target_price * 0.99:  # 1% tolerance
                                return current_date, current_price, day_offset, 'target_reached'
                        else:  # Short position - check if price went down enough
                            if current_price <= target_price * 1.01:  # 1% tolerance
                                return current_date, current_price, day_offset, 'target_reached'
                    
                    # If target not reached, exit at max_days
                    end_pos = min(start_pos + max_days, len(df) - 1)
                    end_date = df.index[end_pos]
                    end_price = df.loc[end_date, 'close'] if 'close' in df.columns else df.loc[end_date, 'Close']
                    return end_date, end_price, end_pos - start_pos, 'max_days_reached'
                    
                except (KeyError, IndexError):
                    return start_date, start_price, 0, 'error'

            # Store exit information for each trade
            exit_info = []
            
            for i in range(len(ensemble_pred)):
                # Calculate trade start and dynamic end dates
                if i < len(test_dates):
                    trade_start = test_dates[i]
                    try:
                        btc_start_price = df.loc[trade_start, 'close'] if 'close' in df.columns else df.loc[trade_start, 'Close']
                        # Get predicted return for this trade
                        pred_return = ensemble_pred[i][0]
                        
                        # Find optimal exit point
                        trade_end, btc_end_price, days_held, exit_reason = find_exit_point(
                            trade_start, btc_start_price, pred_return, max_days=30
                        )
                        exit_info.append({'days_held': days_held, 'exit_reason': exit_reason})
                    except (KeyError, IndexError):
                        trade_end = trade_start
                        btc_start_price = 0.0
                        btc_end_price = 0.0
                        exit_info.append({'days_held': 0, 'exit_reason': 'error'})
                else:
                    trade_start = f"test_day_{i}"
                    trade_end = f"test_day_{i+30}"
                    btc_start_price = 0.0
                    btc_end_price = 0.0
                    exit_info.append({'days_held': 30, 'exit_reason': 'test_data'})
                
                trade_start_dates.append(trade_start)
                trade_end_dates.append(trade_end)
                btc_prices_start.append(btc_start_price)
                btc_prices_end.append(btc_end_price)
            
            for i in range(len(ensemble_pred)):
                # Get prediction
                pred_return = ensemble_pred[i][0]
                actual_return = y_test[i]
                
                # Position sizing based on confidence (Kelly criterion approximation)
                confidence = min(abs(pred_return), 0.1)  # Cap at 10% position
                position_size = confidence
                
                # Calculate actual return based on exit point
                actual_btc_return = float((btc_prices_end[i] - btc_prices_start[i]) / btc_prices_start[i]) if btc_prices_start[i] > 0 else 0.0
                
                trade_info = {
                    'index': i,
                    'date': str(test_dates[i]) if i < len(test_dates) else f"test_{i}",
                    'trade_start_date': str(trade_start_dates[i]),
                    'trade_end_date': str(trade_end_dates[i]),
                    'btc_price_start': float(btc_prices_start[i]),
                    'btc_price_end': float(btc_prices_end[i]),
                    'btc_price_change': float(btc_prices_end[i] - btc_prices_start[i]) if btc_prices_start[i] > 0 else 0.0,
                    'btc_return_actual': actual_btc_return,
                    'predicted_return': float(pred_return),
                    'actual_return': actual_btc_return,  # Use actual BTC return instead of original y_test
                    'confidence': float(confidence),
                    'position_size': float(position_size),
                    'capital_before': float(capital),
                    'trade_type': 'no_trade',
                    'position_value': 0.0,
                    'trade_return': 0.0,
                    'transaction_cost_paid': 0.0,
                    'stop_loss_triggered': False,
                    'risk_adjusted_return': actual_btc_return,
                    'exit_reason': exit_info[i]['exit_reason'],
                    'days_held': exit_info[i]['days_held'],
                    'target_reached': exit_info[i]['exit_reason'] == 'target_reached'
                }
                
                # Determine trade
                if abs(pred_return) > 0.02:  # Only trade if predicted return > 2%
                    if pred_return > 0:
                        # Long position
                        trade_info['trade_type'] = 'long'
                        position_value = capital * position_size
                        # Account for transaction costs
                        position_value *= (1 - transaction_cost)
                        
                        # Apply stop loss logic using actual BTC return
                        risk_adjusted_return = apply_stop_loss(pred_return, actual_btc_return, 'long')
                        trade_info['stop_loss_triggered'] = risk_adjusted_return != actual_btc_return
                        trade_info['risk_adjusted_return'] = float(risk_adjusted_return)
                        
                        # Calculate return with stop loss
                        trade_return = position_value * risk_adjusted_return
                        transaction_cost_paid = position_value * transaction_cost * 2  # Entry + exit
                        capital += trade_return - (position_value * transaction_cost)  # Exit cost
                        
                        trade_info['position_value'] = float(position_value)
                        trade_info['trade_return'] = float(trade_return)
                        trade_info['transaction_cost_paid'] = float(transaction_cost_paid)
                        
                    else:
                        # Short position
                        trade_info['trade_type'] = 'short'
                        position_value = capital * position_size
                        position_value *= (1 - transaction_cost)
                        
                        # Apply stop loss logic (negative actual return for short)
                        risk_adjusted_return = apply_stop_loss(pred_return, actual_btc_return, 'short')
                        trade_info['stop_loss_triggered'] = risk_adjusted_return != actual_btc_return
                        trade_info['risk_adjusted_return'] = float(risk_adjusted_return)
                        
                        trade_return = -position_value * risk_adjusted_return
                        transaction_cost_paid = position_value * transaction_cost * 2  # Entry + exit
                        capital += trade_return - (position_value * transaction_cost)
                        
                        trade_info['position_value'] = float(position_value)
                        trade_info['trade_return'] = float(trade_return)
                        trade_info['transaction_cost_paid'] = float(transaction_cost_paid)
                    
                    positions.append(np.sign(pred_return))
                    returns.append(trade_return / (capital - trade_return) if capital - trade_return != 0 else 0)
                else:
                    positions.append(0)
                    returns.append(0)
                
                trade_info['capital_after'] = float(capital)
                trade_info['profit_loss'] = trade_info['capital_after'] - trade_info['capital_before']
                
                # Only add actual trades to the log (exclude no_trade entries)
                if trade_info['trade_type'] != 'no_trade':
                    trades_log.append(trade_info)
                
                equity_curve.append(capital)
            
            # Calculate metrics
            total_return = (capital - initial_capital) / initial_capital
            returns_array = np.array(returns)
            
            # Remove zero returns for some metrics
            active_returns = returns_array[returns_array != 0]
            
            if len(active_returns) > 0:
                sharpe = np.mean(active_returns) / (np.std(active_returns) + 1e-6) * np.sqrt(252/30)
                win_rate = np.sum(active_returns > 0) / len(active_returns)
                avg_trade_return = np.mean(active_returns)
            else:
                sharpe = 0
                win_rate = 0.5
                avg_trade_return = 0
            
            # Drawdown calculation
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Trade statistics
            n_trades = np.sum(np.array(positions) != 0)
            
            results = {
                'simulation_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'initial_capital': initial_capital,
                    'transaction_cost': transaction_cost,
                    'stop_loss_pct': stop_loss_pct,
                    'min_trade_threshold': 0.02,
                    'test_period_start': str(test_dates[0]) if len(test_dates) > 0 else 'unknown',
                    'test_period_end': str(test_dates[-1]) if len(test_dates) > 0 else 'unknown',
                    'total_test_days': len(ensemble_pred)
                },
                'performance_metrics': {
                    'initial_capital': initial_capital,
                    'final_capital': capital,
                    'total_return': total_return,
                    'annualized_return': total_return * 252/30 / len(y_test) if len(y_test) > 0 else 0,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_drawdown,
                    'n_trades': n_trades,
                    'win_rate': win_rate,
                    'avg_trade_return': avg_trade_return,
                    'trade_frequency': n_trades / len(y_test) if len(y_test) > 0 else 0,
                    'profitable': capital > initial_capital,
                    'meets_sharpe_threshold': sharpe > self.min_acceptable_sharpe,
                    'meets_drawdown_threshold': abs(max_drawdown) < self.max_acceptable_drawdown
                },
                'trades': trades_log,
                'equity_curve': [float(x) for x in equity_curve]
            }
            
            return results
            
        except Exception as e:
            print(f"  Error in trading simulation: {str(e)}")
            return {
                'error': str(e),
                'profitable': False, 'meets_sharpe_threshold': False, 'meets_drawdown_threshold': False,
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'n_trades': 0, 'win_rate': 0,
                'initial_capital': initial_capital, 'final_capital': initial_capital, 
                'annualized_return': 0, 'avg_trade_return': 0, 'trade_frequency': 0,
                'trades': [], 'equity_curve': []
            }
    
    def save_results_to_json(self, results, filename=None):
        """
        Save trading simulation results to JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'trading_simulation_results_{timestamp}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving results to JSON: {str(e)}")
            return None

# Example usage:
if __name__ == "__main__":
    # Load data
    btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data()
    
    # Process sentiment data
    df_news = add_vader_sentiment(df_news)
    df_newsdaily_sentiment = aggregate_daily_sentiment(df_news)
    
    # Feature engineering
    df = engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment)
    
    # Initialize predictor
    improved_predictor = ImprovedBitcoinPredictor(
        sequence_length=60,
        prediction_horizon=30,
    )
    
    # Create trading simulator
    simulator = TradingSimulator(improved_predictor)
    
    # Run trading simulation
    results = simulator.run_trading_simulation(df, initial_capital=10000, transaction_cost=0.001)
    
    if results and 'performance_metrics' in results:
        # Save results to JSON
        filename = simulator.save_results_to_json(results)
        
        # Display results
        metrics = results['performance_metrics']
        metadata = results['simulation_metadata']
        
        print("\n" + "="*60)
        print("TRADING SIMULATION RESULTS")
        print("="*60)
        print(f"Test Period: {metadata['test_period_start']} to {metadata['test_period_end']}")
        print(f"Total Test Days: {metadata['total_test_days']}")
        print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.3f}")
        print(f"Number of Trades: {metrics['n_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.3f}")
        print(f"Average Trade Return: {metrics['avg_trade_return']:.4f}")
        print(f"Trade Frequency: {metrics['trade_frequency']:.3f}")
        print(f"Profitable: {'Yes' if metrics['profitable'] else 'No'}")
        print(f"Meets Sharpe Threshold: {'Yes' if metrics['meets_sharpe_threshold'] else 'No'}")
        print(f"Meets Drawdown Threshold: {'Yes' if metrics['meets_drawdown_threshold'] else 'No'}")
        
        # Show trade statistics
        trades = results['trades']
        active_trades = [t for t in trades if t['trade_type'] != 'no_trade']
        long_trades = [t for t in active_trades if t['trade_type'] == 'long']
        short_trades = [t for t in active_trades if t['trade_type'] == 'short']
        winning_trades = [t for t in active_trades if t['profit_loss'] > 0]
        stop_loss_trades = [t for t in active_trades if t['stop_loss_triggered']]
        
        print(f"\nTRADE BREAKDOWN:")
        print(f"Total Periods Analyzed: {len(trades)}")
        print(f"Active Trades: {len(active_trades)}")
        print(f"Long Trades: {len(long_trades)}")
        print(f"Short Trades: {len(short_trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Stop Loss Triggered: {len(stop_loss_trades)}")
        
        # Show exit strategy statistics
        target_reached_trades = [t for t in active_trades if t.get('target_reached', False)]
        avg_days_held = np.mean([t.get('days_held', 30) for t in active_trades]) if active_trades else 0
        
        print(f"\nEXIT STRATEGY PERFORMANCE:")
        print(f"Target Reached Early: {len(target_reached_trades)}/{len(active_trades)} trades")
        print(f"Average Days Held: {avg_days_held:.1f} days")
        print(f"Early Exit Success Rate: {len(target_reached_trades)/len(active_trades)*100:.1f}%" if active_trades else "N/A")
        
        if filename:
            print(f"\n📄 Detailed results with all trades saved to: {filename}")
    else:
        print("Trading simulation failed!")