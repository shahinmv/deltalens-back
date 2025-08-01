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

class DynamicStopLossCalculator:
    """
    Advanced dynamic stop loss calculator using multiple market factors
    """
    
    def __init__(self, df, atr_period=14, volatility_lookback=30):
        self.df = df
        self.atr_period = atr_period
        self.volatility_lookback = volatility_lookback
        self.price_col = 'close' if 'close' in df.columns else 'Close'
        self.high_col = 'high' if 'high' in df.columns else 'High'
        self.low_col = 'low' if 'low' in df.columns else 'Low'
        
        # Pre-calculate technical indicators
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        """Pre-calculate all technical indicators needed for dynamic stop loss"""
        # Calculate True Range and ATR
        high = self.df[self.high_col]
        low = self.df[self.low_col]
        close = self.df[self.price_col]
        prev_close = close.shift(1)
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.DataFrame([tr1, tr2, tr3]).max()
        
        # Average True Range
        self.df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # Historical volatility (standard deviation of returns)
        returns = close.pct_change()
        self.df['volatility'] = returns.rolling(window=self.volatility_lookback).std() * np.sqrt(252)  # Annualized
        
        # Market regime indicators
        # Trend strength using ADX-like calculation
        self.df['trend_strength'] = self._calculate_trend_strength()
        
        # Support and resistance levels
        self.df['resistance'] = high.rolling(window=20).max()
        self.df['support'] = low.rolling(window=20).min()
        
        # Market momentum (RSI-like)
        self.df['momentum'] = self._calculate_momentum()
        
        # VIX-like fear index for crypto (using rolling volatility)
        self.df['fear_index'] = self._calculate_fear_index()
    
    def _calculate_trend_strength(self):
        """Calculate trend strength similar to ADX"""
        high = self.df[self.high_col]
        low = self.df[self.low_col]
        close = self.df[self.price_col]
        
        # Directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the directional movements
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=14).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=14).mean()
        
        # Calculate trend strength
        trend_strength = abs(plus_dm_smooth - minus_dm_smooth) / (plus_dm_smooth + minus_dm_smooth + 1e-6)
        return trend_strength.fillna(0)
    
    def _calculate_momentum(self):
        """Calculate momentum indicator"""
        close = self.df[self.price_col]
        gain = np.where(close.diff() > 0, close.diff(), 0)
        loss = np.where(close.diff() < 0, -close.diff(), 0)
        
        avg_gain = pd.Series(gain).rolling(window=14).mean()
        avg_loss = pd.Series(loss).rolling(window=14).mean()
        
        rs = avg_gain / (avg_loss + 1e-6)
        momentum = 100 - (100 / (1 + rs))
        return momentum.fillna(50)
    
    def _calculate_fear_index(self):
        """Calculate a crypto fear index based on volatility and volume"""
        volatility = self.df['volatility']
        
        # Normalize volatility to 0-100 scale (like VIX)
        vol_min = volatility.rolling(window=252).min()
        vol_max = volatility.rolling(window=252).max()
        fear_index = 100 * (volatility - vol_min) / (vol_max - vol_min + 1e-6)
        
        return fear_index.fillna(50)
    
    def calculate_dynamic_stop_loss(self, trade_date, predicted_return, confidence, position_type='long'):
        """
        Calculate dynamic stop loss based on multiple market factors
        
        Args:
            trade_date: Date of trade entry
            predicted_return: Model's predicted return
            confidence: Model's confidence in prediction
            position_type: 'long' or 'short'
        
        Returns:
            Dynamic stop loss percentage
        """
        try:
            # Get market data for the trade date
            if trade_date not in self.df.index:
                # Fallback to nearest date
                nearest_date = min(self.df.index, key=lambda x: abs(x - trade_date))
                market_data = self.df.loc[nearest_date]
            else:
                market_data = self.df.loc[trade_date]
            
            # Base stop loss factors
            base_stop = 0.03  # 3% base stop loss
            
            # 1. ATR-based volatility adjustment
            atr = market_data.get('atr', 0)
            current_price = market_data[self.price_col]
            atr_pct = (atr / current_price) if current_price > 0 else 0.02
            volatility_multiplier = np.clip(atr_pct / 0.02, 0.5, 3.0)  # Scale around 2% ATR baseline
            
            # 2. Historical volatility adjustment
            hist_vol = market_data.get('volatility', 0.5)
            vol_multiplier = np.clip(hist_vol / 0.5, 0.6, 2.5)  # Scale around 50% annual vol baseline
            
            # 3. Trend strength adjustment
            trend_strength = market_data.get('trend_strength', 0.3)
            if trend_strength > 0.7:  # Strong trend - wider stops
                trend_multiplier = 1.4
            elif trend_strength > 0.4:  # Moderate trend
                trend_multiplier = 1.0
            else:  # Weak trend - tighter stops
                trend_multiplier = 0.7
            
            # 4. Confidence-based adjustment
            confidence_multiplier = np.clip(2 - confidence * 2, 0.5, 1.5)  # Higher confidence = tighter stops
            
            # 5. Market momentum adjustment
            momentum = market_data.get('momentum', 50)
            if position_type == 'long':
                if momentum > 70:  # Overbought - tighter stops
                    momentum_multiplier = 0.8
                elif momentum < 30:  # Oversold - wider stops
                    momentum_multiplier = 1.3
                else:
                    momentum_multiplier = 1.0
            else:  # Short position
                if momentum > 70:  # Overbought - wider stops for shorts
                    momentum_multiplier = 1.3
                elif momentum < 30:  # Oversold - tighter stops for shorts
                    momentum_multiplier = 0.8
                else:
                    momentum_multiplier = 1.0
            
            # 6. Fear index adjustment
            fear_index = market_data.get('fear_index', 50)
            if fear_index > 75:  # High fear - wider stops
                fear_multiplier = 1.4
            elif fear_index < 25:  # Low fear (greed) - moderate stops
                fear_multiplier = 0.9
            else:
                fear_multiplier = 1.0
            
            # 7. Predicted return magnitude adjustment
            pred_magnitude = abs(predicted_return)
            if pred_magnitude > 0.1:  # High conviction trades get wider stops
                magnitude_multiplier = 1.3
            elif pred_magnitude > 0.05:
                magnitude_multiplier = 1.1
            else:
                magnitude_multiplier = 0.9
            
            # Combine all factors
            dynamic_stop = base_stop * volatility_multiplier * vol_multiplier * trend_multiplier * \
                          confidence_multiplier * momentum_multiplier * fear_multiplier * magnitude_multiplier
            
            # Apply bounds
            min_stop = 0.015  # 1.5% minimum
            max_stop = 0.12   # 12% maximum
            dynamic_stop = np.clip(dynamic_stop, min_stop, max_stop)
            
            return dynamic_stop
            
        except Exception as e:
            print(f"Error calculating dynamic stop loss: {str(e)}")
            return 0.05  # Fallback to 5%
    
    def calculate_trailing_stop(self, entry_price, current_price, entry_date, current_date, 
                               initial_stop_pct, position_type='long'):
        """
        Calculate trailing stop loss that moves in favorable direction
        
        Args:
            entry_price: Price at trade entry
            current_price: Current market price
            entry_date: Date of trade entry
            current_date: Current date
            initial_stop_pct: Initial stop loss percentage
            position_type: 'long' or 'short'
        
        Returns:
            Updated stop loss percentage
        """
        try:
            # Calculate unrealized P&L
            if position_type == 'long':
                unrealized_pnl = (current_price - entry_price) / entry_price
            else:
                unrealized_pnl = (entry_price - current_price) / entry_price
            
            # Only trail in profitable direction
            if unrealized_pnl <= 0:
                return initial_stop_pct
            
            # Trail stop based on profit level
            if unrealized_pnl > 0.15:  # 15%+ profit - trail very tight
                trailing_pct = max(initial_stop_pct * 0.3, 0.01)  # Trail to 30% of original or 1%
            elif unrealized_pnl > 0.08:  # 8%+ profit - trail moderately
                trailing_pct = max(initial_stop_pct * 0.5, 0.015)  # Trail to 50% of original or 1.5%
            elif unrealized_pnl > 0.04:  # 4%+ profit - trail slightly
                trailing_pct = max(initial_stop_pct * 0.7, 0.02)   # Trail to 70% of original or 2%
            else:
                trailing_pct = initial_stop_pct  # No trailing yet
            
            return trailing_pct
            
        except Exception as e:
            print(f"Error calculating trailing stop: {str(e)}")
            return initial_stop_pct

class TradingSimulator:
    """
    Enhanced trading simulation with dynamic stop losses
    """
    
    def __init__(self, predictor, min_acceptable_sharpe=0.5, max_acceptable_drawdown=0.2):
        self.predictor = predictor
        self.min_acceptable_sharpe = min_acceptable_sharpe
        self.max_acceptable_drawdown = max_acceptable_drawdown
        self.stop_loss_calculator = None
    
    def run_trading_simulation(self, df, initial_capital=10000, transaction_cost=0.001):
        """
        Run enhanced trading simulation with dynamic stop losses
        """
        print("="*60)
        print("DYNAMIC STOP LOSS TRADING SIMULATION")
        print("="*60)
        print(f"Data period: {df.index[0]} to {df.index[-1]}")
        print(f"Total days: {len(df)}")
        
        # Initialize dynamic stop loss calculator
        self.stop_loss_calculator = DynamicStopLossCalculator(df)
        
        # Train the model
        print("Training model...")
        try:
            self.predictor.train_ensemble(df, validation_split=0.2, epochs=100, batch_size=32)
            print("✅ Model training completed successfully")
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            return None
        
        # Run trading simulation
        print("\nRunning Enhanced Trading Simulation with Dynamic Stop Losses...")
        try:
            trading_results = self.simulate_trading(df, initial_capital=initial_capital, transaction_cost=transaction_cost)
            print("✅ Trading simulation completed successfully")
            return trading_results
        except Exception as e:
            print(f"❌ Trading simulation failed: {str(e)}")
            return None
    
    def apply_dynamic_stop_loss(self, predicted_return, actual_return, position_type, 
                               trade_date, confidence, entry_price=None, current_price=None, 
                               current_date=None, initial_stop_pct=None):
        """
        Apply dynamic stop loss with trailing functionality
        """
        try:
            # Calculate initial dynamic stop loss
            if initial_stop_pct is None:
                dynamic_stop_pct = self.stop_loss_calculator.calculate_dynamic_stop_loss(
                    trade_date, predicted_return, confidence, position_type
                )
            else:
                dynamic_stop_pct = initial_stop_pct
            
            # Apply trailing stop if we have price data
            if entry_price and current_price and current_date:
                trailing_stop_pct = self.stop_loss_calculator.calculate_trailing_stop(
                    entry_price, current_price, trade_date, current_date, 
                    dynamic_stop_pct, position_type
                )
                # Use the tighter of the two stops
                final_stop_pct = min(dynamic_stop_pct, trailing_stop_pct)
            else:
                final_stop_pct = dynamic_stop_pct
            
            # Apply stop loss logic
            if position_type == 'long' and actual_return < -final_stop_pct:
                return -final_stop_pct, True, final_stop_pct
            elif position_type == 'short' and actual_return > final_stop_pct:
                return -final_stop_pct, True, final_stop_pct
            else:
                return actual_return, False, final_stop_pct
                
        except Exception as e:
            print(f"Error in dynamic stop loss calculation: {str(e)}")
            # Fallback to simple 5% stop
            if position_type == 'long' and actual_return < -0.05:
                return -0.05, True, 0.05
            elif position_type == 'short' and actual_return > 0.05:
                return -0.05, True, 0.05
            else:
                return actual_return, False, 0.05
    
    def simulate_trading(self, df, initial_capital=10000, transaction_cost=0.001):
        """
        Enhanced trading simulation with dynamic stop losses
        """
        try:
            # Prepare data (same as before)
            df_proc = self.predictor.engineer_30day_target(df)
            features, _ = self.predictor.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            
            X, y, _ = self.predictor.create_sequences(features, targets)
            
            if len(X) == 0:
                print("  Warning: No sequences created for trading simulation")
                return self._empty_results(initial_capital)
            
            # Split data
            split_idx = int(0.7 * len(X))
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            if len(X_test) == 0:
                print("  Warning: No test data for trading simulation")
                return self._empty_results(initial_capital)
            
            # Get predictions using pre-trained model
            ensemble_pred, _, _ = self.predictor.predict_ensemble(X_test)
            
            # Simulate trading with dynamic stops
            capital = initial_capital
            positions = []
            returns = []
            equity_curve = [capital]
            trades_log = []
            
            # Get test data indices
            test_start_idx = split_idx
            test_dates = df_proc.index[test_start_idx:test_start_idx + len(ensemble_pred)]
            
            # Calculate BTC prices for test period
            btc_prices_start = []
            btc_prices_end = []
            trade_start_dates = []
            trade_end_dates = []
            exit_info = []
            
            # Pre-calculate exit points (same logic as before)
            for i in range(len(ensemble_pred)):
                if i < len(test_dates):
                    trade_start = test_dates[i]
                    try:
                        btc_start_price = df.loc[trade_start, 'close'] if 'close' in df.columns else df.loc[trade_start, 'Close']
                        pred_return = ensemble_pred[i][0]
                        
                        trade_end, btc_end_price, days_held, exit_reason = self._find_exit_point(
                            df, trade_start, btc_start_price, pred_return, max_days=30
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
            
            # Main trading loop with dynamic stops
            total_dynamic_stops = 0
            total_trailing_stops = 0
            
            for i in range(len(ensemble_pred)):
                pred_return = ensemble_pred[i][0]
                actual_return = y_test[i]
                
                # Position sizing based on confidence
                confidence = min(abs(pred_return), 0.1)
                position_size = confidence
                
                # Calculate actual BTC return
                actual_btc_return = float((btc_prices_end[i] - btc_prices_start[i]) / btc_prices_start[i]) if btc_prices_start[i] > 0 else 0.0
                
                trade_info = {
                    'index': i,
                    'date': str(test_dates[i]) if i < len(test_dates) else f"test_{i}",
                    'trade_start_date': str(trade_start_dates[i]),
                    'trade_end_date': str(trade_end_dates[i]),
                    'btc_price_start': float(btc_prices_start[i]),
                    'btc_price_end': float(btc_prices_end[i]),
                    'predicted_return': float(pred_return),
                    'actual_return': actual_btc_return,
                    'confidence': float(confidence),
                    'position_size': float(position_size),
                    'capital_before': float(capital),
                    'trade_type': 'no_trade',
                    'dynamic_stop_pct': 0.0,
                    'stop_loss_triggered': False,
                    'trailing_stop_used': False,
                    'exit_reason': exit_info[i]['exit_reason'],
                    'days_held': exit_info[i]['days_held']
                }
                
                # Trading logic with dynamic stops
                if abs(pred_return) > 0.02:  # Only trade if predicted return > 2%
                    position_type = 'long' if pred_return > 0 else 'short'
                    trade_info['trade_type'] = position_type
                    
                    # Apply dynamic stop loss
                    risk_adjusted_return, stop_triggered, dynamic_stop_pct = self.apply_dynamic_stop_loss(
                        pred_return, actual_btc_return, position_type, 
                        trade_start_dates[i], confidence,
                        entry_price=btc_prices_start[i],
                        current_price=btc_prices_end[i],
                        current_date=trade_end_dates[i]
                    )
                    
                    trade_info['dynamic_stop_pct'] = float(dynamic_stop_pct)
                    trade_info['stop_loss_triggered'] = stop_triggered
                    trade_info['risk_adjusted_return'] = float(risk_adjusted_return)
                    
                    if stop_triggered:
                        total_dynamic_stops += 1
                    
                    # Calculate position and returns
                    position_value = capital * position_size * (1 - transaction_cost)
                    
                    if position_type == 'long':
                        trade_return = position_value * risk_adjusted_return
                        capital += trade_return - (position_value * transaction_cost)
                        positions.append(1)
                    else:  # Short
                        trade_return = -position_value * risk_adjusted_return
                        capital += trade_return - (position_value * transaction_cost)
                        positions.append(-1)
                    
                    trade_info['position_value'] = float(position_value)
                    trade_info['trade_return'] = float(trade_return)
                    trade_info['transaction_cost_paid'] = float(position_value * transaction_cost * 2)
                    
                    returns.append(trade_return / (capital - trade_return) if capital - trade_return != 0 else 0)
                    trades_log.append(trade_info)
                    
                else:
                    positions.append(0)
                    returns.append(0)
                
                trade_info['capital_after'] = float(capital)
                trade_info['profit_loss'] = trade_info['capital_after'] - trade_info['capital_before']
                equity_curve.append(capital)
            
            # Calculate enhanced metrics
            total_return = (capital - initial_capital) / initial_capital
            returns_array = np.array(returns)
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
            
            n_trades = np.sum(np.array(positions) != 0)
            
            # Enhanced results with dynamic stop loss statistics
            results = {
                'simulation_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'initial_capital': initial_capital,
                    'transaction_cost': transaction_cost,
                    'stop_loss_type': 'dynamic_with_trailing',
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
                'dynamic_stop_loss_stats': {
                    'total_stops_triggered': total_dynamic_stops,
                    'stop_trigger_rate': total_dynamic_stops / max(n_trades, 1),
                    'avg_dynamic_stop_pct': np.mean([t['dynamic_stop_pct'] for t in trades_log if t['trade_type'] != 'no_trade']),
                    'min_dynamic_stop_pct': np.min([t['dynamic_stop_pct'] for t in trades_log if t['trade_type'] != 'no_trade']) if trades_log else 0,
                    'max_dynamic_stop_pct': np.max([t['dynamic_stop_pct'] for t in trades_log if t['trade_type'] != 'no_trade']) if trades_log else 0
                },
                'trades': trades_log,
                'equity_curve': [float(x) for x in equity_curve]
            }
            
            return results
            
        except Exception as e:
            print(f"  Error in enhanced trading simulation: {str(e)}")
            return self._empty_results(initial_capital)
    
    def _find_exit_point(self, df, start_date, start_price, predicted_return, max_days=30):
        """Helper function to find exit point (same as original)"""
        try:
            start_pos = df.index.get_loc(start_date)
            target_price = start_price * (1 + predicted_return)
            
            for day_offset in range(1, min(max_days + 1, len(df) - start_pos)):
                current_date = df.index[start_pos + day_offset]
                current_price = df.loc[current_date, 'close'] if 'close' in df.columns else df.loc[current_date, 'Close']
                
                if predicted_return > 0:
                    if current_price >= target_price * 0.99:
                        return current_date, current_price, day_offset, 'target_reached'
                else:
                    if current_price <= target_price * 1.01:
                        return current_date, current_price, day_offset, 'target_reached'
            
            end_pos = min(start_pos + max_days, len(df) - 1)
            end_date = df.index[end_pos]
            end_price = df.loc[end_date, 'close'] if 'close' in df.columns else df.loc[end_date, 'Close']
            return end_date, end_price, end_pos - start_pos, 'max_days_reached'
            
        except (KeyError, IndexError):
            return start_date, start_price, 0, 'error'
    
    def _empty_results(self, initial_capital):
        """Helper function for empty results"""
        return {
            'error': 'No data available',
            'performance_metrics': {
                'profitable': False, 'meets_sharpe_threshold': False, 'meets_drawdown_threshold': False,
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'n_trades': 0, 'win_rate': 0,
                'initial_capital': initial_capital, 'final_capital': initial_capital, 
                'annualized_return': 0, 'avg_trade_return': 0, 'trade_frequency': 0
            },
            'dynamic_stop_loss_stats': {
                'total_stops_triggered': 0,
                'stop_trigger_rate': 0,
                'avg_dynamic_stop_pct': 0,
                'min_dynamic_stop_pct': 0,
                'max_dynamic_stop_pct': 0
            },
            'trades': [], 'equity_curve': []
        }
    
    def save_results_to_json(self, results, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'dynamic_stop_loss_results_{timestamp}.json'
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Enhanced results saved to: {filename}")
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
    
    # Create enhanced trading simulator
    simulator = TradingSimulator(improved_predictor)
    
    # Run enhanced trading simulation
    results = simulator.run_trading_simulation(df, initial_capital=10000, transaction_cost=0.001)
    
    if results and 'performance_metrics' in results:
        # Save results to JSON
        filename = simulator.save_results_to_json(results)
        
        # Display enhanced results
        metrics = results['performance_metrics']
        metadata = results['simulation_metadata']
        stop_stats = results['dynamic_stop_loss_stats']
        
        print("\n" + "="*60)
        print("ENHANCED TRADING SIMULATION RESULTS")
        print("="*60)
        print(f"Test Period: {metadata['test_period_start']} to {metadata['test_period_end']}")
        print(f"Stop Loss Type: {metadata['stop_loss_type']}")
        print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.3f}")
        print(f"Number of Trades: {metrics['n_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.3f}")
        print(f"Average Trade Return: {metrics['avg_trade_return']:.4f}")
        
        print(f"\n🎯 DYNAMIC STOP LOSS PERFORMANCE:")
        print(f"Total Stops Triggered: {stop_stats['total_stops_triggered']}")
        print(f"Stop Trigger Rate: {stop_stats['stop_trigger_rate']:.1%}")
        print(f"Average Dynamic Stop: {stop_stats['avg_dynamic_stop_pct']:.2%}")
        print(f"Stop Range: {stop_stats['min_dynamic_stop_pct']:.2%} - {stop_stats['max_dynamic_stop_pct']:.2%}")
        
        # Show trade breakdown
        trades = results['trades']
        active_trades = [t for t in trades if t['trade_type'] != 'no_trade']
        long_trades = [t for t in active_trades if t['trade_type'] == 'long']
        short_trades = [t for t in active_trades if t['trade_type'] == 'short']
        winning_trades = [t for t in active_trades if t['profit_loss'] > 0]
        stop_loss_trades = [t for t in active_trades if t['stop_loss_triggered']]
        
        print(f"\n📊 TRADE BREAKDOWN:")
        print(f"Active Trades: {len(active_trades)}")
        print(f"Long Trades: {len(long_trades)}")
        print(f"Short Trades: {len(short_trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Stop Loss Triggered: {len(stop_loss_trades)}")
        
        if filename:
            print(f"\n📄 Detailed results saved to: {filename}")
    else:
        print("Enhanced trading simulation failed!")