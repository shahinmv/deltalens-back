import pandas as pd
import numpy as np
import sqlite3
import schedule
import time
import logging
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, Optional, Tuple
import requests
from dataclasses import dataclass

from main_ml import ImprovedBitcoinPredictor

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_automation.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class TradeSignal:
    """Structure for trade signals"""
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', 'HOLD'
    confidence: float
    predicted_return: float
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float
    timestamp: datetime
    expires_at: datetime

def apply_stop_loss(predicted_return, actual_return, position_type, stop_loss_pct=0.05):
    """Apply stop loss logic to limit downside - matches benchmark.py implementation"""
    if position_type == 'long' and actual_return < -stop_loss_pct:
        return -stop_loss_pct  # Cap loss at stop loss level
    elif position_type == 'short' and actual_return > stop_loss_pct:
        return -stop_loss_pct  # Cap loss for short position
    else:
        return actual_return  # No stop triggered, use actual return

class AutomatedTradingSystem:
    def __init__(self, 
                 db_path: str = "../../db.sqlite3",
                 model_path: str = "trained_model.pkl",
                 config_path: str = "trading_config.json"):
        
        self.db_path = db_path
        self.model_path = model_path
        self.config_path = config_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Trading parameters - aligned with successful benchmark
        self.min_prediction_threshold = self.config.get('min_prediction_threshold', 0.02)  # 2% like benchmark
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.05)  # 5% like benchmark
        self.max_position_size = self.config.get('max_position_size', 0.1)  # 10% max like benchmark
        self.signal_validity_hours = self.config.get('signal_validity_hours', 24)
        
        # Add benchmark-specific parameters
        self.benchmark_threshold = 0.02  # Successful benchmark threshold
        self.bear_market_threshold = 0.03  # Bear market threshold from benchmark
        self.normal_market_threshold = 0.025  # Normal market threshold from benchmark
        self.transaction_cost = 0.001  # Same as benchmark
        
        # Feature consistency tracking
        self.expected_features = None
        self.feature_mismatch_warnings = 0
        
        # Initialize predictor
        self.predictor = None
        self.current_signal = None
        
        # Add benchmark-style consecutive loss tracking
        self.consecutive_losses = 0
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        
    def _load_config(self) -> Dict:
        """Load trading configuration"""
        default_config = {
            "min_prediction_threshold": 0.02,  # Benchmark threshold
            "stop_loss_pct": 0.05,  # Benchmark stop loss
            "max_position_size": 0.1,  # Benchmark max position
            "signal_validity_hours": 24,
            "retrain_frequency_days": 7,
            "data_source": "binance",
            "symbol": "BTCUSDT",
            "readiness_threshold": 0.85,
            # Benchmark-specific parameters
            "benchmark_threshold": 0.02,
            "bear_market_threshold": 0.03,
            "normal_market_threshold": 0.025,
            "transaction_cost": 0.001,
            "max_consecutive_losses": 3  # From benchmark regime-aware logic
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                default_config.update(config)
            return default_config
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return default_config
    
    def extract_latest_data(self) -> bool:
        """Extract and append latest BTC data from Binance to database"""
        try:
            logging.info("Extracting latest BTC data from Binance...")
            
            # Get latest data from database to determine last date
            conn = sqlite3.connect(self.db_path)
            
            # Get the last datetime in database
            last_date_query = "SELECT MAX(datetime) FROM btc_daily_ohlcv"
            result = pd.read_sql_query(last_date_query, conn)['MAX(datetime)'].iloc[0]
            
            if result:
                # Convert to timestamp for Binance API
                last_datetime = pd.to_datetime(result)
                start_time = int((last_datetime + timedelta(days=1)).timestamp() * 1000)
            else:
                # If no data, get last 365 days
                start_time = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)
            
            # Current time
            end_time = int(datetime.now().timestamp() * 1000)
            
            # Binance API endpoint for daily klines
            url = "https://api.binance.com/api/v3/klines"
            
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1d',
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000  # Maximum allowed
            }
            
            # Make API request
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) == 0:
                logging.info("No new data available from Binance")
                conn.close()
                return True
            
            # Convert to DataFrame
            # Binance returns: [timestamp, open, high, low, close, volume, close_time, ...]
            df_new = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Select and format required columns
            df_new = df_new[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Convert timestamp to datetime
            df_new['datetime'] = pd.to_datetime(df_new['timestamp'], unit='ms')
            # Only keep the date part (yyyy-mm-dd)
            df_new['datetime'] = df_new['datetime'].dt.strftime('%Y-%m-%d')
            
            # Convert price and volume columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
            
            # Prepare final DataFrame with correct column order
            df_final = df_new[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Remove any duplicate dates that might already exist
            if result:
                last_datetime_str = last_datetime.strftime('%Y-%m-%d')
                df_final = df_final[df_final['datetime'] > last_datetime_str]
            
            if len(df_final) == 0:
                logging.info("No new data to insert (all data already exists)")
                conn.close()
                return True
            
            # Insert new data
            df_final.to_sql('btc_daily_ohlcv', conn, if_exists='append', index=False)
            
            conn.close()
            logging.info(f"Successfully added {len(df_final)} new records from Binance")
            logging.info(f"Date range: {df_final['datetime'].min()} to {df_final['datetime'].max()}")
            return True
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching data from Binance API: {e}")
            return False
        except Exception as e:
            logging.error(f"Error extracting latest data: {e}")
            return False
    
    def load_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """Load and prepare data using the same pipeline as benchmark"""
        try:
            logging.info("Loading and preparing data...")
            
            # Use the same data loading pipeline as benchmark
            btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data()
            
            if btc_ohlcv is None or len(btc_ohlcv) == 0:
                logging.error("No Bitcoin OHLCV data loaded")
                return None
            
            logging.info(f"Loaded {len(btc_ohlcv)} OHLCV records")
            
            # Process sentiment data if available
            if df_news is not None and len(df_news) > 0:
                logging.info(f"Processing {len(df_news)} news records for sentiment...")
                df_news = add_vader_sentiment(df_news)
                df_newsdaily_sentiment = aggregate_daily_sentiment(df_news)
                logging.info(f"Aggregated sentiment data: {len(df_newsdaily_sentiment)} records")
            else:
                logging.warning("No news data available, using default sentiment")
                df_newsdaily_sentiment = None
            
            # Feature engineering using the same function as benchmark
            df = engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment)
            
            logging.info(f"Successfully engineered features for {len(df)} records")
            logging.info(f"Feature columns: {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading and preparing data: {e}")
            return None
    
    def train_model(self, data: pd.DataFrame) -> bool:
        """Train or retrain the ML model using prepared data"""
        try:
            logging.info("Training ML model...")
            
            if len(data) < 200:  # Need sufficient data for training
                logging.error(f"Insufficient data for training: {len(data)} records")
                return False
            
            # Initialize predictor if not exists
            if self.predictor is None:
                self.predictor = ImprovedBitcoinPredictor(sequence_length=60, prediction_horizon=30)
            
            # Train the ensemble model
            X_val, y_val, regime_seq = self.predictor.train_ensemble(data, validation_split=0.2, epochs=100, batch_size=32)
            
            if X_val is None:
                logging.error("Model training failed")
                return False
            
            # Evaluate the model
            evaluation = self.predictor.evaluate_ensemble(X_val, y_val, regime_seq)
            
            logging.info(f"Model training completed successfully")
            logging.info(f"Model performance - MAE: {evaluation.get('mae', 0):.4f}, "
                        f"Direction Accuracy: {evaluation.get('direction_accuracy', 0):.3f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return False
    
    def generate_trading_signal(self, data: pd.DataFrame) -> Optional[TradeSignal]:
        """Generate trading signal using the same strategy as benchmark.py"""
        try:
            logging.info("Generating trading signal...")
            
            if self.predictor is None:
                logging.error("Model not trained yet")
                return None
            
            # Use the same prediction method as benchmark: prepare features and predict_ensemble
            df_proc = self.predictor.engineer_30day_target(data)
            features, _ = self.predictor.prepare_features(df_proc)
            
            # Create sequences like benchmark
            X, _, _ = self.predictor.create_sequences(features, np.zeros(len(features)))
            
            if len(X) == 0:
                logging.error("No sequences created for prediction")
                return None
            
            # Get prediction using ensemble method (same as benchmark approach)
            # To match benchmark exactly, we need to use the same sequence selection logic
            ensemble_pred, _, _ = self.predictor.predict_ensemble(X)
            
            # Take the last prediction (most recent) - matches benchmark's X_test approach
            predicted_return = ensemble_pred[-1][0]
            
            logging.info(f"Total sequences created: {len(X)}")
            logging.info(f"Using prediction from sequence index: {len(X)-1}")
            
            # Calculate confidence like benchmark (Kelly criterion approximation)
            base_confidence = min(abs(predicted_return), 0.1)
            
            # Get current price for signal generation
            current_price = float(data['close'].iloc[-1])
            
            # Position sizing based on Kelly criterion (matches benchmark.py)
            # Cap confidence at 10% position like benchmark
            kelly_confidence = min(abs(predicted_return), 0.1)
            
            # Apply benchmark thresholds for signal generation (same as benchmark: > 2%)
            if abs(predicted_return) > self.min_prediction_threshold:  # 0.02 = 2%
                if predicted_return > 0:
                    signal_type = 'LONG'
                else:
                    signal_type = 'SHORT'
                position_size = kelly_confidence  # Kelly criterion position sizing
            else:
                signal_type = 'HOLD'
                position_size = 0.0
            
            # Calculate target and stop loss prices
            if signal_type == 'LONG':
                target_price = current_price * (1 + abs(predicted_return))
                stop_loss = current_price * (1 - self.stop_loss_pct)
            elif signal_type == 'SHORT':
                target_price = current_price * (1 - abs(predicted_return))
                stop_loss = current_price * (1 + self.stop_loss_pct)
            else:
                target_price = current_price
                stop_loss = current_price
            
            # Account for transaction costs in signal (like benchmark)
            # Adjust expected return for transaction costs
            adjusted_predicted_return = predicted_return
            total_transaction_cost = 0
            if signal_type != 'HOLD':
                # Account for entry and exit transaction costs
                total_transaction_cost = 2 * self.transaction_cost  # Entry + Exit
                if signal_type == 'LONG':
                    adjusted_predicted_return = predicted_return - total_transaction_cost
                else:  # SHORT
                    adjusted_predicted_return = predicted_return + total_transaction_cost
            
            # Create signal with benchmark-style confidence calculation
            final_confidence = min(base_confidence, kelly_confidence)
            
            signal = TradeSignal(
                symbol="BTCUSDT",
                signal_type=signal_type,
                confidence=final_confidence,
                predicted_return=adjusted_predicted_return,  # Transaction cost adjusted
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                position_size=position_size,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(days=30)
            )
            
            logging.info(f"Generated signal: {signal_type} with confidence {final_confidence}")
            logging.info(f"Predicted return: {predicted_return} (adjusted: {adjusted_predicted_return})")
            logging.info(f"Position size: {position_size} (Kelly criterion)")
            logging.info(f"Transaction costs included: {total_transaction_cost}" if signal_type != 'HOLD' else "No transaction costs (HOLD)")
            
            return signal
            
        except Exception as e:
            logging.error(f"Error generating trading signal: {e}")
            return None
    
    def execute_trade_with_risk_management(self, signal: TradeSignal, current_capital: float, actual_return: float = None) -> Dict:
        """
        Execute trade with stop loss and transaction costs - matches simulate_trading logic exactly
        This method simulates the actual trade execution including risk management
        """
        try:
            if signal.signal_type == 'HOLD' or signal.position_size == 0:
                return {
                    'trade_executed': False,
                    'position_return': 0,
                    'capital_after_trade': current_capital,
                    'transaction_costs': 0,
                    'stop_loss_triggered': False
                }
            
            # Calculate position value
            position_value = current_capital * signal.position_size
            
            # Store original position value for reporting purposes
            original_position_value = position_value
            
            # Deduct entry transaction cost (this matches: position_value *= (1 - transaction_cost))
            position_value_after_entry_cost = position_value * (1 - self.transaction_cost)
            
            # If actual_return is provided, simulate the trade outcome
            if actual_return is not None:
                # Apply stop loss logic (matches simulate_trading)
                risk_adjusted_return = apply_stop_loss(
                    signal.predicted_return, 
                    actual_return, 
                    signal.signal_type.lower(), 
                    self.stop_loss_pct
                )
                
                stop_loss_triggered = (risk_adjusted_return != actual_return)
                
                # Calculate trade return based on position type (EXACTLY like simulate_trading)
                if signal.signal_type == 'LONG':
                    # Long position
                    trade_return = position_value_after_entry_cost * risk_adjusted_return
                    # EXIT COST: Use position_value_after_entry_cost (the reduced value), NOT original!
                    new_capital = current_capital + trade_return - (position_value_after_entry_cost * self.transaction_cost)
                else:  # SHORT
                    # Short position
                    trade_return = -position_value_after_entry_cost * risk_adjusted_return
                    # EXIT COST: Use position_value_after_entry_cost (the reduced value), NOT original!
                    new_capital = current_capital + trade_return - (position_value_after_entry_cost * self.transaction_cost)
                
                # Calculate total transaction costs for reporting
                entry_cost = original_position_value * self.transaction_cost
                exit_cost = position_value_after_entry_cost * self.transaction_cost  # Corrected!
                total_transaction_costs = entry_cost + exit_cost
                
                # Calculate final trade return for reporting
                final_trade_return = new_capital - current_capital
                
                return {
                    'trade_executed': True,
                    'position_return': final_trade_return / current_capital,
                    'capital_after_trade': new_capital,
                    'transaction_costs': total_transaction_costs,
                    'stop_loss_triggered': stop_loss_triggered,
                    'risk_adjusted_return': risk_adjusted_return,
                    'original_return': actual_return,
                    'position_value': original_position_value,
                    'signal_type': signal.signal_type
                }
            else:
                # Just return the expected trade setup (no execution)
                # Note: Total transaction cost is slightly less than 2 * original * rate due to compounding
                expected_entry_cost = position_value * self.transaction_cost
                expected_exit_cost = (position_value * (1 - self.transaction_cost)) * self.transaction_cost
                
                return {
                    'trade_executed': False,
                    'expected_position_value': position_value,
                    'expected_transaction_costs': expected_entry_cost + expected_exit_cost,
                    'signal_ready': True
                }
                
        except Exception as e:
            logging.error(f"Error executing trade: {e}")
            return {
                'trade_executed': False,
                'error': str(e),
                'capital_after_trade': current_capital
            }
        
    def save_signal_to_database(self, signal: TradeSignal) -> bool:
        """Save trading signal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    predicted_return REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    target_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    position_size REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT DEFAULT 'ACTIVE'
                )
            """)
            
            # Insert signal
            cursor.execute("""
                INSERT INTO trading_signals 
                (symbol, signal_type, confidence, predicted_return, entry_price, 
                 target_price, stop_loss, position_size, timestamp, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol,
                signal.signal_type,
                signal.confidence,
                signal.predicted_return,
                signal.entry_price,
                signal.target_price,
                signal.stop_loss,
                signal.position_size,
                signal.timestamp.isoformat(),
                signal.expires_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Signal saved to database successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error saving signal to database: {e}")
            return False
    
    def daily_trading_check(self):
        """Main daily trading check routine"""
        logging.info("=" * 50)
        logging.info("Starting daily trading check")
        
        try:
            # 1. Extract latest data
            if not self.extract_latest_data():
                logging.error("Failed to extract latest data")
                return
            
            # 2. Load and prepare all data
            data = self.load_and_prepare_data()
            if data is None:
                logging.error("Failed to load and prepare data")
                return
            
            # 3. Train or retrain model if needed
            if not self.train_model(data):
                logging.error("Failed to train model")
                return
            
            # 4. Generate trading signal
            signal = self.generate_trading_signal(data)
            if signal is None:
                logging.error("Failed to generate trading signal")
                return
            
            # 5. Save signal to database
            if not self.save_signal_to_database(signal):
                logging.error("Failed to save signal to database")
                return
            
            # 6. Store current signal
            self.current_signal = signal
            
            logging.info("Daily trading check completed successfully")
            logging.info(f"Current signal: {signal.signal_type} - {signal.symbol}")
            
        except Exception as e:
            logging.error(f"Error in daily trading check: {e}")
    
    def start_automated_system(self):
        """Start the automated trading system with scheduling"""
        logging.info("Starting automated trading system...")
        
        # Schedule daily checks at 9 AM UTC (after markets open)
        schedule.every().day.at("09:00").do(self.daily_trading_check)
        
        logging.info("Automated system scheduled. Running initial check...")
        
        # Run initial check
        self.daily_trading_check()
        
        # Keep the system running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logging.info("Automated system stopped by user")
        except Exception as e:
            logging.error(f"Error in automated system: {e}")
    
    def get_current_signal(self) -> Optional[TradeSignal]:
        """Get the current active signal"""
        return self.current_signal
    
    def is_signal_valid(self, signal: TradeSignal) -> bool:
        """Check if a signal is still valid (not expired)"""
        if signal is None:
            return False
        return datetime.now() < signal.expires_at
    
    def simulate_trading_strategy(self, data: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """
        Simulate trading strategy performance - matches benchmark.py simulate_trading logic
        This method tests the strategy on historical data
        """
        try:
            logging.info("Simulating trading strategy performance...")
            
            # Prepare data using the same pipeline as benchmark
            df_proc = self.predictor.engineer_30day_target(data)
            features, _ = self.predictor.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            
            X, y, _ = self.predictor.create_sequences(features, targets)
            
            if len(X) == 0:
                logging.warning("No sequences created for trading simulation")
                return {
                    'error': 'No sequences created',
                    'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                    'n_trades': 0, 'win_rate': 0.5, 'final_capital': initial_capital
                }
            
            # Split data for out-of-sample testing (same as benchmark)
            split_idx = int(0.7 * len(X))
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            if len(X_test) == 0:
                logging.warning("No test data for trading simulation")
                return {
                    'error': 'No test data',
                    'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                    'n_trades': 0, 'win_rate': 0.5, 'final_capital': initial_capital
                }
            
            # Get predictions using ensemble model
            ensemble_pred, _, _ = self.predictor.predict_ensemble(X_test)
            
            # Initialize trading simulation
            capital = initial_capital
            returns = []
            equity_curve = [capital]
            trades_executed = 0
            winning_trades = 0
            
            # Simulate trading on test data
            for i in range(len(ensemble_pred)):
                predicted_return = ensemble_pred[i][0]
                actual_return = y_test[i]
                
                # Create a temporary signal for this prediction
                temp_signal = TradeSignal(
                    symbol="BTCUSDT",
                    signal_type='HOLD',
                    confidence=min(abs(predicted_return), 0.1),
                    predicted_return=predicted_return,
                    entry_price=100,  # Dummy price
                    target_price=100,
                    stop_loss=100,
                    position_size=0,
                    timestamp=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=24)
                )
                
                # Apply same logic as generate_trading_signal
                kelly_confidence = min(abs(predicted_return), 0.1)
                
                if abs(predicted_return) > self.min_prediction_threshold:  # 2% threshold
                    if predicted_return > 0:
                        temp_signal.signal_type = 'LONG'
                    else:
                        temp_signal.signal_type = 'SHORT'
                    temp_signal.position_size = kelly_confidence
                    
                    # Execute trade with risk management
                    trade_result = self.execute_trade_with_risk_management(
                        temp_signal, capital, actual_return
                    )
                    
                    if trade_result['trade_executed']:
                        trades_executed += 1
                        old_capital = capital
                        capital = trade_result['capital_after_trade']
                        
                        # Calculate return exactly like benchmark.py: trade_return / (capital - trade_return)
                        trade_return_amount = capital - old_capital  # Actual profit/loss amount
                        benchmark_return = trade_return_amount / (old_capital) if old_capital != 0 else 0
                        returns.append(benchmark_return)
                        
                        if trade_return_amount > 0:
                            winning_trades += 1
                            
                        # Log trade details
                        if trade_result['stop_loss_triggered']:
                            logging.debug(f"Stop loss triggered on trade {trades_executed}")
                    else:
                        returns.append(0)
                else:
                    returns.append(0)  # HOLD
                
                equity_curve.append(capital)
            
            # Calculate performance metrics
            total_return = (capital - initial_capital) / initial_capital
            returns_array = np.array(returns)
            
            # Active returns (non-zero)
            active_returns = returns_array[returns_array != 0]
            
            if len(active_returns) > 0:
                sharpe_ratio = np.mean(active_returns) / (np.std(active_returns) + 1e-6) * np.sqrt(252/30)
                win_rate = winning_trades / trades_executed if trades_executed > 0 else 0.5
                avg_trade_return = np.mean(active_returns)
            else:
                sharpe_ratio = 0
                win_rate = 0.5
                avg_trade_return = 0
            
            # Calculate maximum drawdown
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            results = {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': total_return,
                'annualized_return': total_return * 252/30 / len(y_test) if len(y_test) > 0 else 0,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'n_trades': trades_executed,
                'win_rate': win_rate,
                'avg_trade_return': avg_trade_return,
                'trade_frequency': trades_executed / len(y_test) if len(y_test) > 0 else 0,
                'strategy_matches_benchmark': True  # Flag indicating strategy alignment
            }
            
            logging.info(f"Trading simulation completed:")
            logging.info(f"  Total Return: {total_return:.2%}")
            logging.info(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
            logging.info(f"  Max Drawdown: {max_drawdown:.2%}")
            logging.info(f"  Win Rate: {win_rate:.2%}")
            logging.info(f"  Number of Trades: {trades_executed}")
            
            return results
            
        except Exception as e:
            logging.error(f"Error in trading simulation: {e}")
            return {
                'error': str(e),
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
                'n_trades': 0, 'win_rate': 0.5, 'final_capital': initial_capital
            }

# Example usage and configuration
if __name__ == "__main__":

    
    # Initialize and start system
    trading_system = AutomatedTradingSystem()
    
    # For testing, run single check
    trading_system.daily_trading_check()
    
    # To start automated system (uncomment):
    # trading_system.start_automated_system()