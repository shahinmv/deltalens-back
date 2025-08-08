import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

# Add the parent directory to path for imports
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from main_ml import ImprovedBitcoinPredictor
from data_loader import load_all_data
from sentiment import add_vader_sentiment, aggregate_daily_sentiment
from feature_engineering_prod import engineer_features
from iterative_signal_prod import DynamicStopLossCalculator, apply_dynamic_stop_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_benchmark_simulator.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class DailySignal:
    """Individual daily signal"""
    signal_date: str
    predicted_return: float
    confidence: float
    position_type: str
    dynamic_stop_loss: float
    entry_price: float
    target_price: float
    stop_loss_price: float
    market_volatility: float
    trend_strength: float

@dataclass
class TradeResult:
    """Result of a completed trade"""
    signal_date: str
    entry_price: float
    exit_price: float
    exit_date: str
    days_held: int
    position_type: str
    predicted_return: float
    actual_return: float
    applied_return: float  # After transaction costs
    target_price: float
    stop_loss_price: float
    exit_reason: str  # 'TARGET_HIT', 'STOP_LOSS_HIT', 'EXPIRED'
    hit_target: bool
    hit_stop_loss: bool
    confidence: float
    dynamic_stop_loss: float
    market_volatility: float
    trend_strength: float

@dataclass
class BenchmarkResult:
    """Results for a single benchmark period (30 days of daily signals)"""
    test_date: str
    market_scenario: str
    training_start: str
    training_end: str
    daily_signals: List[DailySignal]
    completed_trades: List[TradeResult]
    total_signals: int
    active_trades: int
    hold_signals: int
    avg_return: float
    total_return: float
    win_rate: float
    stop_loss_rate: float

class TradingBenchmarkSimulator:
    def __init__(self, db_path: str = "../../db.sqlite3"):
        self.db_path = db_path
        self.predictor = None
        self.stop_loss_calculator = None
        
        # Benchmark test dates with market scenarios
        self.benchmark_dates = {
            "2024-03-01": "Bull Market Rally (ETF Approval Impact)",
            "2023-07-01": "Bear Market Recovery Attempt", 
            "2022-11-01": "FTX Collapse Period",
            "2022-01-01": "Market Peak Before Major Correction",
            "2021-09-01": "Strong Bull Market (Institutional Adoption)",
            "2021-05-01": "China Mining Ban Period",
            "2020-10-01": "COVID Recovery Bull Run Start",
            "2020-03-01": "COVID Market Crash"
        }
        
        # Trading parameters matching iterative_signal_prod
        self.min_prediction_threshold = 0.02  # 2%
        self.base_stop_loss_pct = 0.03  # 3%
        self.transaction_cost = 0.001  # 0.1%
        self.trade_duration_days = 30  # Hold for 30 days like prediction horizon
        
        self.results = []
        
    def get_training_period(self, test_date: str) -> Tuple[str, str]:
        """Get 18-month training period prior to test date"""
        test_dt = datetime.strptime(test_date, '%Y-%m-%d')
        training_end = test_dt - timedelta(days=1)  # Day before test
        training_start = training_end - timedelta(days=18*30)  # Approximately 18 months
        
        return training_start.strftime('%Y-%m-%d'), training_end.strftime('%Y-%m-%d')
    
    def load_data_for_period(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load and prepare data for specified period"""
        try:
            logging.info(f"Loading data from {start_date} to {end_date}")
            
            # Load raw data
            btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data()
            
            if btc_ohlcv is None or len(btc_ohlcv) == 0:
                logging.error("No Bitcoin OHLCV data loaded")
                return None
            
            print(f"btc_ohlcv index type: {type(btc_ohlcv.index)}")
            print(f"btc_ohlcv columns: {btc_ohlcv.columns.tolist()}")
            
            # Filter data for the specified period using the datetime index
            btc_ohlcv.index = pd.to_datetime(btc_ohlcv.index)
            btc_ohlcv = btc_ohlcv[
                (btc_ohlcv.index >= start_date) & 
                (btc_ohlcv.index <= end_date)
            ].copy()
            
            if len(btc_ohlcv) == 0:
                logging.error(f"No data available for period {start_date} to {end_date}")
                return None
            
            logging.info(f"Loaded {len(btc_ohlcv)} OHLCV records for period")
            
            # Process sentiment data if available
            df_newsdaily_sentiment = None
            if df_news is not None and len(df_news) > 0:
                # Handle datetime column or index for news data
                if 'datetime' in df_news.columns:
                    df_news['datetime'] = pd.to_datetime(df_news['datetime'])
                    df_news_filtered = df_news[
                        (df_news['datetime'] >= start_date) & 
                        (df_news['datetime'] <= end_date)
                    ]
                else:
                    # datetime is likely the index
                    df_news.index = pd.to_datetime(df_news.index)
                    df_news_filtered = df_news[
                        (df_news.index >= start_date) & 
                        (df_news.index <= end_date)
                    ]
                if len(df_news_filtered) > 0:
                    df_news_filtered = add_vader_sentiment(df_news_filtered)
                    df_newsdaily_sentiment = aggregate_daily_sentiment(df_news_filtered)
            
            # Filter other data sources similarly
            if daily_oi is not None:
                daily_oi.index = pd.to_datetime(daily_oi.index)
                daily_oi = daily_oi[
                    (daily_oi.index >= start_date) & 
                    (daily_oi.index <= end_date)
                ]
            
            if daily_funding_rate is not None:
                daily_funding_rate.index = pd.to_datetime(daily_funding_rate.index)
                daily_funding_rate = daily_funding_rate[
                    (daily_funding_rate.index >= start_date) & 
                    (daily_funding_rate.index <= end_date)
                ]
            
            # Engineer features
            df = engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment)
            
            logging.info(f"Successfully engineered features for {len(df)} records")
            return df
            
        except Exception as e:
            logging.error(f"Error loading data for period {start_date} to {end_date}: {e}")
            return None
    
    def check_trade_exit(self, signal: DailySignal, ohlcv_data: pd.DataFrame) -> Tuple[str, float, str, int]:
        """Check if trade should exit based on target/stop loss or expiry"""
        entry_price = signal.entry_price
        target_price = signal.target_price
        stop_loss_price = signal.stop_loss_price
        position_type = signal.position_type
        
        # Check each day for target/stop loss hit
        for i, (date, row) in enumerate(ohlcv_data.iterrows()):
            current_date = date.strftime('%Y-%m-%d')
            days_held = i + 1
            
            if position_type == 'LONG':
                # Check if target hit (high >= target)
                if row['high'] >= target_price:
                    return 'TARGET_HIT', target_price, current_date, days_held
                # Check if stop loss hit (low <= stop loss)
                elif row['low'] <= stop_loss_price:
                    return 'STOP_LOSS_HIT', stop_loss_price, current_date, days_held
                    
            elif position_type == 'SHORT':
                # Check if target hit (low <= target)
                if row['low'] <= target_price:
                    return 'TARGET_HIT', target_price, current_date, days_held
                # Check if stop loss hit (high >= stop loss)
                elif row['high'] >= stop_loss_price:
                    return 'STOP_LOSS_HIT', stop_loss_price, current_date, days_held
        
        # If no exit condition met, hold for full period
        exit_price = ohlcv_data.iloc[-1]['close']
        exit_date = ohlcv_data.index[-1].strftime('%Y-%m-%d')
        days_held = len(ohlcv_data)
        
        return 'EXPIRED', exit_price, exit_date, days_held
    
    def train_model_for_period(self, data: pd.DataFrame) -> bool:
        """Train model on the provided data"""
        try:
            logging.info(f"Training model on {len(data)} records...")
            
            if len(data) < 200:
                logging.error(f"Insufficient data for training: {len(data)} records")
                return False
            
            # Initialize predictor
            self.predictor = ImprovedBitcoinPredictor(sequence_length=60, prediction_horizon=30)
            
            # Train the ensemble model
            X_val, y_val, regime_seq = self.predictor.train_ensemble(
                data, validation_split=0.2, epochs=100, batch_size=32
            )
            
            if X_val is None:
                logging.error("Model training failed")
                return False
            
            # Initialize dynamic stop loss calculator
            self.stop_loss_calculator = DynamicStopLossCalculator(data)
            
            # Evaluate model
            evaluation = self.predictor.evaluate_ensemble(X_val, y_val, regime_seq)
            logging.info(f"Model training completed - MAE: {evaluation.get('mae', 0):.4f}, "
                        f"Direction Accuracy: {evaluation.get('direction_accuracy', 0):.3f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training model: {e}")
            return False
    
    def generate_prediction(self, data: pd.DataFrame, test_date: str) -> Tuple[float, float, str, float]:
        """Generate prediction for the test date"""
        try:
            if self.predictor is None:
                logging.error("Model not trained")
                return 0.0, 0.0, 'HOLD', 0.0
            
            # Prepare features using the same method as iterative_signal_prod
            df_proc = self.predictor.engineer_30day_target(data)
            features, _ = self.predictor.prepare_features(df_proc)
            
            # Create sequences
            X, _, _ = self.predictor.create_sequences(features, np.zeros(len(features)))
            
            if len(X) == 0:
                logging.error("No sequences created for prediction")
                return 0.0, 0.0, 'HOLD', 0.0
            
            # Get ensemble prediction
            ensemble_pred, _, _ = self.predictor.predict_ensemble(X)
            predicted_return = ensemble_pred[-1][0]
            
            # Calculate confidence (Kelly criterion approximation)
            confidence = min(abs(predicted_return), 0.1)
            
            # Determine position type based on threshold
            if abs(predicted_return) > self.min_prediction_threshold:
                position_type = 'LONG' if predicted_return > 0 else 'SHORT'
            else:
                position_type = 'HOLD'
            
            # Calculate dynamic stop loss
            current_date = data.index[-1]
            dynamic_stop_pct = self.stop_loss_calculator.calculate_dynamic_stop_loss(
                current_date, predicted_return, confidence, position_type.lower()
            )
            
            logging.info(f"Generated prediction: {predicted_return:.4f} ({predicted_return*100:.2f}%), "
                        f"Position: {position_type}, Dynamic Stop: {dynamic_stop_pct*100:.2f}%")
            
            return predicted_return, confidence, position_type, dynamic_stop_pct
            
        except Exception as e:
            logging.error(f"Error generating prediction: {e}")
            return 0.0, 0.0, 'HOLD', 0.0
    
    def get_entry_price(self, test_date: str) -> float:
        """Get entry price for the test date"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT close FROM btc_daily_ohlcv 
                WHERE datetime = ?
            """
            
            result = pd.read_sql_query(query, conn, params=(test_date,))
            conn.close()
            
            if result.empty:
                logging.error(f"No price data found for {test_date}")
                return 0.0
            
            return float(result.iloc[0]['close'])
            
        except Exception as e:
            logging.error(f"Error getting entry price for {test_date}: {e}")
            return 0.0
    
    def get_market_conditions(self, test_date: str) -> Tuple[float, float]:
        """Get market volatility and trend strength for the test date"""
        try:
            if self.stop_loss_calculator is None:
                return 0.5, 0.3  # Default values
            
            market_info = self.stop_loss_calculator.get_market_regime_info(test_date)
            return market_info['volatility'], market_info['trend_strength']
            
        except Exception as e:
            logging.warning(f"Error getting market conditions: {e}")
            return 0.5, 0.3
    
    def generate_daily_signals(self, training_data: pd.DataFrame, start_date: str) -> List[DailySignal]:
        """Generate signals for 30 consecutive days"""
        daily_signals = []
        
        for day_offset in range(30):
            signal_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=day_offset)).strftime('%Y-%m-%d')
            
            try:
                # Generate prediction for this day
                predicted_return, confidence, position_type, dynamic_stop_pct = self.generate_prediction(
                    training_data, signal_date
                )
                
                # Get entry price for this day
                entry_price = self.get_entry_price(signal_date)
                if entry_price == 0.0:
                    logging.warning(f"No price data for {signal_date}, skipping")
                    continue
                
                # Calculate target and stop loss prices
                if position_type == 'LONG':
                    target_price = entry_price * (1 + abs(predicted_return))
                    stop_loss_price = entry_price * (1 - dynamic_stop_pct)
                elif position_type == 'SHORT':
                    target_price = entry_price * (1 - abs(predicted_return))
                    stop_loss_price = entry_price * (1 + dynamic_stop_pct)
                else:  # HOLD
                    target_price = entry_price
                    stop_loss_price = entry_price
                
                # Get market conditions
                market_volatility, trend_strength = self.get_market_conditions(signal_date)
                
                # Create daily signal
                signal = DailySignal(
                    signal_date=signal_date,
                    predicted_return=predicted_return,
                    confidence=confidence,
                    position_type=position_type,
                    dynamic_stop_loss=dynamic_stop_pct,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_loss_price=stop_loss_price,
                    market_volatility=market_volatility,
                    trend_strength=trend_strength
                )
                
                daily_signals.append(signal)
                
                logging.info(f"Day {day_offset+1} ({signal_date}): {position_type} - "
                           f"Pred: {predicted_return*100:.2f}%, Entry: ${entry_price:.2f}, "
                           f"Target: ${target_price:.2f}, Stop: ${stop_loss_price:.2f}")
                
            except Exception as e:
                logging.error(f"Error generating signal for {signal_date}: {e}")
                continue
        
        return daily_signals
    
    def execute_trades(self, signals: List[DailySignal]) -> List[TradeResult]:
        """Execute trades based on daily signals with proper exit logic"""
        completed_trades = []
        
        for signal in signals:
            if signal.position_type == 'HOLD':
                continue
                
            try:
                # Get OHLCV data for the next 30 days from signal date
                signal_dt = datetime.strptime(signal.signal_date, '%Y-%m-%d')
                end_dt = signal_dt + timedelta(days=30)
                
                conn = sqlite3.connect(self.db_path)
                query = """
                    SELECT datetime, open, high, low, close, volume 
                    FROM btc_daily_ohlcv 
                    WHERE datetime > ? AND datetime <= ?
                    ORDER BY datetime ASC
                """
                
                ohlcv_data = pd.read_sql_query(
                    query, conn, 
                    params=(signal.signal_date, end_dt.strftime('%Y-%m-%d'))
                )
                conn.close()
                
                if ohlcv_data.empty:
                    logging.warning(f"No future price data for trade starting {signal.signal_date}")
                    continue
                
                # Convert datetime to index
                ohlcv_data['datetime'] = pd.to_datetime(ohlcv_data['datetime'])
                ohlcv_data.set_index('datetime', inplace=True)
                
                # Check for exit conditions
                exit_reason, exit_price, exit_date, days_held = self.check_trade_exit(signal, ohlcv_data)
                
                # Calculate returns
                actual_return = (exit_price - signal.entry_price) / signal.entry_price
                if signal.position_type == 'SHORT':
                    actual_return = -actual_return  # Invert for short positions
                
                # Apply transaction costs
                applied_return = actual_return - (2 * self.transaction_cost)
                
                # Create trade result
                trade_result = TradeResult(
                    signal_date=signal.signal_date,
                    entry_price=signal.entry_price,
                    exit_price=exit_price,
                    exit_date=exit_date,
                    days_held=days_held,
                    position_type=signal.position_type,
                    predicted_return=signal.predicted_return,
                    actual_return=actual_return,
                    applied_return=applied_return,
                    target_price=signal.target_price,
                    stop_loss_price=signal.stop_loss_price,
                    exit_reason=exit_reason,
                    hit_target=(exit_reason == 'TARGET_HIT'),
                    hit_stop_loss=(exit_reason == 'STOP_LOSS_HIT'),
                    confidence=signal.confidence,
                    dynamic_stop_loss=signal.dynamic_stop_loss,
                    market_volatility=signal.market_volatility,
                    trend_strength=signal.trend_strength
                )
                
                completed_trades.append(trade_result)
                
                logging.info(f"Trade completed: {signal.signal_date} {signal.position_type} - "
                           f"Exit: {exit_reason}, Return: {applied_return*100:.2f}%, Days: {days_held}")
                
            except Exception as e:
                logging.error(f"Error executing trade for {signal.signal_date}: {e}")
                continue
        
        return completed_trades
    
    def run_benchmark(self) -> List[BenchmarkResult]:
        """Run benchmark tests with daily signal generation"""
        logging.info("=" * 80)
        logging.info("STARTING DAILY TRADING BENCHMARK SIMULATION")
        logging.info("=" * 80)
        
        results = []
        
        for test_date, scenario in self.benchmark_dates.items():
            logging.info(f"\n{'='*60}")
            logging.info(f"TESTING: {test_date} - {scenario}")
            logging.info(f"Generating 30 daily signals starting from {test_date}")
            logging.info(f"{'='*60}")
            
            try:
                # Get training period (18 months prior)
                training_start, training_end = self.get_training_period(test_date)
                logging.info(f"Training period: {training_start} to {training_end}")
                
                # Load training data
                training_data = self.load_data_for_period(training_start, training_end)
                if training_data is None:
                    logging.error(f"Failed to load training data for {test_date}")
                    continue
                
                # Train model
                if not self.train_model_for_period(training_data):
                    logging.error(f"Failed to train model for {test_date}")
                    continue
                
                # Generate 30 daily signals
                daily_signals = self.generate_daily_signals(training_data, test_date)
                if not daily_signals:
                    logging.error(f"Failed to generate daily signals for {test_date}")
                    continue
                
                logging.info(f"Generated {len(daily_signals)} daily signals")
                
                # Execute trades based on signals
                completed_trades = self.execute_trades(daily_signals)
                logging.info(f"Completed {len(completed_trades)} trades")
                
                # Calculate summary statistics
                total_signals = len(daily_signals)
                active_trades = len([s for s in daily_signals if s.position_type != 'HOLD'])
                hold_signals = total_signals - active_trades
                
                if completed_trades:
                    returns = [t.applied_return for t in completed_trades]
                    avg_return = np.mean(returns)
                    total_return = np.sum(returns)
                    winning_trades = len([t for t in completed_trades if t.applied_return > 0])
                    win_rate = winning_trades / len(completed_trades)
                    stop_loss_hits = len([t for t in completed_trades if t.hit_stop_loss])
                    stop_loss_rate = stop_loss_hits / len(completed_trades)
                else:
                    avg_return = total_return = win_rate = stop_loss_rate = 0.0
                
                # Create benchmark result
                result = BenchmarkResult(
                    test_date=test_date,
                    market_scenario=scenario,
                    training_start=training_start,
                    training_end=training_end,
                    daily_signals=daily_signals,
                    completed_trades=completed_trades,
                    total_signals=total_signals,
                    active_trades=active_trades,
                    hold_signals=hold_signals,
                    avg_return=avg_return,
                    total_return=total_return,
                    win_rate=win_rate,
                    stop_loss_rate=stop_loss_rate
                )
                
                results.append(result)
                
                # Log summary results
                logging.info(f"SUMMARY for {test_date}:")
                logging.info(f"  Total Signals: {total_signals}")
                logging.info(f"  Active Trades: {active_trades}")
                logging.info(f"  Hold Signals: {hold_signals}")
                logging.info(f"  Completed Trades: {len(completed_trades)}")
                logging.info(f"  Average Return: {avg_return:.4f} ({avg_return*100:.2f}%)")
                logging.info(f"  Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
                logging.info(f"  Win Rate: {win_rate:.3f} ({win_rate*100:.1f}%)")
                logging.info(f"  Stop Loss Rate: {stop_loss_rate:.3f} ({stop_loss_rate*100:.1f}%)")
                
            except Exception as e:
                logging.error(f"Error processing {test_date}: {e}")
                continue
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict:
        """Analyze benchmark results and generate summary statistics"""
        if not self.results:
            return {}
        
        logging.info("\n" + "="*80)
        logging.info("DAILY TRADING BENCHMARK RESULTS ANALYSIS")
        logging.info("="*80)
        
        # Aggregate statistics across all benchmark periods
        all_trades = []
        total_signals = 0
        total_active_signals = 0
        total_hold_signals = 0
        
        for result in self.results:
            all_trades.extend(result.completed_trades)
            total_signals += result.total_signals
            total_active_signals += result.active_trades
            total_hold_signals += result.hold_signals
        
        # Calculate overall performance metrics
        if all_trades:
            returns = [t.applied_return for t in all_trades]
            avg_return = np.mean(returns)
            total_return = np.sum(returns)
            
            # Win/loss statistics
            winning_trades = len([t for t in all_trades if t.applied_return > 0])
            win_rate = winning_trades / len(all_trades)
            
            # Exit reason statistics
            target_hits = len([t for t in all_trades if t.hit_target])
            stop_loss_hits = len([t for t in all_trades if t.hit_stop_loss])
            expired_trades = len([t for t in all_trades if t.exit_reason == 'EXPIRED'])
            
            target_hit_rate = target_hits / len(all_trades)
            stop_loss_rate = stop_loss_hits / len(all_trades)
            expiry_rate = expired_trades / len(all_trades)
            
            # Direction accuracy
            correct_predictions = sum(1 for t in all_trades 
                                    if (t.predicted_return > 0 and t.actual_return > 0) or 
                                       (t.predicted_return < 0 and t.actual_return < 0))
            direction_accuracy = correct_predictions / len(all_trades)
            
            # Average holding period
            avg_days_held = np.mean([t.days_held for t in all_trades])
            
        else:
            avg_return = total_return = win_rate = 0
            target_hit_rate = stop_loss_rate = expiry_rate = direction_accuracy = 0
            avg_days_held = 0
            target_hits = stop_loss_hits = expired_trades = 0
        
        # Print overall summary
        logging.info(f"OVERALL STATISTICS:")
        logging.info(f"Total Benchmark Periods: {len(self.results)}")
        logging.info(f"Total Signals Generated: {total_signals}")
        logging.info(f"Active Signals: {total_active_signals}")
        logging.info(f"Hold Signals: {total_hold_signals}")
        logging.info(f"Completed Trades: {len(all_trades)}")
        logging.info(f"")
        
        if all_trades:
            logging.info(f"PERFORMANCE METRICS:")
            logging.info(f"Average Return per Trade: {avg_return:.4f} ({avg_return*100:.2f}%)")
            logging.info(f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
            logging.info(f"Win Rate: {win_rate:.3f} ({win_rate*100:.1f}%)")
            logging.info(f"Direction Accuracy: {direction_accuracy:.3f} ({direction_accuracy*100:.1f}%)")
            logging.info(f"Average Days Held: {avg_days_held:.1f}")
            logging.info(f"")
            logging.info(f"EXIT STATISTICS:")
            logging.info(f"Target Hit Rate: {target_hit_rate:.3f} ({target_hit_rate*100:.1f}%) - {target_hits} trades")
            logging.info(f"Stop Loss Rate: {stop_loss_rate:.3f} ({stop_loss_rate*100:.1f}%) - {stop_loss_hits} trades")
            logging.info(f"Expiry Rate: {expiry_rate:.3f} ({expiry_rate*100:.1f}%) - {expired_trades} trades")
        
        # Per-scenario analysis
        logging.info(f"\nPER-SCENARIO PERFORMANCE:")
        logging.info(f"{'Scenario':<35} {'Signals':<8} {'Trades':<7} {'Avg Return':<12} {'Win Rate':<9}")
        logging.info("-" * 75)
        
        for result in sorted(self.results, key=lambda x: x.test_date):
            scenario_return = result.avg_return if result.completed_trades else 0
            scenario_win_rate = result.win_rate if result.completed_trades else 0
            
            logging.info(f"{result.market_scenario[:33]:<35} {result.total_signals:<8} "
                        f"{len(result.completed_trades):<7} {scenario_return*100:>9.2f}% "
                        f"{scenario_win_rate*100:>6.1f}%")
        
        # Market scenario grouping
        scenario_stats = {}
        for result in self.results:
            scenario = result.market_scenario
            if scenario not in scenario_stats:
                scenario_stats[scenario] = {
                    'returns': [],
                    'total_trades': 0,
                    'total_signals': 0
                }
            
            scenario_stats[scenario]['returns'].extend([t.applied_return for t in result.completed_trades])
            scenario_stats[scenario]['total_trades'] += len(result.completed_trades)
            scenario_stats[scenario]['total_signals'] += result.total_signals
        
        # Summary dictionary
        summary = {
            'total_periods': len(self.results),
            'total_signals': total_signals,
            'total_active_signals': total_active_signals,
            'total_hold_signals': total_hold_signals,
            'total_completed_trades': len(all_trades),
            'avg_return_per_trade': avg_return,
            'total_return': total_return,
            'win_rate': win_rate,
            'direction_accuracy': direction_accuracy,
            'target_hit_rate': target_hit_rate,
            'stop_loss_rate': stop_loss_rate,
            'expiry_rate': expiry_rate,
            'avg_days_held': avg_days_held,
            'scenario_stats': scenario_stats
        }
        
        return summary
    
    def save_results_to_csv(self, 
                           trades_filename: str = "daily_trading_results.csv",
                           signals_filename: str = "daily_signals_results.csv"):
        """Save detailed results to CSV files"""
        if not self.results:
            logging.warning("No results to save")
            return
        
        try:
            # Save all completed trades
            trades_data = []
            for result in self.results:
                for trade in result.completed_trades:
                    trades_data.append({
                        'benchmark_date': result.test_date,
                        'market_scenario': result.market_scenario,
                        'training_start': result.training_start,
                        'training_end': result.training_end,
                        'signal_date': trade.signal_date,
                        'position_type': trade.position_type,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'exit_date': trade.exit_date,
                        'days_held': trade.days_held,
                        'predicted_return': trade.predicted_return,
                        'actual_return': trade.actual_return,
                        'applied_return': trade.applied_return,
                        'target_price': trade.target_price,
                        'stop_loss_price': trade.stop_loss_price,
                        'exit_reason': trade.exit_reason,
                        'hit_target': trade.hit_target,
                        'hit_stop_loss': trade.hit_stop_loss,
                        'confidence': trade.confidence,
                        'dynamic_stop_loss_pct': trade.dynamic_stop_loss,
                        'market_volatility': trade.market_volatility,
                        'trend_strength': trade.trend_strength
                    })
            
            if trades_data:
                trades_df = pd.DataFrame(trades_data)
                trades_df.to_csv(trades_filename, index=False)
                logging.info(f"Trade results saved to {trades_filename}")
            
            # Save all daily signals (including HOLD)
            signals_data = []
            for result in self.results:
                for signal in result.daily_signals:
                    signals_data.append({
                        'benchmark_date': result.test_date,
                        'market_scenario': result.market_scenario,
                        'signal_date': signal.signal_date,
                        'position_type': signal.position_type,
                        'predicted_return': signal.predicted_return,
                        'confidence': signal.confidence,
                        'entry_price': signal.entry_price,
                        'target_price': signal.target_price,
                        'stop_loss_price': signal.stop_loss_price,
                        'dynamic_stop_loss_pct': signal.dynamic_stop_loss,
                        'market_volatility': signal.market_volatility,
                        'trend_strength': signal.trend_strength
                    })
            
            if signals_data:
                signals_df = pd.DataFrame(signals_data)
                signals_df.to_csv(signals_filename, index=False)
                logging.info(f"Signal results saved to {signals_filename}")
            
        except Exception as e:
            logging.error(f"Error saving results to CSV: {e}")

def main():
    """Main execution function"""
    try:
        # Initialize simulator
        simulator = TradingBenchmarkSimulator()
        
        # Run benchmark tests
        results = simulator.run_benchmark()
        
        if results:
            # Analyze results
            summary = simulator.analyze_results()
            
            # Save results
            simulator.save_results_to_csv()
            
            logging.info("\n" + "="*80)
            logging.info("BENCHMARK SIMULATION COMPLETED SUCCESSFULLY")
            logging.info("="*80)
            
        else:
            logging.error("No benchmark results generated")
            
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()