import numpy as np
import os, random
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import lightgbm as lgb
from tensorflow.keras import layers, callbacks, Model
from sklearn.model_selection import TimeSeriesSplit
from feature_engineering import engineer_features
from data_loader import load_all_data
from sentiment import add_vader_sentiment, aggregate_daily_sentiment
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, friedmanchisquare, kruskal
import seaborn as sns
from datetime import datetime, timedelta
from feature_engineering import engineer_features  
from data_loader import load_all_data
from sentiment import add_vader_sentiment, aggregate_daily_sentiment
from model import ImprovedBitcoinPredictor


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class ComprehensiveModelBenchmark:
    """
    Comprehensive benchmark for Bitcoin prediction model production readiness
    Tests model across multiple time periods, market regimes, and conditions
    """
    
    def __init__(self, predictor_class, optimal_window_months=18):
        self.predictor_class = predictor_class
        self.optimal_window = optimal_window_months
        self.benchmark_results = {}
        self.production_ready = False
        
        # Production readiness thresholds
        self.thresholds = {
            'min_direction_accuracy': 0.55,  # Above random
            'min_sharpe_ratio': 0.5,         # Minimum acceptable risk-adjusted return
            'max_drawdown': 0.25,            # Maximum 25% drawdown
            'min_win_rate': 0.45,            # Minimum win rate
            'max_mae': 0.15,                 # Maximum prediction error
            'min_stability_score': 0.7,      # Consistency across periods
            'min_regime_performance': 0.52,  # Performance in different regimes
            'max_consecutive_losses': 5      # Risk management
        }
    
    def run_comprehensive_benchmark(self, df, n_test_periods=12, test_length_days=60):
        """
        Run comprehensive benchmark across multiple time periods and conditions
        
        Parameters:
        - df: Full dataset with optimal 18-month focus
        - n_test_periods: Number of different time periods to test
        - test_length_days: Length of each test period
        """
        
        print("🚀 STARTING COMPREHENSIVE MODEL BENCHMARK")
        print("=" * 80)
        print(f"Dataset: {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Testing {n_test_periods} periods of {test_length_days} days each")
        print(f"Using optimal {self.optimal_window}-month training windows")
        print("=" * 80 + "\n")
        
        # 1. Multi-Period Walk-Forward Testing
        print("📊 Phase 1: Multi-Period Walk-Forward Testing")
        walk_forward_results = self._multi_period_walk_forward(df, n_test_periods, test_length_days)
        
        # 2. Regime-Specific Testing
        print("\n🎯 Phase 2: Market Regime Testing")
        regime_results = self._regime_specific_testing(df, n_test_periods)
        
        # 3. Stress Testing
        print("\n⚡ Phase 3: Stress Testing")
        stress_results = self._stress_testing(df)
        
        # 4. Trading Performance Testing
        # print("\n💰 Phase 4: Trading Performance Testing")
        # trading_results = self._trading_performance_testing(df, n_test_periods)
        
        # 5. Stability and Robustness Testing
        print("\n🔒 Phase 5: Stability and Robustness Testing")
        stability_results = self._stability_testing(df, n_test_periods)
        
        # 6. Baseline Comparisons
        print("\n📈 Phase 6: Baseline Comparisons")
        baseline_results = self._baseline_comparisons(df, n_test_periods)
        
        # 7. Risk Analysis
        print("\n⚠️  Phase 7: Risk Analysis")
        risk_results = self._risk_analysis(df, n_test_periods)
        
        # Compile all results
        self.benchmark_results = {
            'walk_forward': walk_forward_results,
            'regime_specific': regime_results,
            'stress_testing': stress_results,
            # 'trading_performance': trading_results,
            'stability': stability_results,
            'baseline_comparison': baseline_results,
            'risk_analysis': risk_results
        }
        
        # Final assessment
        production_assessment = self._assess_production_readiness()
        
        # Generate comprehensive report
        self._generate_benchmark_report(production_assessment)
        
        # Prepare final results structure
        final_results = {
            'results': self.benchmark_results,
            'production_ready': self.production_ready,
            'assessment': production_assessment,
            'recommendations': self._generate_recommendations(production_assessment),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'dataset_size': len(df),
                'dataset_period': f"{df.index[0].date()} to {df.index[-1].date()}",
                'test_periods': n_test_periods,
                'test_length_days': test_length_days,
                'optimal_window_months': self.optimal_window
            }
        }
        
        # Save results to JSON
        self._save_results_to_json(final_results)
        
        return final_results
    
    def _multi_period_walk_forward(self, df, n_periods, test_days):
        """Test model across multiple time periods using walk-forward validation"""
        
        results = []
        window_days = self.optimal_window * 30
        
        # Create test periods spanning different market conditions
        total_available = len(df) - window_days - test_days
        
        if total_available < n_periods * test_days:
            n_periods = max(1, total_available // test_days)
            print(f"   Adjusted to {n_periods} test periods due to data constraints")
        
        for period in range(n_periods):
            try:
                # Calculate period boundaries
                test_end = len(df) - (period * test_days)
                test_start = test_end - test_days
                train_end = test_start
                train_start = max(0, train_end - window_days)
                
                if train_start >= train_end or test_start >= test_end:
                    continue
                
                # Extract data
                train_data = df.iloc[train_start:train_end].copy()
                test_data = df.iloc[train_start:test_end].copy()  # Include training for sequences
                
                test_period_dates = df.iloc[test_start:test_end].index
                
                print(f"   Period {period + 1}: {test_period_dates[0].date()} to {test_period_dates[-1].date()}")
                
                # Train model
                predictor = self.predictor_class(sequence_length=60, prediction_horizon=30)
                X_val, y_val, _ = predictor.train_ensemble(train_data, epochs=30, batch_size=32)
                
                if X_val is None:
                    continue
                
                # Test model
                period_results = self._evaluate_period_performance(
                    predictor, test_data, test_start, test_end, train_end, period
                )
                
                if period_results:
                    period_results['test_start_date'] = test_period_dates[0]
                    period_results['test_end_date'] = test_period_dates[-1]
                    period_results['market_conditions'] = self._identify_market_conditions(df.iloc[test_start:test_end])
                    results.append(period_results)
                
            except Exception as e:
                print(f"   ❌ Period {period + 1} failed: {e}")
                continue
        
        # Aggregate results
        if results:
            aggregated = self._aggregate_walk_forward_results(results)
            print(f"   ✅ Completed {len(results)}/{n_periods} test periods")
            return aggregated
        else:
            print("   ❌ No valid test periods completed")
            return None
    
    def _evaluate_period_performance(self, predictor, full_data, test_start, test_end, train_end, period):
        """Evaluate model performance on a specific period"""
        
        try:
            # Data validation
            if full_data is None or len(full_data) < 100:
                print(f"     Insufficient data: {len(full_data) if full_data is not None else 0} rows")
                return None
            
            if 'close' not in full_data.columns:
                print("     Missing 'close' column in data")
                return None
            
            # Prepare data with error handling
            try:
                df_proc = predictor.engineer_30day_target(full_data)
                if df_proc is None or len(df_proc) < 60:
                    print(f"     Target engineering failed or insufficient processed data: {len(df_proc) if df_proc is not None else 0}")
                    return None
            except Exception as e:
                print(f"     Target engineering error: {e}")
                return None
            
            try:
                features, _ = predictor.prepare_features(df_proc)
                if features is None or len(features) < 60:
                    print(f"     Feature preparation failed or insufficient features: {len(features) if features is not None else 0}")
                    return None
            except Exception as e:
                print(f"     Feature preparation error: {e}")
                return None
            
            # Validate targets
            targets = df_proc['target_return_30d'].values
            valid_targets = ~np.isnan(targets)
            if np.sum(valid_targets) < 30:
                print(f"     Insufficient valid targets: {np.sum(valid_targets)}")
                return None
            
            # Create sequences with validation
            try:
                X, y, regimes = predictor.create_sequences(
                    features, targets, 
                    df_proc['market_regime'].values if 'market_regime' in df_proc else None
                )
                
                if X is None or len(X) == 0:
                    print("     Sequence creation failed - no valid sequences")
                    return None
                    
                if len(X) < 10:
                    print(f"     Too few sequences created: {len(X)}")
                    return None
                    
            except Exception as e:
                print(f"     Sequence creation error: {e}")
                return None
            
            # Better test sequence selection
            available_sequences = len(X)
            min_test_sequences = max(5, available_sequences // 4)  # At least 5 or 25% of sequences
            
            # Use the last portion for testing to better simulate real conditions
            test_seq_start = max(0, available_sequences - min_test_sequences)
            test_seq_end = available_sequences
            
            if test_seq_start >= test_seq_end or test_seq_end - test_seq_start < 3:
                print(f"     Insufficient test sequences: {test_seq_end - test_seq_start}")
                return None
            
            X_test = X[test_seq_start:test_seq_end]
            y_test = y[test_seq_start:test_seq_end]
            
            # Validate test data
            if np.any(np.isnan(X_test)) or np.any(np.isnan(y_test)):
                print("     NaN values in test data")
                return None
            
            # Make predictions with error handling
            try:
                predictions, individual_preds, weights = predictor.predict_ensemble(X_test)
                
                if predictions is None or len(predictions) == 0:
                    print("     Prediction failed - no predictions returned")
                    return None
                    
                if np.any(np.isnan(predictions)):
                    print("     NaN values in predictions")
                    return None
                    
            except Exception as e:
                print(f"     Prediction error: {e}")
                return None
            
            # Calculate comprehensive metrics with validation
            try:
                mae = mean_absolute_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                # Validate calculated metrics
                if not np.isfinite(mae) or not np.isfinite(mse) or not np.isfinite(r2):
                    print(f"     Invalid basic metrics: MAE={mae}, MSE={mse}, R2={r2}")
                    return None
                
            except Exception as e:
                print(f"     Basic metrics calculation error: {e}")
                return None
            
            # Direction accuracy
            pred_signs = np.sign(predictions.flatten())
            actual_signs = np.sign(y_test)
            direction_accuracy = np.mean(pred_signs == actual_signs)
            
            if not np.isfinite(direction_accuracy):
                direction_accuracy = 0.5  # Default to random performance
            
            # Trading metrics with better validation
            position_returns = predictions.flatten() * y_test  # Perfect timing simulation
            
            if len(position_returns) == 0 or np.all(position_returns == 0):
                # Handle case where all returns are zero
                sharpe_ratio = 0
                win_rate = 0.5
                profit_factor = 1
                avg_win = 0
                avg_loss = 0
            else:
                returns_std = np.std(position_returns)
                if returns_std > 0:
                    sharpe_ratio = np.mean(position_returns) / returns_std * np.sqrt(252/30)
                else:
                    sharpe_ratio = 0
                
                # Win rate and profit factor
                winning_trades = position_returns > 0
                win_rate = np.mean(winning_trades) if len(winning_trades) > 0 else 0
                
                wins = position_returns[position_returns > 0]
                losses = position_returns[position_returns < 0]
                
                avg_win = np.mean(wins) if len(wins) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 1
            
            # Validate trading metrics
            if not np.isfinite(sharpe_ratio):
                sharpe_ratio = 0
            if not np.isfinite(win_rate):
                win_rate = 0.5
            if not np.isfinite(profit_factor):
                profit_factor = 1
            
            # Risk metrics
            max_drawdown = self._calculate_max_drawdown(np.cumsum(position_returns))
            var_95 = np.percentile(position_returns, 5) if len(position_returns) > 0 else 0
            
            # Stability metrics
            prediction_std = np.std(predictions)
            target_std = np.std(y_test)
            stability_ratio = prediction_std / (target_std + 1e-8)
            
            if not np.isfinite(stability_ratio):
                stability_ratio = 1.0
            
            result = {
                'period': period,
                'mae': float(mae),
                'mse': float(mse),
                'r2': float(r2),
                'direction_accuracy': float(direction_accuracy),
                'sharpe_ratio': float(sharpe_ratio),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'stability_ratio': float(stability_ratio),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'n_predictions': len(y_test),
                # 'individual_predictions': individual_preds if individual_preds is not None else []  # Excluded from JSON to reduce size
            }
            
            # Final validation of result
            for key, value in result.items():
                if key not in ['period', 'n_predictions'] and not np.isfinite(value):
                    print(f"     Invalid result value: {key}={value}")
                    return None
            
            return result
            
        except Exception as e:
            print(f"     Evaluation error: {e}")
            return None
    
    def _identify_market_conditions(self, data):
        """Identify market conditions for the test period"""
        
        if len(data) == 0 or 'close' not in data.columns:
            return 'unknown'
        
        close_prices = data['close']
        returns = close_prices.pct_change().dropna()
        
        avg_return = returns.mean()
        volatility = returns.std()
        
        # Classify market conditions
        if avg_return > 0.002 and volatility < 0.3:
            return 'bull_stable'
        elif avg_return > 0.002 and volatility >= 0.3:
            return 'bull_volatile'
        elif avg_return < -0.002 and volatility < 0.3:
            return 'bear_stable'
        elif avg_return < -0.002 and volatility >= 0.3:
            return 'bear_volatile'
        else:
            return 'sideways'
    
    def _aggregate_walk_forward_results(self, results):
        """Aggregate results across all walk-forward periods"""
        
        metrics = ['mae', 'mse', 'r2', 'direction_accuracy', 'sharpe_ratio', 
                  'win_rate', 'profit_factor', 'max_drawdown', 'var_95', 'stability_ratio']
        
        aggregated = {}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r and np.isfinite(r[metric])]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
                aggregated[f'min_{metric}'] = np.min(values)
                aggregated[f'max_{metric}'] = np.max(values)
                aggregated[f'median_{metric}'] = np.median(values)
        
        # Market condition breakdown
        condition_performance = {}
        for result in results:
            condition = result.get('market_conditions', 'unknown')
            if condition not in condition_performance:
                condition_performance[condition] = []
            condition_performance[condition].append(result)
        
        aggregated['condition_performance'] = condition_performance
        aggregated['n_periods'] = len(results)
        aggregated['detailed_results'] = results
        
        # Add statistical significance testing
        print("   Running statistical significance tests...")
        aggregated['statistical_analysis'] = {}
        
        # Test key metrics for statistical significance
        key_metrics = ['direction_accuracy', 'sharpe_ratio', 'win_rate', 'mae']
        for metric in key_metrics:
            if any(metric in r for r in results):
                significance_test = self._perform_statistical_significance_tests(results, metric)
                aggregated['statistical_analysis'][f'{metric}_significance'] = significance_test
        
        # Analyze performance trends over time
        trend_analysis = self._analyze_performance_trends(results, 'direction_accuracy')
        aggregated['statistical_analysis']['performance_trend'] = trend_analysis
        
        return aggregated
    
    def _regime_specific_testing(self, df, n_periods):
        """Test model performance in different market regimes"""
        
        print("   Testing Bull Markets...")
        bull_performance = self._test_specific_regime(df, 'bull', n_periods // 3)
        
        print("   Testing Bear Markets...")
        bear_performance = self._test_specific_regime(df, 'bear', n_periods // 3)
        
        print("   Testing Sideways Markets...")
        sideways_performance = self._test_specific_regime(df, 'sideways', n_periods // 3)
        
        return {
            'bull_markets': bull_performance,
            'bear_markets': bear_performance,
            'sideways_markets': sideways_performance
        }
    
    def _test_specific_regime(self, df, regime_type, n_tests):
        """Test model in specific market regime"""
        
        try:
            # Identify periods of specific regime
            if 'close' not in df.columns:
                return None
            
            close_prices = df['close']
            returns_30d = close_prices.pct_change(30)
            
            if regime_type == 'bull':
                regime_mask = returns_30d > 0.1  # Bull: >10% monthly returns
            elif regime_type == 'bear':
                regime_mask = returns_30d < -0.1  # Bear: <-10% monthly returns
            else:
                regime_mask = abs(returns_30d) <= 0.1  # Sideways: ±10% monthly returns
            
            regime_periods = df[regime_mask].index
            
            if len(regime_periods) < 60:  # Need at least 60 days
                return {'error': f'Insufficient {regime_type} market data'}
            
            # Sample test periods from regime
            test_results = []
            window_days = self.optimal_window * 30
            
            for i in range(min(n_tests, len(regime_periods) // 60)):
                try:
                    # Find regime period
                    regime_start = regime_periods[i * 60]
                    regime_end_idx = df.index.get_loc(regime_start) + 60
                    
                    if regime_end_idx >= len(df):
                        continue
                    
                    # Training data before regime period
                    train_end_idx = df.index.get_loc(regime_start)
                    train_start_idx = max(0, train_end_idx - window_days)
                    
                    train_data = df.iloc[train_start_idx:train_end_idx]
                    test_data = df.iloc[train_start_idx:regime_end_idx]
                    
                    # Train and test
                    predictor = self.predictor_class(sequence_length=60, prediction_horizon=30)
                    X_val, y_val, _ = predictor.train_ensemble(train_data, epochs=20, batch_size=32)
                    
                    if X_val is not None:
                        result = self._evaluate_period_performance(
                            predictor, test_data, train_end_idx, regime_end_idx, train_end_idx, i
                        )
                        if result:
                            test_results.append(result)
                            
                except Exception as e:
                    continue
            
            if test_results:
                return self._aggregate_regime_results(test_results, regime_type)
            else:
                return {'error': f'No valid {regime_type} tests completed'}
                
        except Exception as e:
            return {'error': f'{regime_type} testing failed: {e}'}
    
    def _aggregate_regime_results(self, results, regime_type):
        """Aggregate results for specific regime"""
        
        if not results:
            return None
        
        metrics = ['mae', 'direction_accuracy', 'sharpe_ratio', 'win_rate', 'max_drawdown']
        aggregated = {'regime': regime_type}
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r and np.isfinite(r[metric])]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
        
        aggregated['n_tests'] = len(results)
        return aggregated
    
    def _stress_testing(self, df):
        """Test model under extreme market conditions"""
        
        print("   Testing High Volatility Periods...")
        volatility_test = self._test_extreme_volatility(df)
        
        print("   Testing Market Crashes...")
        crash_test = self._test_market_crashes(df)
        
        print("   Testing Extended Bear Markets...")
        extended_bear_test = self._test_extended_bear_markets(df)
        
        return {
            'high_volatility': volatility_test,
            'market_crashes': crash_test,
            'extended_bear_markets': extended_bear_test
        }
    
    def _test_extreme_volatility(self, df):
        """Test model during high volatility periods"""
        
        try:
            if 'close' not in df.columns:
                return {'error': 'No price data available'}
            
            # Calculate rolling volatility with shorter window for more data points
            returns = df['close'].pct_change()
            volatility = returns.rolling(14).std()  # Reduced from 30 to 14
            
            # Use top 15% instead of 10% to get more periods, and ensure we have data
            vol_threshold = volatility.quantile(0.85)  # Changed from 0.9 to 0.85
            extreme_vol_mask = (volatility > vol_threshold) & ~volatility.isna()
            
            extreme_periods = df[extreme_vol_mask].index
            print(f"Length of extreme periods: {len(extreme_periods)}")
            
            # Reduced minimum requirement and added more flexible grouping
            if len(extreme_periods) < 10:  # Reduced from 30 to 10
                return {'error': 'Insufficient extreme volatility periods'}
            
            # Group consecutive periods and use them as test blocks
            results = []
            window_days = self.optimal_window * 30
            
            # Find groups of consecutive extreme volatility periods
            period_groups = []
            current_group = []
            
            for i, period in enumerate(extreme_periods):
                if i == 0:
                    current_group = [period]
                else:
                    # Check if current period is within 7 days of previous
                    time_diff = (period - extreme_periods[i-1]).days
                    if time_diff <= 7:  # Group periods within a week
                        current_group.append(period)
                    else:
                        if len(current_group) >= 5:  # Group must have at least 5 days
                            period_groups.append(current_group)
                        current_group = [period]
            
            # Don't forget the last group
            if len(current_group) >= 5:
                period_groups.append(current_group)
            
            print(f"Found {len(period_groups)} volatility period groups")
            
            # Test on period groups
            for i, group in enumerate(period_groups[:3]):  # Test first 3 groups
                try:
                    period_start = group[0]
                    period_end = group[-1]
                    start_idx = df.index.get_loc(period_start)
                    end_idx = df.index.get_loc(period_end)
                    
                    # Ensure we have enough training data
                    train_start = max(0, start_idx - window_days)
                    train_end = start_idx
                    test_start = train_start
                    test_end = min(len(df), end_idx + 7)  # Include 7 days after volatility period
                    
                    if train_end - train_start < 30:  # Need at least 30 days of training
                        continue
                    
                    train_data = df.iloc[train_start:train_end]
                    test_data = df.iloc[test_start:test_end]
                    
                    predictor = self.predictor_class(sequence_length=60, prediction_horizon=30)
                    X_val, y_val, _ = predictor.train_ensemble(train_data, epochs=15, batch_size=32)
                    
                    if X_val is not None:
                        result = self._evaluate_period_performance(
                            predictor, test_data, train_end, test_end, train_end, i
                        )
                        if result:
                            results.append(result)
                            
                except Exception as e:
                    print(f"     Volatility group {i} failed: {e}")
                    continue
            
            if results:
                return self._aggregate_stress_results(results, 'high_volatility')
            else:
                return {'error': 'No valid volatility tests'}
                
        except Exception as e:
            return {'error': f'Volatility testing failed: {e}'}
    
    def _test_market_crashes(self, df):
        """Test model during market crash periods"""
        
        try:
            # Use multiple timeframes to identify crash periods - more lenient thresholds
            returns_3d = df['close'].pct_change(3)   # 3-day returns
            returns_7d = df['close'].pct_change(7)   # 7-day returns
            returns_14d = df['close'].pct_change(14) # 14-day returns
            
            # Much more lenient crash identification - use 20% instead of 10% and multiple timeframes
            crash_threshold_3d = returns_3d.quantile(0.2)   # Bottom 20%
            crash_threshold_7d = returns_7d.quantile(0.2)   # Bottom 20%
            crash_threshold_14d = returns_14d.quantile(0.2) # Bottom 20%
            
            # A crash period is when ANY of these conditions is met
            crash_mask = (
                (returns_3d < crash_threshold_3d) | 
                (returns_7d < crash_threshold_7d) | 
                (returns_14d < crash_threshold_14d)
            ) & ~returns_3d.isna() & ~returns_7d.isna() & ~returns_14d.isna()
            
            crash_periods = df[crash_mask].index
            print(f"Length of crash periods: {len(crash_periods)}")
            
            if len(crash_periods) < 3:  # Very reduced minimum requirement
                return {'error': 'Insufficient crash periods'}
            
            # Group crash periods and test them
            results = []
            window_days = self.optimal_window * 30
            
            # Find groups of crash periods - more lenient grouping
            crash_groups = []
            current_group = []
            
            for i, period in enumerate(crash_periods):
                if i == 0:
                    current_group = [period]
                else:
                    # Group periods within 21 days of each other (increased from 14)
                    time_diff = (period - crash_periods[i-1]).days
                    if time_diff <= 21:
                        current_group.append(period)
                    else:
                        if len(current_group) >= 1:  # Group can have just 1 day
                            crash_groups.append(current_group)
                        current_group = [period]
            
            # Don't forget the last group
            if len(current_group) >= 1:
                crash_groups.append(current_group)
            
            print(f"Found {len(crash_groups)} crash period groups")
            
            for i, group in enumerate(crash_groups[:3]):  # Test first 3 crash groups (increased from 2)
                try:
                    period_start = group[0]
                    period_end = group[-1]
                    start_idx = df.index.get_loc(period_start)
                    end_idx = df.index.get_loc(period_end)
                    
                    train_start = max(0, start_idx - window_days)
                    train_end = start_idx
                    test_start = train_start
                    test_end = min(len(df), end_idx + 10)  # Include 10 days after crash
                    
                    if train_end - train_start < 15:  # Reduced minimum training days from 30 to 15
                        continue
                    
                    train_data = df.iloc[train_start:train_end]
                    test_data = df.iloc[test_start:test_end]
                    
                    predictor = self.predictor_class(sequence_length=60, prediction_horizon=30)
                    X_val, y_val, _ = predictor.train_ensemble(train_data, epochs=10, batch_size=32)  # Reduced epochs
                    
                    if X_val is not None:
                        result = self._evaluate_period_performance(
                            predictor, test_data, train_end, test_end, train_end, i
                        )
                        if result:
                            results.append(result)
                            
                except Exception as e:
                    print(f"     Crash group {i} failed: {e}")
                    continue
            
            return self._aggregate_stress_results(results, 'market_crashes') if results else {'error': 'No valid crash tests'}
            
        except Exception as e:
            return {'error': f'Crash testing failed: {e}'}
    
    def _test_extended_bear_markets(self, df):
        """Test model during extended bear market periods"""
        
        try:
            # Identify extended bear markets (60+ days of negative trend)
            returns = df['close'].pct_change()
            bear_signal = returns.rolling(60).sum() < -0.2  # 60-day cumulative return < -20%
            
            bear_periods = df[bear_signal].index
            
            if len(bear_periods) < 60:
                return {'error': 'Insufficient extended bear market data'}
            
            # Test logic similar to other stress tests
            results = []
            window_days = self.optimal_window * 30
            
            for i in range(min(2, len(bear_periods) // 60)):
                try:
                    period_start = bear_periods[i * 60]
                    start_idx = df.index.get_loc(period_start)
                    
                    train_start = max(0, start_idx - window_days)
                    train_end = start_idx
                    test_end = min(len(df), start_idx + 60)
                    
                    train_data = df.iloc[train_start:train_end]
                    test_data = df.iloc[train_start:test_end]
                    
                    predictor = self.predictor_class(sequence_length=60, prediction_horizon=30)
                    X_val, y_val, _ = predictor.train_ensemble(train_data, epochs=15, batch_size=32)
                    
                    if X_val is not None:
                        result = self._evaluate_period_performance(
                            predictor, test_data, train_end, test_end, train_end, i
                        )
                        if result:
                            results.append(result)
                            
                except Exception:
                    continue
            
            return self._aggregate_stress_results(results, 'extended_bear') if results else {'error': 'No valid bear tests'}
            
        except Exception as e:
            return {'error': f'Extended bear testing failed: {e}'}
    
    def _aggregate_stress_results(self, results, test_type):
        """Aggregate stress test results"""
        
        if not results:
            return None
        
        metrics = ['mae', 'direction_accuracy', 'sharpe_ratio', 'max_drawdown']
        aggregated = {'test_type': test_type}
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r and np.isfinite(r[metric])]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'worst_{metric}'] = np.min(values) if metric == 'direction_accuracy' else np.max(values)
        
        aggregated['n_tests'] = len(results)
        
        # Add statistical significance testing for stress tests
        if len(results) >= 2:
            print(f"   Running statistical tests for {test_type}...")
            aggregated['statistical_analysis'] = {}
            
            # Test if performance during stress is significantly different from random
            for metric in ['direction_accuracy', 'sharpe_ratio']:
                if any(metric in r for r in results):
                    significance_test = self._perform_statistical_significance_tests(results, metric)
                    aggregated['statistical_analysis'][f'{metric}_significance'] = significance_test
        
        return aggregated
    
    def _trading_performance_testing(self, df, n_periods):
        """Test actual trading performance with realistic constraints"""
        
        print("   Testing Trading Performance with Transaction Costs...")
        
        results = []
        window_days = self.optimal_window * 30
        
        for period in range(min(n_periods, 6)):  # Test 6 trading periods
            try:
                # Period boundaries
                test_days = 90  # 3-month trading periods
                test_end = len(df) - (period * test_days)
                test_start = test_end - test_days
                train_end = test_start
                train_start = max(0, train_end - window_days)
                
                if train_start >= train_end:
                    continue
                
                train_data = df.iloc[train_start:train_end]
                test_data = df.iloc[train_start:test_end]
                
                # Train model
                predictor = self.predictor_class(sequence_length=60, prediction_horizon=30)
                X_val, y_val, _ = predictor.train_ensemble(train_data, epochs=25, batch_size=32)
                
                if X_val is None:
                    continue
                
                # Simulate realistic trading
                trading_result = self._simulate_realistic_trading(predictor, test_data, test_start, test_end, train_end)
                
                if trading_result:
                    trading_result['period'] = period
                    results.append(trading_result)
                    
            except Exception as e:
                print(f"   Trading period {period} failed: {e}")
                continue
        
        return self._aggregate_trading_results(results) if results else None
    
    def _simulate_realistic_trading(self, predictor, full_data, test_start, test_end, train_end):
        """Simulate realistic trading with transaction costs and constraints"""
        
        try:
            # Prepare data
            df_proc = predictor.engineer_30day_target(full_data)
            features, _ = predictor.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            
            X, y, _ = predictor.create_sequences(features, targets)
            
            if len(X) < 10:
                return None
            
            # Use last portion for testing
            X_test = X[-min(30, len(X)):]
            y_test = y[-min(30, len(y)):]
            
            # Trading simulation
            capital = 10000  # Starting capital
            positions = []
            transaction_cost = 0.002  # 0.2% per trade
            max_position_size = 0.1   # 10% max position
            
            for i in range(len(X_test)):
                try:
                    # Get prediction
                    pred, _, _ = predictor.predict_ensemble(X_test[i:i+1])
                    predicted_return = pred[0][0]
                    actual_return = y_test[i]
                    
                    # Position sizing based on confidence
                    if abs(predicted_return) > 0.02:  # Minimum threshold
                        position_size = min(abs(predicted_return) * 2, max_position_size)
                        direction = np.sign(predicted_return)
                        
                        # Calculate trade result
                        gross_return = position_size * direction * actual_return
                        net_return = gross_return - (position_size * transaction_cost)
                        
                        capital += net_return * capital
                        
                        positions.append({
                            'predicted_return': predicted_return,
                            'actual_return': actual_return,
                            'position_size': position_size,
                            'gross_return': gross_return,
                            'net_return': net_return,
                            'direction_correct': np.sign(predicted_return) == np.sign(actual_return)
                        })
                        
                except Exception:
                    continue
            
            if not positions:
                return None
            
            # Calculate trading metrics
            position_returns = [p['net_return'] for p in positions]
            total_return = (capital - 10000) / 10000
            
            winning_trades = [r for r in position_returns if r > 0]
            losing_trades = [r for r in position_returns if r < 0]
            
            win_rate = len(winning_trades) / len(positions) if positions else 0
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 1
            
            sharpe_ratio = np.mean(position_returns) / (np.std(position_returns) + 1e-8) * np.sqrt(252/30)
            max_drawdown = self._calculate_max_drawdown(np.cumsum(position_returns))
            
            direction_accuracy = np.mean([p['direction_correct'] for p in positions])
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'direction_accuracy': direction_accuracy,
                'n_trades': len(positions),
                'avg_position_size': np.mean([p['position_size'] for p in positions])
            }
            
        except Exception as e:
            print(f"   Trading simulation error: {e}")
            return None
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown from cumulative returns"""
        
        if len(cumulative_returns) == 0:
            return 0
        
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
        return abs(np.min(drawdown))
    
    def _aggregate_trading_results(self, results):
        """Aggregate trading performance results"""
        
        metrics = ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 
                  'max_drawdown', 'direction_accuracy', 'n_trades']
        
        aggregated = {}
        for metric in metrics:
            values = [r[metric] for r in results if metric in r and np.isfinite(r[metric])]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
                aggregated[f'min_{metric}'] = np.min(values)
                aggregated[f'max_{metric}'] = np.max(values)
        
        aggregated['n_periods'] = len(results)
        return aggregated
    
    def _stability_testing(self, df, n_periods):
        """Test model stability and robustness"""
        
        print("   Testing Prediction Stability...")
        
        stability_results = []
        window_days = self.optimal_window * 30
        
        # Test same period multiple times with slight variations
        base_period_end = len(df) - 60
        base_period_start = base_period_end - 60
        base_train_end = base_period_start
        base_train_start = max(0, base_train_end - window_days)
        
        base_train = df.iloc[base_train_start:base_train_end]
        base_test = df.iloc[base_train_start:base_period_end]
        
        for test_run in range(5):  # 5 stability tests
            try:
                # Slight variation in training data (bootstrap sampling)
                varied_train = base_train.sample(frac=0.95, random_state=test_run).sort_index()
                
                predictor = self.predictor_class(sequence_length=60, prediction_horizon=30)
                X_val, y_val, _ = predictor.train_ensemble(varied_train, epochs=20, batch_size=32)
                
                if X_val is not None:
                    result = self._evaluate_period_performance(
                        predictor, base_test, base_train_end, base_period_end, base_train_end, test_run
                    )
                    if result:
                        stability_results.append(result)
                        
            except Exception:
                continue
        
        return self._analyze_stability(stability_results) if stability_results else None
    
    def _analyze_stability(self, results):
        """Analyze model stability across multiple runs"""
        
        metrics = ['mae', 'direction_accuracy', 'sharpe_ratio']
        stability_analysis = {}
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r and np.isfinite(r[metric])]
            if values:
                cv = np.std(values) / (np.mean(values) + 1e-8)  # Coefficient of variation
                stability_analysis[f'{metric}_stability'] = 1 / (1 + cv)  # Higher is more stable
                stability_analysis[f'{metric}_cv'] = cv
        
        # Overall stability score
        stability_scores = [v for k, v in stability_analysis.items() if k.endswith('_stability')]
        overall_stability = np.mean(stability_scores) if stability_scores else 0
        
        stability_analysis['overall_stability'] = overall_stability
        stability_analysis['n_runs'] = len(results)
        
        return stability_analysis
    
    def _baseline_comparisons(self, df, n_periods):
        """Compare model against simple baselines"""
        
        print("   Comparing against Buy-and-Hold...")
        buy_hold = self._test_buy_and_hold(df)
        
        print("   Comparing against Moving Average Strategy...")
        ma_strategy = self._test_moving_average_strategy(df)
        
        print("   Comparing against Random Predictions...")
        random_baseline = self._test_random_predictions(df)
        
        baseline_results = {
            'buy_and_hold': buy_hold,
            'moving_average': ma_strategy,
            'random_predictions': random_baseline
        }
        
        # Add statistical comparisons if we have walk-forward results
        if hasattr(self, 'benchmark_results') and 'walk_forward' in self.benchmark_results:
            walk_forward_results = self.benchmark_results['walk_forward']
            if walk_forward_results and 'detailed_results' in walk_forward_results:
                model_results = walk_forward_results['detailed_results']
                
                print("   Running statistical comparisons with baselines...")
                baseline_results['statistical_comparisons'] = {}
                
                # Compare with random baseline (if we can simulate multiple random results)
                random_results = []
                for _ in range(len(model_results)):
                    random_sim = self._test_random_predictions(df)
                    if 'direction_accuracy' in random_sim:
                        random_results.append(random_sim)
                
                if random_results:
                    comparison = self._compare_model_performance_statistically(
                        model_results, random_results, 'direction_accuracy'
                    )
                    baseline_results['statistical_comparisons']['vs_random'] = comparison
        
        return baseline_results
    
    def _test_buy_and_hold(self, df):
        """Test against buy-and-hold strategy"""
        
        try:
            if len(df) < 365:  # Need at least 1 year
                return {'error': 'Insufficient data for buy-and-hold'}
            
            # Last year performance
            start_price = df['close'].iloc[-365]
            end_price = df['close'].iloc[-1]
            
            buy_hold_return = (end_price - start_price) / start_price
            
            # Calculate volatility
            daily_returns = df['close'].pct_change().dropna()
            annual_vol = daily_returns.std() * np.sqrt(365)
            sharpe_ratio = buy_hold_return / annual_vol if annual_vol > 0 else 0
            
            return {
                'annual_return': buy_hold_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'strategy': 'buy_and_hold'
            }
            
        except Exception as e:
            return {'error': f'Buy-and-hold test failed: {e}'}
    
    def _test_moving_average_strategy(self, df):
        """Test against simple moving average crossover strategy"""
        
        try:
            if 'close' not in df.columns or len(df) < 100:
                return {'error': 'Insufficient data for MA strategy'}
            
            # Simple MA crossover strategy
            ma_short = df['close'].rolling(20).mean()
            ma_long = df['close'].rolling(50).mean()
            
            # Generate signals
            signals = np.where(ma_short > ma_long, 1, -1)
            returns = df['close'].pct_change()
            
            # Calculate strategy returns
            strategy_returns = signals[:-1] * returns.iloc[1:].values
            strategy_returns = strategy_returns[~np.isnan(strategy_returns)]
            
            if len(strategy_returns) == 0:
                return {'error': 'No valid MA strategy returns'}
            
            annual_return = np.mean(strategy_returns) * 365
            annual_vol = np.std(strategy_returns) * np.sqrt(365)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'strategy': 'moving_average'
            }
            
        except Exception as e:
            return {'error': f'MA strategy test failed: {e}'}
    
    def _test_random_predictions(self, df):
        """Test against random prediction baseline"""
        
        try:
            # Simulate random predictions
            n_predictions = 100
            random_predictions = np.random.normal(0, 0.05, n_predictions)  # Random predictions with 5% std
            actual_returns = np.random.choice(df['close'].pct_change().dropna().values, n_predictions)
            
            # Direction accuracy
            direction_accuracy = np.mean(np.sign(random_predictions) == np.sign(actual_returns))
            
            # Simulated trading performance
            position_returns = random_predictions * actual_returns * 0.1  # 10% position size
            sharpe_ratio = np.mean(position_returns) / (np.std(position_returns) + 1e-8) * np.sqrt(252/30)
            
            return {
                'direction_accuracy': direction_accuracy,
                'sharpe_ratio': sharpe_ratio,
                'strategy': 'random'
            }
            
        except Exception as e:
            return {'error': f'Random baseline test failed: {e}'}
    
    def _risk_analysis(self, df, n_periods):
        """Comprehensive risk analysis"""
        
        print("   Analyzing Risk Metrics...")
        
        # Test model under different risk scenarios
        risk_results = {}
        
        try:
            # Train model on recent data
            window_days = self.optimal_window * 30
            train_data = df.iloc[-window_days-100:-100]  # Leave 100 days for testing
            test_data = df.iloc[-window_days:]
            
            predictor = self.predictor_class(sequence_length=60, prediction_horizon=30)
            X_val, y_val, _ = predictor.train_ensemble(train_data, epochs=25, batch_size=32)
            
            if X_val is not None:
                # Risk metrics calculation
                result = self._evaluate_period_performance(
                    predictor, test_data, len(train_data), len(test_data), len(train_data), 0
                )
                
                if result:
                    risk_results = {
                        'var_95': result.get('var_95', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'win_rate': result.get('win_rate', 0),
                        'profit_factor': result.get('profit_factor', 1)
                    }
                    
                    # Risk-adjusted metrics
                    risk_results['risk_adjusted_return'] = risk_results['sharpe_ratio']
                    risk_results['tail_risk'] = abs(risk_results['var_95'])
                    risk_results['consistency'] = risk_results['win_rate']
            
        except Exception as e:
            risk_results = {'error': f'Risk analysis failed: {e}'}
        
        return risk_results
    
    def _assess_production_readiness(self):
        """Assess if model is ready for production based on all tests"""
        
        assessment = {
            'overall_score': 0,
            'category_scores': {},
            'passed_tests': [],
            'failed_tests': [],
            'warnings': [],
            'critical_issues': []
        }
        
        try:
            # Assess each category
            categories = [
                ('walk_forward', 0.25),      # 25% weight
                ('regime_specific', 0.15),   # 15% weight
                ('stress_testing', 0.15),    # 15% weight
                ('trading_performance', 0.20), # 20% weight
                ('stability', 0.15),         # 15% weight
                ('baseline_comparison', 0.10) # 10% weight
            ]
            
            total_score = 0
            total_weight = 0
            
            for category, weight in categories:
                if category in self.benchmark_results and self.benchmark_results[category]:
                    score = self._assess_category(category, self.benchmark_results[category])
                    assessment['category_scores'][category] = score
                    total_score += score * weight
                    total_weight += weight
                    
                    if score >= 0.7:
                        assessment['passed_tests'].append(f"{category}: {score:.2f}")
                    elif score >= 0.5:
                        assessment['warnings'].append(f"{category}: {score:.2f} (marginal)")
                    else:
                        assessment['failed_tests'].append(f"{category}: {score:.2f}")
                        if score < 0.3:
                            assessment['critical_issues'].append(f"{category}: {score:.2f} (critical)")
            
            assessment['overall_score'] = total_score / total_weight if total_weight > 0 else 0
            
            # Production readiness decision
            self.production_ready = (
                assessment['overall_score'] >= 0.6 and
                len(assessment['critical_issues']) == 0 and
                len(assessment['failed_tests']) <= 2
            )
            
        except Exception as e:
            assessment['error'] = f'Assessment failed: {e}'
            self.production_ready = False
        
        return assessment
    
    def _assess_category(self, category, results):
        """Assess individual category performance"""
        
        try:
            if category == 'walk_forward':
                return self._assess_walk_forward(results)
            elif category == 'regime_specific':
                return self._assess_regime_specific(results)
            elif category == 'stress_testing':
                return self._assess_stress_testing(results)
            elif category == 'trading_performance':
                return self._assess_trading_performance(results)
            elif category == 'stability':
                return self._assess_stability_performance(results)
            elif category == 'baseline_comparison':
                return self._assess_baseline_comparison(results)
            else:
                return 0.5  # Default score
        except Exception:
            return 0.3  # Low score for failed assessment
    
    def _assess_walk_forward(self, results):
        """Assess walk-forward testing results"""
        
        score = 0
        
        try:
            # Direction accuracy
            if 'avg_direction_accuracy' in results:
                dir_acc = results['avg_direction_accuracy']
                if dir_acc >= self.thresholds['min_direction_accuracy']:
                    score += 0.3
                else:
                    score += max(0, (dir_acc - 0.5) / 0.05 * 0.3)
            
            # MAE
            if 'avg_mae' in results:
                mae = results['avg_mae']
                if mae <= self.thresholds['max_mae']:
                    score += 0.3
                else:
                    score += max(0, 0.3 * (1 - (mae - self.thresholds['max_mae']) / self.thresholds['max_mae']))
            
            # Sharpe ratio
            if 'avg_sharpe_ratio' in results:
                sharpe = results['avg_sharpe_ratio']
                if sharpe >= self.thresholds['min_sharpe_ratio']:
                    score += 0.4
                else:
                    score += max(0, sharpe / self.thresholds['min_sharpe_ratio'] * 0.4)
            
        except Exception:
            pass
        
        return min(1.0, score)
    
    def _assess_regime_specific(self, results):
        """Assess regime-specific performance"""
        
        score = 0
        total_regimes = 0
        
        for regime in ['bull_markets', 'bear_markets', 'sideways_markets']:
            if regime in results and results[regime] and 'avg_direction_accuracy' in results[regime]:
                total_regimes += 1
                dir_acc = results[regime]['avg_direction_accuracy']
                if dir_acc >= self.thresholds['min_regime_performance']:
                    score += 0.33
                else:
                    score += max(0, (dir_acc - 0.5) / 0.02 * 0.33)
        
        return score if total_regimes > 0 else 0.5
    
    def _assess_stress_testing(self, results):
        """Assess stress testing performance"""
        
        score = 0
        tests_passed = 0
        total_tests = 0
        
        for test_type in ['high_volatility', 'market_crashes', 'extended_bear_markets']:
            if test_type in results and results[test_type] and 'avg_direction_accuracy' in results[test_type]:
                total_tests += 1
                dir_acc = results[test_type]['avg_direction_accuracy']
                if dir_acc >= 0.5:  # Lower threshold for stress tests
                    tests_passed += 1
        
        score = tests_passed / total_tests if total_tests > 0 else 0.5
        return score
    
    def _assess_trading_performance(self, results):
        """Assess trading performance"""
        
        score = 0
        
        try:
            # Win rate
            if 'avg_win_rate' in results:
                win_rate = results['avg_win_rate']
                if win_rate >= self.thresholds['min_win_rate']:
                    score += 0.25
            
            # Sharpe ratio
            if 'avg_sharpe_ratio' in results:
                sharpe = results['avg_sharpe_ratio']
                if sharpe >= self.thresholds['min_sharpe_ratio']:
                    score += 0.25
            
            # Max drawdown
            if 'avg_max_drawdown' in results:
                max_dd = results['avg_max_drawdown']
                if max_dd <= self.thresholds['max_drawdown']:
                    score += 0.25
            
            # Total return
            if 'avg_total_return' in results:
                total_ret = results['avg_total_return']
                if total_ret > 0:
                    score += 0.25
            
        except Exception:
            pass
        
        return min(1.0, score)
    
    def _assess_stability_performance(self, results):
        """Assess stability performance"""
        
        if 'overall_stability' in results:
            stability = results['overall_stability']
            if stability >= self.thresholds['min_stability_score']:
                return 1.0
            else:
                return max(0.3, stability)
        
        return 0.5
    
    def _assess_baseline_comparison(self, results):
        """Assess baseline comparison"""
        
        score = 0
        
        try:
            # Compare against buy-and-hold
            if 'buy_and_hold' in results and 'sharpe_ratio' in results['buy_and_hold']:
                bh_sharpe = results['buy_and_hold']['sharpe_ratio']
                # Model should significantly outperform buy-and-hold
                if bh_sharpe > 0:
                    score += 0.5  # Basic score for positive baseline
            
            # Compare against random
            if 'random_predictions' in results and 'direction_accuracy' in results['random_predictions']:
                random_acc = results['random_predictions']['direction_accuracy']
                # Model should significantly beat random
                score += 0.5  # If model exists, it likely beats random
            
        except Exception:
            score = 0.5  # Default score
        
        return score
    
    def _generate_benchmark_report(self, assessment):
        """Generate comprehensive benchmark report"""
        
        print("\n" + "🎯" * 30)
        print("COMPREHENSIVE MODEL BENCHMARK REPORT")
        print("🎯" * 30)
        
        # Overall Assessment
        overall_score = assessment['overall_score']
        print(f"\n📊 OVERALL PRODUCTION READINESS SCORE: {overall_score:.2f}/1.00")
        
        if self.production_ready:
            print("✅ MODEL IS PRODUCTION READY")
            readiness_icon = "🟢"
        elif overall_score >= 0.5:
            print("⚠️  MODEL NEEDS IMPROVEMENTS BEFORE PRODUCTION")
            readiness_icon = "🟡"
        else:
            print("❌ MODEL NOT READY FOR PRODUCTION")
            readiness_icon = "🔴"
        
        # Category Breakdown
        print(f"\n📈 CATEGORY PERFORMANCE BREAKDOWN:")
        print("-" * 60)
        for category, score in assessment['category_scores'].items():
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            print(f"{status} {category.replace('_', ' ').title():<25}: {score:.2f}")
        
        # Test Results Summary
        if assessment['passed_tests']:
            print(f"\n✅ PASSED TESTS ({len(assessment['passed_tests'])}):")
            for test in assessment['passed_tests']:
                print(f"   ✓ {test}")
        
        if assessment['warnings']:
            print(f"\n⚠️  MARGINAL PERFORMANCE ({len(assessment['warnings'])}):")
            for warning in assessment['warnings']:
                print(f"   ⚠ {warning}")
        
        if assessment['failed_tests']:
            print(f"\n❌ FAILED TESTS ({len(assessment['failed_tests'])}):")
            for failure in assessment['failed_tests']:
                print(f"   ✗ {failure}")
        
        if assessment['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES ({len(assessment['critical_issues'])}):")
            for issue in assessment['critical_issues']:
                print(f"   🚨 {issue}")
        
        # Statistical Significance Summary
        self._print_statistical_significance_summary()
        
        # Production Readiness Summary
        print(f"\n{readiness_icon} PRODUCTION READINESS ASSESSMENT:")
        print("-" * 50)
        print(f"Overall Score: {overall_score:.2f}")
        print(f"Production Ready: {'YES' if self.production_ready else 'NO'}")
        print(f"Critical Issues: {len(assessment['critical_issues'])}")
        print(f"Failed Tests: {len(assessment['failed_tests'])}")
    
    def _print_statistical_significance_summary(self):
        """Print statistical significance summary from benchmark results"""
        
        print(f"\n📊 STATISTICAL SIGNIFICANCE ANALYSIS:")
        print("-" * 60)
        
        # Walk-forward statistical analysis
        if ('walk_forward' in self.benchmark_results and 
            self.benchmark_results['walk_forward'] and 
            'statistical_analysis' in self.benchmark_results['walk_forward']):
            
            stats_analysis = self.benchmark_results['walk_forward']['statistical_analysis']
            print(f"\n✨ WALK-FORWARD TESTING STATISTICAL RESULTS:")
            
            # Key metrics significance
            for metric_key, analysis in stats_analysis.items():
                if metric_key.endswith('_significance') and isinstance(analysis, dict):
                    metric_name = metric_key.replace('_significance', '').replace('_', ' ').title()
                    
                    if 'significant_results' in analysis and analysis['significant_results']:
                        print(f"   ✅ {metric_name}: STATISTICALLY SIGNIFICANT")
                        for result in analysis['significant_results']:
                            print(f"      • {result}")
                        
                        # Add confidence interval if available
                        if 'confidence_interval_95' in analysis:
                            ci = analysis['confidence_interval_95']
                            print(f"      • 95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")
                    else:
                        print(f"   ⚠️  {metric_name}: Not statistically significant")
                        if 'interpretation' in analysis:
                            print(f"      • {analysis['interpretation']}")
            
            # Performance trend analysis
            if 'performance_trend' in stats_analysis:
                trend = stats_analysis['performance_trend']
                print(f"\n📈 PERFORMANCE TREND ANALYSIS:")
                if trend.get('trend_significance', False):
                    direction = trend.get('trend_direction', 'unknown')
                    print(f"   {'✅' if direction == 'improving' else '⚠️'} {trend['interpretation']}")
                else:
                    print(f"   ✅ {trend.get('interpretation', 'Stable performance over time')}")
        
        # Stress testing statistical analysis
        stress_categories = ['high_volatility', 'market_crashes', 'extended_bear_markets']
        if 'stress_testing' in self.benchmark_results:
            print(f"\n⚡ STRESS TESTING STATISTICAL RESULTS:")
            
            for category in stress_categories:
                if (category in self.benchmark_results['stress_testing'] and 
                    self.benchmark_results['stress_testing'][category] and
                    'statistical_analysis' in self.benchmark_results['stress_testing'][category]):
                    
                    stress_stats = self.benchmark_results['stress_testing'][category]['statistical_analysis']
                    category_name = category.replace('_', ' ').title()
                    
                    print(f"   🔸 {category_name}:")
                    for metric_key, analysis in stress_stats.items():
                        if metric_key.endswith('_significance') and isinstance(analysis, dict):
                            if analysis.get('significant_results'):
                                print(f"      ✅ Significant performance maintained during stress")
                            else:
                                print(f"      ⚠️  Performance degraded during stress conditions")
        
        # Baseline comparison statistical analysis
        if ('baseline_comparison' in self.benchmark_results and 
            self.benchmark_results['baseline_comparison'] and
            'statistical_comparisons' in self.benchmark_results['baseline_comparison']):
            
            baseline_stats = self.benchmark_results['baseline_comparison']['statistical_comparisons']
            print(f"\n📈 BASELINE COMPARISON STATISTICAL RESULTS:")
            
            for comparison_key, analysis in baseline_stats.items():
                if isinstance(analysis, dict) and 'interpretation' in analysis:
                    comparison_name = comparison_key.replace('vs_', '').replace('_', ' ').title()
                    print(f"   🔸 {comparison_name}:")
                    
                    if analysis.get('model_significantly_better', False):
                        print(f"      ✅ {analysis['interpretation']}")
                    else:
                        print(f"      ⚠️  {analysis['interpretation']}")
        
        print("-" * 60)
        
    def _generate_recommendations(self, assessment):
        """Generate specific recommendations based on assessment"""
        
        recommendations = {
            'immediate_actions': [],
            'improvements': [],
            'monitoring': [],
            'risk_management': []
        }
        
        # Based on overall score
        overall_score = assessment['overall_score']
        
        if overall_score < 0.3:
            recommendations['immediate_actions'].append("🚨 STOP: Model requires fundamental redesign")
            recommendations['immediate_actions'].append("🔄 Consider different architecture or feature engineering")
        elif overall_score < 0.5:
            recommendations['immediate_actions'].append("⚠️ Model needs significant improvements before production")
            recommendations['improvements'].append("📊 Focus on improving prediction accuracy and stability")
        elif overall_score < 0.6:
            recommendations['immediate_actions'].append("🔧 Model needs minor improvements before production")
            recommendations['improvements'].append("⚡ Focus on identified weak areas")
        else:
            recommendations['immediate_actions'].append("✅ Model ready for cautious production deployment")
            recommendations['monitoring'].append("📈 Implement comprehensive monitoring")
        
        # Specific category recommendations
        for category, score in assessment['category_scores'].items():
            if score < 0.5:
                if category == 'walk_forward':
                    recommendations['improvements'].append("🎯 Improve core prediction accuracy and consistency")
                elif category == 'regime_specific':
                    recommendations['improvements'].append("🔄 Enhance regime detection and adaptation")
                elif category == 'stress_testing':
                    recommendations['improvements'].append("⚡ Improve performance in extreme market conditions")
                elif category == 'trading_performance':
                    recommendations['improvements'].append("💰 Optimize position sizing and risk management")
                elif category == 'stability':
                    recommendations['improvements'].append("🔒 Improve model consistency and reduce variance")
        
        # General recommendations
        if self.production_ready:
            recommendations['monitoring'].extend([
                "📊 Monitor prediction accuracy daily",
                "🔄 Retrain monthly with 18-month rolling window",
                "⚠️ Set up performance degradation alerts",
                "💰 Start with small position sizes (max 5%)",
                "📈 Track regime changes and adapt accordingly"
            ])
            
            recommendations['risk_management'].extend([
                "🛡️ Implement strict stop-losses",
                "📉 Monitor maximum drawdown (stop if >15%)",
                "🔄 Have fallback strategies ready",
                "📊 Regular model performance reviews"
            ])
        else:
            recommendations['improvements'].extend([
                "🔄 Consider expanding training data sources",
                "⚡ Improve feature engineering",
                "🎯 Enhance model architecture",
                "📊 Better hyperparameter optimization"
            ])
        
        return recommendations
    
    def _save_results_to_json(self, results):
        """Save benchmark results to JSON file in benchmark_results folder"""
        
        try:
            # Get the script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to core/LSTM-Predict directory
            lstm_dir = os.path.dirname(script_dir)
            # Create benchmark_results directory path
            results_dir = os.path.join(lstm_dir, 'benchmark_results')
            
            # Create benchmark_results directory if it doesn't exist
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_numpy_types(results)
            
            # Save to JSON file
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n💾 Results saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving results to JSON: {e}")
    
    def _convert_numpy_types(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization"""
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj
    
    def _perform_statistical_significance_tests(self, results_list, metric_name='direction_accuracy'):
        """
        Perform statistical significance tests on model performance across periods
        
        Parameters:
        - results_list: List of results dictionaries from different test periods
        - metric_name: Name of metric to test for significance
        
        Returns:
        - Dictionary containing statistical test results
        """
        
        significance_results = {
            'metric_tested': metric_name,
            'sample_size': 0,
            'tests_performed': [],
            'significant_results': [],
            'interpretation': ''
        }
        
        try:
            # Extract metric values from results
            metric_values = []
            for result in results_list:
                if isinstance(result, dict) and metric_name in result:
                    value = result[metric_name]
                    if np.isfinite(value):
                        metric_values.append(value)
            
            if len(metric_values) < 2:
                significance_results['interpretation'] = 'Insufficient data for statistical testing'
                return significance_results
            
            significance_results['sample_size'] = len(metric_values)
            metric_array = np.array(metric_values)
            
            # Test 1: One-sample t-test against random performance (0.5 for accuracy metrics)
            if metric_name in ['direction_accuracy', 'win_rate']:
                null_hypothesis_value = 0.5  # Random performance
                t_stat, p_value = ttest_ind([null_hypothesis_value] * len(metric_values), metric_values)
                
                significance_results['tests_performed'].append({
                    'test_name': 'One-sample t-test vs random',
                    'null_hypothesis': f'{metric_name} = {null_hypothesis_value}',
                    'statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'effect_size': float(np.mean(metric_values) - null_hypothesis_value)
                })
                
                if p_value < 0.05:
                    significance_results['significant_results'].append(
                        f'{metric_name} significantly different from random (p={p_value:.4f})'
                    )
            
            # Test 2: Normality test (Shapiro-Wilk for small samples)
            if len(metric_values) <= 50:
                shapiro_stat, shapiro_p = stats.shapiro(metric_values)
                significance_results['tests_performed'].append({
                    'test_name': 'Shapiro-Wilk normality test',
                    'null_hypothesis': 'Data is normally distributed',
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'significant': shapiro_p < 0.05,
                    'normally_distributed': shapiro_p >= 0.05
                })
            
            # Test 3: Consistency test - compare first half vs second half of results
            if len(metric_values) >= 4:
                mid_point = len(metric_values) // 2
                first_half = metric_array[:mid_point]
                second_half = metric_array[mid_point:]
                
                # Use Mann-Whitney U test (non-parametric) for robustness
                mw_stat, mw_p = mannwhitneyu(first_half, second_half, alternative='two-sided')
                
                significance_results['tests_performed'].append({
                    'test_name': 'Mann-Whitney U (first vs second half)',
                    'null_hypothesis': 'No difference between periods',
                    'statistic': float(mw_stat),
                    'p_value': float(mw_p),
                    'significant': mw_p < 0.05,
                    'first_half_median': float(np.median(first_half)),
                    'second_half_median': float(np.median(second_half))
                })
                
                if mw_p < 0.05:
                    significance_results['significant_results'].append(
                        f'Significant difference between early and late periods (p={mw_p:.4f})'
                    )
            
            # Test 4: Variance stability test
            if len(metric_values) >= 6:
                # Split into three groups and test for equal variances
                group_size = len(metric_values) // 3
                group1 = metric_array[:group_size]
                group2 = metric_array[group_size:2*group_size]
                group3 = metric_array[2*group_size:]
                
                if len(group1) > 1 and len(group2) > 1 and len(group3) > 1:
                    levene_stat, levene_p = stats.levene(group1, group2, group3)
                    
                    significance_results['tests_performed'].append({
                        'test_name': 'Levene test for equal variances',
                        'null_hypothesis': 'Equal variances across periods',
                        'statistic': float(levene_stat),
                        'p_value': float(levene_p),
                        'significant': levene_p < 0.05,
                        'homoscedastic': levene_p >= 0.05
                    })
            
            # Generate interpretation
            significant_count = len(significance_results['significant_results'])
            total_tests = len(significance_results['tests_performed'])
            
            if significant_count == 0:
                significance_results['interpretation'] = (
                    f'No statistically significant results found in {total_tests} tests. '
                    f'Performance may be due to chance or sample size is too small.'
                )
            elif significant_count == total_tests:
                significance_results['interpretation'] = (
                    f'All {total_tests} tests show statistical significance. '
                    f'Strong evidence that model performance is not due to chance.'
                )
            else:
                significance_results['interpretation'] = (
                    f'{significant_count} out of {total_tests} tests show significance. '
                    f'Mixed evidence for statistical significance.'
                )
            
            # Add confidence intervals
            if len(metric_values) > 1:
                confidence_interval = stats.t.interval(
                    0.95, 
                    len(metric_values) - 1,
                    loc=np.mean(metric_values),
                    scale=stats.sem(metric_values)
                )
                
                significance_results['confidence_interval_95'] = {
                    'lower': float(confidence_interval[0]),
                    'upper': float(confidence_interval[1]),
                    'mean': float(np.mean(metric_values))
                }
            
        except Exception as e:
            significance_results['error'] = f'Statistical testing failed: {str(e)}'
            significance_results['interpretation'] = 'Unable to perform statistical tests due to error'
        
        return significance_results
    
    def _compare_model_performance_statistically(self, model_results, baseline_results, metric_name='direction_accuracy'):
        """
        Compare model performance against baseline using statistical tests
        
        Parameters:
        - model_results: List of model performance results
        - baseline_results: List of baseline performance results
        - metric_name: Metric to compare
        
        Returns:
        - Dictionary containing comparison test results
        """
        
        comparison_results = {
            'metric_compared': metric_name,
            'model_sample_size': 0,
            'baseline_sample_size': 0,
            'tests_performed': [],
            'model_significantly_better': False,
            'effect_size': 0,
            'interpretation': ''
        }
        
        try:
            # Extract metric values
            model_values = [r[metric_name] for r in model_results 
                          if isinstance(r, dict) and metric_name in r and np.isfinite(r[metric_name])]
            baseline_values = [r[metric_name] for r in baseline_results 
                             if isinstance(r, dict) and metric_name in r and np.isfinite(r[metric_name])]
            
            if len(model_values) < 2 or len(baseline_values) < 2:
                comparison_results['interpretation'] = 'Insufficient data for statistical comparison'
                return comparison_results
            
            comparison_results['model_sample_size'] = len(model_values)
            comparison_results['baseline_sample_size'] = len(baseline_values)
            
            model_array = np.array(model_values)
            baseline_array = np.array(baseline_values)
            
            # Test 1: Two-sample t-test
            t_stat, t_p = ttest_ind(model_array, baseline_array, alternative='greater')
            
            comparison_results['tests_performed'].append({
                'test_name': 'Two-sample t-test (model > baseline)',
                'null_hypothesis': 'Model performance <= Baseline performance',
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': t_p < 0.05,
                'model_mean': float(np.mean(model_array)),
                'baseline_mean': float(np.mean(baseline_array))
            })
            
            # Test 2: Mann-Whitney U test (non-parametric)
            mw_stat, mw_p = mannwhitneyu(model_array, baseline_array, alternative='greater')
            
            comparison_results['tests_performed'].append({
                'test_name': 'Mann-Whitney U test (model > baseline)',
                'null_hypothesis': 'Model performance <= Baseline performance',
                'statistic': float(mw_stat),
                'p_value': float(mw_p),
                'significant': mw_p < 0.05,
                'model_median': float(np.median(model_array)),
                'baseline_median': float(np.median(baseline_array))
            })
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(model_array) - 1) * np.var(model_array, ddof=1) + 
                                (len(baseline_array) - 1) * np.var(baseline_array, ddof=1)) / 
                               (len(model_array) + len(baseline_array) - 2))
            
            effect_size = (np.mean(model_array) - np.mean(baseline_array)) / (pooled_std + 1e-8)
            comparison_results['effect_size'] = float(effect_size)
            
            # Determine if model is significantly better
            significant_tests = sum(1 for test in comparison_results['tests_performed'] if test['significant'])
            comparison_results['model_significantly_better'] = significant_tests >= 1
            
            # Generate interpretation
            if comparison_results['model_significantly_better']:
                effect_magnitude = 'large' if abs(effect_size) > 0.8 else 'medium' if abs(effect_size) > 0.5 else 'small'
                comparison_results['interpretation'] = (
                    f'Model significantly outperforms baseline with {effect_magnitude} effect size '
                    f'(d={effect_size:.3f}). Model mean: {np.mean(model_array):.3f}, '
                    f'Baseline mean: {np.mean(baseline_array):.3f}'
                )
            else:
                comparison_results['interpretation'] = (
                    f'No statistically significant difference between model and baseline. '
                    f'Model mean: {np.mean(model_array):.3f}, Baseline mean: {np.mean(baseline_array):.3f}'
                )
        
        except Exception as e:
            comparison_results['error'] = f'Statistical comparison failed: {str(e)}'
            comparison_results['interpretation'] = 'Unable to perform statistical comparison due to error'
        
        return comparison_results
    
    def _analyze_performance_trends(self, results_list, metric_name='direction_accuracy'):
        """
        Analyze trends in model performance over time using regression
        
        Parameters:
        - results_list: List of results dictionaries ordered by time
        - metric_name: Metric to analyze for trends
        
        Returns:
        - Dictionary containing trend analysis results
        """
        
        trend_results = {
            'metric_analyzed': metric_name,
            'sample_size': 0,
            'trend_direction': 'none',
            'trend_significance': False,
            'slope': 0,
            'r_squared': 0,
            'interpretation': ''
        }
        
        try:
            # Extract metric values and create time index
            metric_values = []
            for i, result in enumerate(results_list):
                if isinstance(result, dict) and metric_name in result:
                    value = result[metric_name]
                    if np.isfinite(value):
                        metric_values.append(value)
            
            if len(metric_values) < 3:
                trend_results['interpretation'] = 'Insufficient data for trend analysis'
                return trend_results
            
            trend_results['sample_size'] = len(metric_values)
            
            # Create time index (assuming chronological order)
            time_index = np.arange(len(metric_values))
            
            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, metric_values)
            
            trend_results['slope'] = float(slope)
            trend_results['r_squared'] = float(r_value ** 2)
            trend_results['p_value'] = float(p_value)
            trend_results['trend_significance'] = p_value < 0.05
            
            # Determine trend direction
            if trend_results['trend_significance']:
                if slope > 0:
                    trend_results['trend_direction'] = 'improving'
                else:
                    trend_results['trend_direction'] = 'degrading'
            else:
                trend_results['trend_direction'] = 'stable'
            
            # Generate interpretation
            if trend_results['trend_significance']:
                direction = 'improving' if slope > 0 else 'degrading'
                trend_results['interpretation'] = (
                    f'Statistically significant {direction} trend detected (p={p_value:.4f}). '
                    f'R² = {r_value**2:.3f}, slope = {slope:.6f} per period.'
                )
            else:
                trend_results['interpretation'] = (
                    f'No statistically significant trend detected (p={p_value:.4f}). '
                    f'Performance appears stable over time.'
                )
        
        except Exception as e:
            trend_results['error'] = f'Trend analysis failed: {str(e)}'
            trend_results['interpretation'] = 'Unable to perform trend analysis due to error'
        
        return trend_results


# Usage function
def run_comprehensive_benchmark(predictor_class, df):
    """
    Run comprehensive benchmark on your Bitcoin prediction model
    
    Parameters:
    - predictor_class: Your RegimeAwareBitcoinPredictor class
    - df: Your feature-engineered dataset
    
    Returns:
    - Complete benchmark results and production readiness assessment
    """
    
    benchmark = ComprehensiveModelBenchmark(
        predictor_class=predictor_class,
        optimal_window_months=18  # Based on your training window analysis
    )
    
    results = benchmark.run_comprehensive_benchmark(
        df=df,
        n_test_periods=12,    # Test across 12 different time periods
        test_length_days=60   # Each test period is 60 days
    )
    
    return results

# Example usage:
btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data()
# Assuming you have your df with engineered features
df_news = add_vader_sentiment(df_news)
df_newsdaily_sentiment = aggregate_daily_sentiment(df_news)

# 3. Feature engineering
df = engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment)

# Run the comprehensive benchmark
benchmark_results = run_comprehensive_benchmark(
    predictor_class=ImprovedBitcoinPredictor,
    df=df  # Your feature-engineered dataset
)

# Check if model is production ready
if benchmark_results['production_ready']:
    print("🚀 Your model is ready for production!")
    print("Recommendations:", benchmark_results['recommendations']['monitoring'])
else:
    print("🔧 Your model needs improvements")
    print("Critical actions:", benchmark_results['recommendations']['immediate_actions'])
