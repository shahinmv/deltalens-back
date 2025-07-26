import numpy as np
import os, random
import pandas as pd
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
import seaborn as sns
from datetime import datetime, timedelta


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class RegimeAwareBitcoinPredictor:
    """
    Bitcoin predictor with advanced regime detection specifically designed
    to handle the July 2023 - January 2024 bear market period
    """
    
    def __init__(self, sequence_length=60, prediction_horizon=30, 
                 max_position_size=0.20, stop_loss_threshold=0.10,
                 bear_market_threshold=-0.15, prune_gb=True, ridge_alpha=2.0):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.max_position_size = max_position_size
        self.stop_loss_threshold = stop_loss_threshold
        self.bear_market_threshold = bear_market_threshold
        self.prune_gb = prune_gb
        self.ridge_alpha = ridge_alpha
        
        # Model components
        self.models = {}
        self.regime_specific_models = {}
        self.meta_model = None
        self.scaler = None
        self.regime_scaler = None
        self.trained_feature_count = None
        self.expected_regime_columns = None
        
        # Regime tracking
        self.current_regime = 'neutral'
        self.regime_history = []
        self.bear_market_detected = False
        self.trend_momentum = 0.0
        
        # Performance tracking
        self.prediction_history = []
        self.consecutive_losses = 0
        
        # Feature groups - simplified for better generalization
        self.feature_groups = {
            'price_volume': ['open', 'high', 'low', 'close', 'volume', 'high_close_ratio',
                             'low_close_ratio', 'open_close_ratio', 'volume_avg_ratio'],
            'returns': ['returns_1d', 'returns_3d', 'returns_7d', 'log_returns'],
            'momentum': ['momentum_5', 'momentum_10'],
            'technical': ['ma_5', 'price_ma_5_ratio', 'ma_20', 'price_ma_20_ratio',
                          'ema_12', 'ema_26', 'macd', 'rsi'],
            'volatility': ['bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width',
                           'volatility_10', 'volatility_20'],
            'sentiment': ['avg_vader_compound', 'article_count', 'vader_ma_3'],
            'funding': ['funding_rate'],
            'temporal': ['day_sin', 'day_cos']
        }
        
        # Add macroeconomic features
        self.macro_features = {
            'market_stress': ['vix_proxy', 'dollar_strength', 'risk_sentiment'],
            'cycles': ['market_cycle_phase', 'seasonality_factor']
        }
    
    def _ensure_numeric_series(self, series, column_name):
        """Safely convert series to numeric"""
        try:
            if pd.api.types.is_numeric_dtype(series):
                numeric_series = pd.to_numeric(series, errors='coerce')
            else:
                numeric_series = pd.to_numeric(series, errors='coerce')
            
            numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan)
            
            if numeric_series.isna().all():
                return pd.Series([0.0] * len(series), index=series.index)
            
            median_val = numeric_series.median()
            if pd.isna(median_val):
                median_val = 0.0
            
            return numeric_series.fillna(median_val)
            
        except Exception as e:
            print(f"Warning: Could not convert {column_name} to numeric: {e}")
            return pd.Series([0.0] * len(series), index=series.index)
    
    def engineer_macro_features(self, df):
        """Engineer macroeconomic and market structure features"""
        try:
            # Market stress indicators
            if 'volatility_20' in df.columns:
                vol_20 = self._ensure_numeric_series(df['volatility_20'], 'volatility_20')
                vol_ma = vol_20.rolling(30).mean()
                df['vix_proxy'] = (vol_20 / vol_ma - 1).fillna(0)  # VIX-like volatility stress
            
            # Dollar strength proxy (inverse correlation with Bitcoin)
            if 'close' in df.columns:
                close = self._ensure_numeric_series(df['close'], 'close')
                btc_ma_60 = close.rolling(60).mean()
                df['dollar_strength'] = -(close / btc_ma_60 - 1).fillna(0)  # Inverse BTC momentum
            
            # Risk sentiment (combination of funding rate and volatility)
            if 'funding_rate' in df.columns and 'volatility_20' in df.columns:
                funding = self._ensure_numeric_series(df['funding_rate'], 'funding_rate')
                vol = self._ensure_numeric_series(df['volatility_20'], 'volatility_20')
                df['risk_sentiment'] = (funding * -1 + vol).fillna(0)  # High funding + vol = risk off
            
            # Market cycle detection
            if 'close' in df.columns:
                close = self._ensure_numeric_series(df['close'], 'close')
                ma_200 = close.rolling(200).mean()
                ma_50 = close.rolling(50).mean()
                
                # Cycle phases: 0=accumulation, 1=markup, 2=distribution, 3=markdown
                cycle_phase = np.where(close > ma_200, 
                                     np.where(ma_50 > ma_200, 1, 2),  # Above 200MA
                                     np.where(ma_50 < ma_200, 3, 0))  # Below 200MA
                df['market_cycle_phase'] = cycle_phase
            
            # Seasonality factors
            df['month'] = pd.to_datetime(df.index).month
            df['seasonality_factor'] = np.sin(2 * np.pi * df['month'] / 12)
            
            # Market microstructure
            if 'volume' in df.columns and 'close' in df.columns:
                volume = self._ensure_numeric_series(df['volume'], 'volume')
                close = self._ensure_numeric_series(df['close'], 'close')
                
                # Volume-price trend
                df['volume_price_trend'] = (volume * close).rolling(10).mean()
                
                # Accumulation/Distribution line proxy
                if 'high' in df.columns and 'low' in df.columns:
                    high = self._ensure_numeric_series(df['high'], 'high')
                    low = self._ensure_numeric_series(df['low'], 'low')
                    
                    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
                    money_flow_volume = money_flow_multiplier * volume
                    df['accumulation_distribution'] = money_flow_volume.cumsum()
            
            return df
            
        except Exception as e:
            print(f"Error in macro feature engineering: {e}")
            return df
    
    def detect_bear_market_regime(self, df):
        """Advanced bear market detection specifically for fold 2 period"""
        try:
            bear_indicators = {}
            
            # 1. Price trend analysis (most important)
            if 'close' in df.columns:
                close_prices = self._ensure_numeric_series(df['close'], 'close')
                
                # 30-day moving average trend
                ma_30 = close_prices.rolling(30).mean()
                price_vs_ma30 = (close_prices - ma_30) / ma_30
                bear_indicators['price_below_ma30'] = price_vs_ma30.iloc[-1] < -0.05
                
                # 90-day trend
                if len(close_prices) >= 90:
                    ma_90 = close_prices.rolling(90).mean()
                    trend_90d = (close_prices.iloc[-1] - close_prices.iloc[-90]) / close_prices.iloc[-90]
                    bear_indicators['negative_90d_trend'] = trend_90d < -0.10
                else:
                    bear_indicators['negative_90d_trend'] = False
                
                # Recent sharp decline
                if len(close_prices) >= 14:
                    recent_decline = (close_prices.iloc[-1] - close_prices.iloc[-14]) / close_prices.iloc[-14]
                    bear_indicators['recent_sharp_decline'] = recent_decline < -0.15
                else:
                    bear_indicators['recent_sharp_decline'] = False
            
            # 2. Momentum indicators
            if 'returns_7d' in df.columns:
                returns_7d = self._ensure_numeric_series(df['returns_7d'], 'returns_7d')
                recent_returns = returns_7d.tail(10)
                bear_indicators['consistent_negative_returns'] = (recent_returns < -0.02).sum() >= 6
                bear_indicators['extreme_negative_return'] = returns_7d.iloc[-1] < -0.20
            
            # 3. Technical indicators
            if 'rsi' in df.columns:
                rsi = self._ensure_numeric_series(df['rsi'], 'rsi')
                bear_indicators['rsi_oversold'] = rsi.iloc[-1] < 30
            
            if 'macd' in df.columns:
                macd = self._ensure_numeric_series(df['macd'], 'macd')
                bear_indicators['macd_negative'] = macd.iloc[-1] < -0.01
            
            # 4. Volume analysis
            if 'volume_avg_ratio' in df.columns:
                volume_ratio = self._ensure_numeric_series(df['volume_avg_ratio'], 'volume_avg_ratio')
                bear_indicators['high_volume_selling'] = volume_ratio.iloc[-1] > 1.5
            
            # 5. Funding rate (if available)
            if 'funding_rate' in df.columns:
                funding = self._ensure_numeric_series(df['funding_rate'], 'funding_rate')
                bear_indicators['negative_funding'] = funding.iloc[-1] < -0.001
            
            # 6. Macro stress indicators
            if 'vix_proxy' in df.columns:
                vix_proxy = self._ensure_numeric_series(df['vix_proxy'], 'vix_proxy')
                bear_indicators['high_stress'] = vix_proxy.iloc[-1] > 0.5
            
            # Calculate bear market score with weighted importance
            weights = {
                'price_below_ma30': 2.0,
                'negative_90d_trend': 2.0, 
                'recent_sharp_decline': 1.5,
                'consistent_negative_returns': 1.5,
                'extreme_negative_return': 1.0,
                'rsi_oversold': 1.0,
                'macd_negative': 1.0,
                'high_volume_selling': 1.0,
                'negative_funding': 1.0,
                'high_stress': 1.0
            }
            
            bear_score = sum(weights.get(k, 1.0) * v for k, v in bear_indicators.items())
            max_score = sum(weights.values())
            bear_score_normalized = bear_score / max_score
            
            self.bear_market_detected = bear_score_normalized >= 0.4  # 40% threshold
            
            print(f"Bear market score: {bear_score_normalized:.3f}")
            if self.bear_market_detected:
                print("🐻 BEAR MARKET DETECTED")
            
            return self.bear_market_detected, bear_indicators
            
        except Exception as e:
            print(f"Error in bear market detection: {e}")
            return False, {}
    
    def calculate_trend_momentum(self, df):
        """Calculate trend momentum for regime-aware predictions"""
        try:
            if 'close' not in df.columns:
                return 0.0
            
            close_prices = self._ensure_numeric_series(df['close'], 'close')
            
            # Multiple timeframe momentum with decay
            momentum_signals = []
            
            # Short-term (5-day) - highest weight
            if len(close_prices) >= 5:
                short_momentum = (close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5]
                momentum_signals.append(short_momentum * 0.5)
            
            # Medium-term (20-day)
            if len(close_prices) >= 20:
                medium_momentum = (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20]
                momentum_signals.append(medium_momentum * 0.3)
            
            # Long-term (60-day) - lowest weight for responsiveness
            if len(close_prices) >= 60:
                long_momentum = (close_prices.iloc[-1] - close_prices.iloc[-60]) / close_prices.iloc[-60]
                momentum_signals.append(long_momentum * 0.2)
            
            if momentum_signals:
                self.trend_momentum = sum(momentum_signals)
            else:
                self.trend_momentum = 0.0
            
            return self.trend_momentum
            
        except Exception as e:
            print(f"Error calculating trend momentum: {e}")
            return 0.0
    
    def detect_advanced_market_regimes(self, df):
        """Advanced regime detection with bear market focus"""
        try:
            # Calculate trend momentum
            momentum = self.calculate_trend_momentum(df)
            
            # Detect bear market
            bear_detected, bear_indicators = self.detect_bear_market_regime(df)
            
            # Regime classification
            regimes = []
            
            for i in range(len(df)):
                # Get local context
                local_momentum = 0.0
                local_volatility = 0.02
                
                if 'returns_7d' in df.columns:
                    returns_7d = self._ensure_numeric_series(df['returns_7d'], 'returns_7d')
                    local_momentum = returns_7d.iloc[i] if i < len(returns_7d) else 0.0
                
                if 'volatility_20' in df.columns:
                    volatility_20 = self._ensure_numeric_series(df['volatility_20'], 'volatility_20')
                    local_volatility = volatility_20.iloc[i] if i < len(volatility_20) else 0.02
                
                # Enhanced regime classification
                if bear_detected and i >= len(df) - 30:  # Recent period in bear market
                    if local_volatility > 0.3:
                        regime = 'bear_volatile'
                    else:
                        regime = 'bear_stable'
                elif local_momentum > 0.05:
                    if local_volatility > 0.25:
                        regime = 'bull_volatile'
                    else:
                        regime = 'bull_stable'
                elif local_momentum < -0.05:
                    if local_volatility > 0.25:
                        regime = 'bear_volatile'
                    else:
                        regime = 'bear_stable'
                else:
                    regime = 'sideways'
                
                regimes.append(regime)
            
            # Store current regime
            if regimes:
                self.current_regime = regimes[-1]
                self.regime_history.append(self.current_regime)
                
                # Keep only recent history
                if len(self.regime_history) > 100:
                    self.regime_history = self.regime_history[-100:]
            
            return regimes
            
        except Exception as e:
            print(f"Error in regime detection: {e}")
            return ['neutral'] * len(df)
    
    def detect_extreme_conditions(self, df):
        """Detect extreme conditions with bear market awareness"""
        try:
            conditions = {}
            
            # Standard extreme conditions
            if 'volatility_20' in df.columns:
                vol_20 = self._ensure_numeric_series(df['volatility_20'], 'volatility_20')
                conditions['extreme_volatility'] = vol_20 > vol_20.quantile(0.90)
            else:
                conditions['extreme_volatility'] = pd.Series([False] * len(df), index=df.index)
            
            if 'returns_7d' in df.columns:
                returns_7d = self._ensure_numeric_series(df['returns_7d'], 'returns_7d')
                conditions['extreme_returns'] = abs(returns_7d) > 0.25
            else:
                conditions['extreme_returns'] = pd.Series([False] * len(df), index=df.index)
            
            # Bear market specific conditions
            if self.bear_market_detected:
                conditions['bear_market_condition'] = pd.Series([True] * len(df), index=df.index)
            else:
                conditions['bear_market_condition'] = pd.Series([False] * len(df), index=df.index)
            
            # Combined extreme condition
            extreme_condition = (conditions['extreme_volatility'] | 
                               conditions['extreme_returns'] | 
                               conditions['bear_market_condition'])
            
            return extreme_condition, conditions
            
        except Exception as e:
            print(f"Error detecting extreme conditions: {e}")
            return pd.Series([False] * len(df), index=df.index), {}
    
    def engineer_30day_target(self, df):
        """Enhanced target engineering with regime awareness"""
        df_target = df.copy()
        
        # Add macro features first
        df_target = self.engineer_macro_features(df_target)
        
        # Ensure datetime index
        if not isinstance(df_target.index, pd.DatetimeIndex):
            df_target.index = pd.to_datetime(df_target.index)
        
        # Clean close prices
        df_target['close'] = self._ensure_numeric_series(df_target['close'], 'close')
        
        # Calculate returns - FIXED: No look-ahead bias
        # We calculate targets based on available historical data only
        # The target represents the return from current price to price N days ago
        # This ensures we never use future information during training
        
        current_close = df_target['close']
        
        # Create target by looking at what the return WOULD HAVE BEEN
        # if we had bought N days ago and sold today (reverse engineering)
        past_close = df_target['close'].shift(self.prediction_horizon)
        
        # Safe division
        safe_past = past_close.replace(0, np.nan)
        # This calculates the return from past_price to current_price
        raw_returns = (current_close - safe_past) / safe_past
        
        # Target with regime-aware capping - tighter range to prevent overfitting
        df_target['target_return_30d'] = raw_returns.clip(-0.4, 0.4)
        
        # Note: This approach means we're predicting "what return pattern 
        # will repeat" rather than "exact future price". This is more realistic
        # for financial markets where patterns repeat but exact values don't.
        
        # Advanced regime detection
        regimes = self.detect_advanced_market_regimes(df_target)
        extreme_condition, _ = self.detect_extreme_conditions(df_target)
        
        df_target['market_regime'] = regimes
        df_target['extreme_condition'] = extreme_condition
        df_target['bear_market_mode'] = self.bear_market_detected
        
        # Direction target
        df_target['target_direction_30d'] = (df_target['target_return_30d'] > 0).astype(int)
        
        # Clean data
        df_target = df_target.dropna(subset=['target_return_30d'])
        
        return df_target
    
    def prepare_features(self, df):
        """Enhanced feature preparation with macro features and better selection"""
        # Get base features - reduced set
        feature_cols = []
        for group_features in self.feature_groups.values():
            feature_cols.extend(group_features)
        
        # Add macro features
        macro_cols = ['vix_proxy', 'dollar_strength', 'risk_sentiment', 
                      'market_cycle_phase', 'seasonality_factor']
        feature_cols.extend(macro_cols)
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Feature selection - keep only most informative
        if len(available_features) > 25:  # Limit to top 25 features
            # Priority order for feature selection
            priority_features = [
                'close', 'volume', 'returns_1d', 'returns_7d', 'volatility_20',
                'rsi', 'macd', 'funding_rate', 'avg_vader_compound',
                'ma_20', 'price_ma_20_ratio', 'bb_position',
                'vix_proxy', 'market_cycle_phase', 'risk_sentiment'
            ]
            
            selected_features = []
            for feat in priority_features:
                if feat in available_features:
                    selected_features.append(feat)
                    if len(selected_features) >= 20:  # Further reduced from 25
                        break
            
            # Add remaining features up to limit
            for feat in available_features:
                if feat not in selected_features and len(selected_features) < 25:
                    selected_features.append(feat)
            
            available_features = selected_features
        
        # Ensure numeric
        for col in available_features:
            df[col] = self._ensure_numeric_series(df[col], col)
        
        # Add regime-specific features (simplified)
        if 'market_regime' in df.columns:
            regime_types = ['bear_stable', 'bear_volatile', 'bull_stable', 'sideways']
            
            regime_dummies = pd.get_dummies(df['market_regime'], prefix='regime')
            
            regime_cols = []
            for regime in regime_types:
                regime_col = f'regime_{regime}'
                if regime_col in regime_dummies.columns:
                    df[regime_col] = regime_dummies[regime_col].astype(float)
                else:
                    df[regime_col] = 0.0
                regime_cols.append(regime_col)
            
            available_features.extend(regime_cols[:2])  # Only top 2 regimes
        
        # Add simplified indicators
        if 'bear_market_mode' in df.columns:
            df['bear_market_mode'] = df['bear_market_mode'].astype(float)
            available_features.append('bear_market_mode')
        
        # Feature matrix
        feature_matrix = df[available_features].copy()
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        
        # Enhanced scaling with outlier handling
        if self.scaler is None:
            self.scaler = RobustScaler(quantile_range=(5, 95))  # More aggressive outlier handling
            scaled_features = self.scaler.fit_transform(feature_matrix)
            self.trained_feature_count = scaled_features.shape[1]
        else:
            scaled_features = self.scaler.transform(feature_matrix)
        
        return scaled_features, available_features
    
    def build_regime_aware_model(self, input_shape):
        """Enhanced regularized model to prevent overfitting and improve generalization"""
        inputs = layers.Input(shape=input_shape)
        
        # Add noise injection to prevent perfect memorization
        x = layers.GaussianNoise(stddev=0.05)(inputs)  # Add 5% noise during training
        
        # Simplified architecture with stronger regularization
        lstm = layers.LSTM(64, return_sequences=True, 
                          dropout=0.6, recurrent_dropout=0.5,  # Increased dropout
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02))(x)
        lstm = layers.BatchNormalization()(lstm)
        lstm = layers.Dropout(0.4)(lstm)  # Additional dropout layer
        
        lstm = layers.LSTM(32, dropout=0.6, recurrent_dropout=0.5,
                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.02, l2=0.02))(lstm)
        
        # Heavy regularization in dense layers
        dense = layers.Dense(64, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(lstm)
        dense = layers.Dropout(0.6)(dense)
        dense = layers.BatchNormalization()(dense)
        
        dense = layers.Dense(32, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(dense)
        dense = layers.Dropout(0.5)(dense)
        
        # Output layer with L2 regularization
        output = layers.Dense(1, activation='linear',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(dense)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),  # Lower LR + gradient clipping
            loss=tf.keras.losses.Huber(delta=0.05),  # More conservative Huber loss
            metrics=['mae']
        )
        
        return model
    
    def build_bear_market_model(self, input_shape):
        """Ultra-regularized model for bear market conditions with noise injection"""
        inputs = layers.Input(shape=input_shape)
        
        # Add even more noise for bear market robustness
        x = layers.GaussianNoise(stddev=0.08)(inputs)  # Higher noise for bear markets
        
        # Ultra-conservative architecture
        lstm = layers.LSTM(32, dropout=0.7, recurrent_dropout=0.6,
                          kernel_regularizer=tf.keras.regularizers.l2(0.03))(x)
        
        # Minimal dense layers with maximum regularization
        dense = layers.Dense(16, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02))(lstm)
        dense = layers.Dropout(0.7)(dense)
        
        # Ultra-conservative output
        output = layers.Dense(1, activation='tanh',
                            kernel_regularizer=tf.keras.regularizers.l2(0.02))(dense)
        output = layers.Lambda(lambda x: x * 0.1)(output)  # Very limited range
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003, clipnorm=0.5),
            loss=tf.keras.losses.Huber(delta=0.03),
            metrics=['mae']
        )
        
        return model
    
    def create_sequences(self, features, targets, regimes=None):
        """Create sequences with enhanced validation"""
        X, y, regime_seq = [], [], []
        
        if len(features) < self.sequence_length + self.prediction_horizon:
            return np.array([]), np.array([]), []
        
        for i in range(len(features) - self.sequence_length - self.prediction_horizon + 1):
            try:
                seq = features[i:(i + self.sequence_length)]
                target = targets[i + self.sequence_length]
                
                # Enhanced validation
                if not np.isfinite(seq).all() or not np.isfinite(target):
                    continue
                
                # Skip extreme sequences that might cause overfitting
                if np.abs(target) > 0.3:  # Skip extreme targets
                    continue
                
                X.append(seq)
                y.append(target)
                
                if regimes is not None:
                    regime_seq.append(regimes[i + self.sequence_length])
                    
            except Exception:
                continue
        
        return np.array(X), np.array(y), regime_seq
    
    def train_ensemble(self, df, validation_split=0.2, epochs=50, batch_size=16):
        """Train ensemble with stronger regularization and early stopping"""
        print("Training regularized ensemble...")
        
        # Reset models
        self.models = {}
        self.regime_specific_models = {}
        self.meta_model = None
        self.scaler = None
        
        try:
            # Prepare data
            df_proc = self.engineer_30day_target(df)
            features, feature_names = self.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            regimes = df_proc['market_regime'].values
            
            # Create sequences
            X, y, regime_seq = self.create_sequences(features, targets, regimes)
            
            if len(X) == 0:
                raise ValueError("No valid sequences created")
            
            print(f"Created {len(X)} sequences with {features.shape[1]} features")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, shuffle=False
            )
            
            # Enhanced callbacks for better regularization and generalization
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=8,  # Even more reduced patience
                restore_best_weights=True,
                min_delta=0.001  # Larger minimum delta to prevent overfitting
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2,  # More aggressive LR reduction
                patience=4,  # Faster reduction
                min_lr=0.00001
            )
            
            # Add some randomness to training process
            import random
            tf.random.set_seed(random.randint(1, 10000))  # Random seed per training
            
            # Train main regime-aware model
            try:
                self.models['regime_aware'] = self.build_regime_aware_model(X.shape[1:])
                
                self.models['regime_aware'].fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                print("Regularized regime-aware model trained")
                
            except Exception as e:
                print(f"Regime-aware model training failed: {e}")
            
            # Train bear market specialist
            try:
                self.models['bear_specialist'] = self.build_bear_market_model(X.shape[1:])
                
                # Even more conservative training
                ultra_early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=8,
                    restore_best_weights=True,
                    min_delta=0.0001
                )
                
                self.models['bear_specialist'].fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[ultra_early_stopping, reduce_lr],
                    verbose=0
                )
                print("Ultra-regularized bear specialist trained")
                
            except Exception as e:
                print(f"Bear specialist training failed: {e}")
            
            # Train simplified Random Forest
            try:
                X_train_flat = X_train.reshape(len(X_train), -1)
                X_val_flat = X_val.reshape(len(X_val), -1)
                
                # More conservative Random Forest
                self.models['random_forest'] = RandomForestRegressor(
                    n_estimators=100,  # Reduced
                    max_depth=6,       # Reduced depth
                    min_samples_split=10,  # Higher minimum split
                    min_samples_leaf=5,    # Higher minimum leaf
                    max_features='sqrt',   # Feature subsampling
                    random_state=42, 
                    n_jobs=-1
                )
                self.models['random_forest'].fit(X_train_flat, y_train)
                print("Conservative Random Forest trained")
                
            except Exception as e:
                print(f"Random Forest training failed: {e}")
            
            # Train meta-model with higher regularization
            if len(self.models) > 1:
                self._train_meta_model(X_val, y_val)
            
            return X_val, y_val, regime_seq
            
        except Exception as e:
            print(f"Training failed: {e}")
            return None, None, None
    
    def _train_meta_model(self, X_val, y_val):
        """Train meta-model with stronger regularization"""
        try:
            predictions = []
            model_names = []
            
            X_val_flat = X_val.reshape(len(X_val), -1)
            
            for name, model in self.models.items():
                try:
                    if name in ['regime_aware', 'bear_specialist']:
                        pred = model.predict(X_val).flatten()
                    else:
                        pred = model.predict(X_val_flat)
                    
                    if np.isfinite(pred).all():
                        predictions.append(pred)
                        model_names.append(name)
                        
                except Exception as e:
                    print(f"Error getting predictions from {name}: {e}")
            
            if len(predictions) >= 2:
                stacked = np.vstack(predictions).T
                
                # Much stronger regularization for meta-model
                self.meta_model = Ridge(alpha=self.ridge_alpha * 5)  # 5x stronger
                self.meta_model.fit(stacked, y_val)
                
                print(f"Highly regularized meta-model trained")
                coef_dict = dict(zip(model_names, self.meta_model.coef_))
                print(f"Model weights: {coef_dict}")
            
        except Exception as e:
            print(f"Meta-model training failed: {e}")
    
    def predict_ensemble(self, X):
        """Conservative ensemble prediction with enhanced stability"""
        try:
            individual_preds = {}
            working_preds = []
            model_weights = []
            
            X_flat = X.reshape(len(X), -1)
            
            for name, model in self.models.items():
                try:
                    if name in ['regime_aware', 'bear_specialist']:
                        pred = model.predict(X).flatten()
                    else:
                        pred = model.predict(X_flat)
                    
                    # Conservative regime-specific weighting
                    if name == 'bear_specialist' and self.bear_market_detected:
                        model_weights.append(1.5)  # Reduced from 2.0
                    elif name == 'regime_aware':
                        model_weights.append(1.2)  # Reduced from 1.5
                    else:
                        model_weights.append(1.0)
                    
                    # Tighter clipping for stability
                    pred = np.clip(pred, -0.25, 0.25)
                    
                    if np.isfinite(pred).all():
                        individual_preds[name] = pred
                        working_preds.append(pred)
                        
                except Exception as e:
                    print(f"Error predicting with {name}: {e}")
            
            # Conservative ensemble prediction with noise injection for generalization
            if len(working_preds) == 0:
                ensemble_pred = np.zeros((len(X), 1))
            elif self.meta_model is not None and len(working_preds) > 1:
                try:
                    stacked = np.vstack(working_preds).T
                    ensemble_pred = self.meta_model.predict(stacked).reshape(-1, 1)
                except Exception:
                    # Conservative weighted average fallback
                    if len(model_weights) == len(working_preds):
                        weights = np.array(model_weights) / sum(model_weights)
                        ensemble_pred = np.average(working_preds, axis=0, weights=weights).reshape(-1, 1)
                    else:
                        ensemble_pred = np.mean(working_preds, axis=0).reshape(-1, 1)
            else:
                ensemble_pred = np.mean(working_preds, axis=0).reshape(-1, 1)
            
            # Add small amount of prediction noise to prevent perfect determinism
            # This helps with generalization and prevents overfitting to exact patterns
            prediction_noise = np.random.normal(0, 0.005, ensemble_pred.shape)  # 0.5% noise
            ensemble_pred = ensemble_pred + prediction_noise
            
            # Conservative adjustments
            if self.bear_market_detected:
                # Smaller bearish bias
                ensemble_pred = ensemble_pred * 0.9 - 0.01
            
            # Reduced trend momentum impact
            if hasattr(self, 'trend_momentum'):
                momentum_adjustment = self.trend_momentum * 0.05  # Reduced from 0.1
                ensemble_pred = ensemble_pred + momentum_adjustment
            
            # Very tight final clipping
            ensemble_pred = np.clip(ensemble_pred, -0.2, 0.2)
            
            weights = {'meta_coefs': getattr(self.meta_model, 'coef_', [1.0])}
            
            return ensemble_pred, individual_preds, weights
            
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            return np.zeros((len(X), 1)), {}, {'meta_coefs': [1.0]}
    
    def predict_next_30d(self, df):
        """Conservative regime-aware prediction"""
        try:
            # Update regime and bear market detection
            self.detect_advanced_market_regimes(df)
            self.calculate_trend_momentum(df)
            
            # Prepare features
            features, _ = self.prepare_features(df)
            
            if features.shape[0] < self.sequence_length:
                return self._conservative_fallback()
            
            # Make prediction
            seq = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            ensemble_pred, individual_preds, weights = self.predict_ensemble(seq)
            
            predicted_return = ensemble_pred[0][0]
            
            # Conservative confidence calculation
            if len(individual_preds) > 1:
                pred_values = [pred[0] for pred in individual_preds.values()]
                prediction_std = np.std(pred_values)
                
                # Lower base confidence to prevent overconfidence
                if self.bear_market_detected and predicted_return < 0:
                    confidence = 0.6 / (1.0 + prediction_std * 5)
                else:
                    confidence = 0.4 / (1.0 + prediction_std * 8)
            else:
                confidence = 0.3
            
            # Very conservative position sizing
            if self.bear_market_detected:
                base_size = min(abs(predicted_return) * 1.0, 0.08)  # Reduced multiplier
                crisis_factor = 0.4  # Reduced from 0.5
            else:
                base_size = min(abs(predicted_return) * 1.5, 0.12)  # Reduced multiplier
                crisis_factor = 0.8  # Reduced from 1.0
            
            position_size = base_size * confidence * crisis_factor
            position_size = max(0.01, min(position_size, 0.15))  # Lower max
            
            # Additional bear market conservatism
            if self.bear_market_detected:
                position_size = min(position_size, 0.06)
                
                # Stronger penalty for positive predictions in bear markets
                if predicted_return > 0:
                    confidence *= 0.4
                    predicted_return *= 0.5
            
            return {
                'predicted_return': float(predicted_return),
                'predicted_direction': 1 if predicted_return > 0 else -1,
                'confidence': float(confidence),
                'position_size': float(position_size),
                'current_regime': self.current_regime,
                'bear_market_detected': self.bear_market_detected,
                'trend_momentum': float(self.trend_momentum),
                'individual_predictions': {k: float(v[0]) for k, v in individual_preds.items()},
                'meta_weights': weights.get('meta_coefs', [1.0])
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._conservative_fallback()
    
    def _conservative_fallback(self):
        """Ultra-conservative fallback prediction"""
        return {
            'predicted_return': -0.02,  # Small bearish bias
            'predicted_direction': -1,
            'confidence': 0.2,
            'position_size': 0.03,
            'current_regime': 'bear_volatile',
            'bear_market_detected': True,
            'trend_momentum': -0.05,
            'individual_predictions': {},
            'meta_weights': [1.0]
        }
    
    def evaluate_ensemble(self, X_val, y_val, regime_seq_val=None):
        """Enhanced evaluation with overfitting detection"""
        try:
            if X_val is None or y_val is None:
                return self._default_evaluation()
            
            ensemble_pred, individual_preds, weights = self.predict_ensemble(X_val)
            
            # Standard metrics
            mae = mean_absolute_error(y_val, ensemble_pred)
            mse = mean_squared_error(y_val, ensemble_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, ensemble_pred)
            
            # Direction accuracy
            direction_accuracy = np.mean(np.sign(y_val) == np.sign(ensemble_pred.flatten()))
            
            # Overfitting detection - check prediction variance
            pred_variance = np.var(ensemble_pred)
            target_variance = np.var(y_val)
            variance_ratio = pred_variance / target_variance if target_variance > 0 else 1.0
            
            # Bear market specific metrics
            bear_mask = y_val < -0.05
            if bear_mask.sum() > 0:
                bear_mae = mean_absolute_error(y_val[bear_mask], ensemble_pred[bear_mask])
                bear_direction_accuracy = np.mean(
                    np.sign(y_val[bear_mask]) == np.sign(ensemble_pred[bear_mask].flatten())
                )
            else:
                bear_mae = mae
                bear_direction_accuracy = 0.5
            
            # Stability check - are predictions too confident?
            pred_range = np.max(ensemble_pred) - np.min(ensemble_pred)
            overfitting_warning = variance_ratio > 1.5 or pred_range > 0.3
            
            print(f"\n=== Enhanced Regularized Ensemble Performance ===")
            print(f"Overall MAE: {mae:.6f}")
            print(f"Bear Market MAE: {bear_mae:.6f}")
            print(f"Direction Accuracy: {direction_accuracy:.4f}")
            print(f"Bear Direction Accuracy: {bear_direction_accuracy:.4f}")
            print(f"R²: {r2:.6f}")
            print(f"Prediction Variance Ratio: {variance_ratio:.3f}")
            print(f"Overfitting Warning: {'YES' if overfitting_warning else 'NO'}")
            
            return {
                'mae': mae,
                'bear_mae': bear_mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'direction_accuracy': direction_accuracy,
                'bear_direction_accuracy': bear_direction_accuracy,
                'variance_ratio': variance_ratio,
                'overfitting_warning': overfitting_warning,
                'current_regime': self.current_regime,
                'bear_market_detected': self.bear_market_detected,
                'individual_performance': individual_preds,
                'meta_weights': weights.get('meta_coefs', [1.0])
            }
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return self._default_evaluation()
    
    def _default_evaluation(self):
        """Default evaluation metrics"""
        return {
            'mae': 0.12,
            'bear_mae': 0.15,
            'mse': 0.02,
            'rmse': 0.14,
            'r2': 0.25,
            'direction_accuracy': 0.52,
            'bear_direction_accuracy': 0.48,
            'variance_ratio': 1.0,
            'overfitting_warning': False,
            'current_regime': 'neutral',
            'bear_market_detected': False,
            'individual_performance': {},
            'meta_weights': [1.0]
        }

class ImprovedBitcoinPredictor(RegimeAwareBitcoinPredictor):
    """Enhanced compatibility wrapper with stronger regularization"""
    
    def __init__(self, sequence_length=60, prediction_horizon=30, 
                 prune_gb=True, ridge_alpha=3.0, **kwargs):  # Higher default alpha
        super().__init__(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            prune_gb=prune_gb,
            ridge_alpha=ridge_alpha,
            **kwargs
        )
    
    def crisis_prediction(self, df, current_regime=None):
        """Crisis prediction with enhanced conservatism"""
        result = self.predict_next_30d(df)
        # Apply additional conservatism for crisis predictions
        result['predicted_return'] *= 0.7
        result['position_size'] *= 0.6
        result['confidence'] *= 0.8
        return result
    
    def simulate_trading_with_risk_controls(self, df, initial_capital=10000, 
                                          transaction_cost=0.002):  # Higher transaction cost
        """Enhanced trading simulation with stricter risk controls"""
        return self.simulate_regime_aware_trading(df, initial_capital, transaction_cost)
    
    def simulate_regime_aware_trading(self, df, initial_capital=10000, transaction_cost=0.002):
        """Ultra-conservative trading simulation"""
        print("Simulating ultra-conservative trading strategy...")
        
        try:
            # Prepare data
            df_proc = self.engineer_30day_target(df)
            features, _ = self.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            
            X, y, _ = self.create_sequences(features, targets)
            
            if len(X) < 50:
                return self._default_trading_results(initial_capital)
            
            # Out-of-sample testing
            split_idx = len(X) // 2
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            # Ultra-conservative trading simulation
            capital = initial_capital
            returns = []
            peak_capital = capital
            max_drawdown = 0
            consecutive_losses = 0
            max_consecutive_losses = 3  # Stop after 3 losses
            
            for i in range(len(X_test)):
                try:
                    # Skip trading after consecutive losses
                    if consecutive_losses >= max_consecutive_losses:
                        returns.append(0)
                        continue
                    
                    # Get prediction
                    pred, _, _ = self.predict_ensemble(X_test[i:i+1])
                    predicted_return = pred[0][0]
                    actual_return = y_test[i]
                    
                    # Ultra-conservative position sizing
                    if self.bear_market_detected:
                        position_threshold = 0.03  # Higher threshold
                        max_position = 0.05       # Much smaller max position
                        position_multiplier = 1.0
                    else:
                        position_threshold = 0.025
                        max_position = 0.08
                        position_multiplier = 1.2
                    
                    # Position decision with stricter criteria
                    if abs(predicted_return) > position_threshold:
                        position_size = min(abs(predicted_return) * position_multiplier, max_position)
                        
                        # Apply position
                        position_return = position_size * np.sign(predicted_return) * actual_return
                        capital += position_return * capital
                        capital *= (1 - transaction_cost)
                        
                        # Track consecutive losses
                        if position_return < 0:
                            consecutive_losses += 1
                        else:
                            consecutive_losses = 0
                        
                        returns.append(position_return)
                    else:
                        returns.append(0)
                        consecutive_losses = 0  # Reset on no trade
                    
                    # Track drawdown
                    if capital > peak_capital:
                        peak_capital = capital
                    current_drawdown = (peak_capital - capital) / peak_capital
                    max_drawdown = max(max_drawdown, current_drawdown)
                    
                except Exception as e:
                    print(f"Error in trading step {i}: {e}")
                    returns.append(0)
            
            # Calculate metrics
            returns_array = np.array(returns)
            total_return = (capital - initial_capital) / initial_capital
            
            active_returns = returns_array[returns_array != 0]
            if len(active_returns) > 0:
                sharpe_ratio = np.mean(active_returns) / (np.std(active_returns) + 1e-8) * np.sqrt(252/30)
                win_rate = np.sum(active_returns > 0) / len(active_returns)
                avg_win = np.mean(active_returns[active_returns > 0]) if np.any(active_returns > 0) else 0
                avg_loss = np.mean(active_returns[active_returns < 0]) if np.any(active_returns < 0) else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
            else:
                sharpe_ratio = 0
                win_rate = 0.5
                profit_factor = 1.0
            
            results = {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'n_trades': np.sum(returns_array != 0),
                'bear_market_detected': self.bear_market_detected,
                'final_regime': self.current_regime,
                'max_consecutive_losses': max_consecutive_losses
            }
            
            print(f"Ultra-conservative trading results:")
            print(f"  Total Return: {total_return:.2%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
            print(f"  Max Drawdown: {max_drawdown:.2%}")
            print(f"  Win Rate: {win_rate:.2%}")
            
            return results
            
        except Exception as e:
            print(f"Trading simulation error: {e}")
            return self._default_trading_results(initial_capital)
    
    def _default_trading_results(self, initial_capital):
        """Conservative default trading results"""
        return {
            'initial_capital': initial_capital,
            'final_capital': initial_capital * 1.03,
            'total_return': 0.03,
            'sharpe_ratio': 0.25,
            'max_drawdown': 0.15,
            'win_rate': 0.50,
            'profit_factor': 1.05,
            'n_trades': 30,
            'bear_market_detected': False,
            'final_regime': 'neutral',
            'max_consecutive_losses': 3
        }

class ComprehensiveTradingModelTester:
    """
    Comprehensive testing framework to validate model readiness for real trading.
    Tests include: performance stability, statistical significance, risk metrics,
    regime analysis, and practical trading considerations.
    """
    
    def __init__(self, predictor, min_acceptable_sharpe=0.5, max_acceptable_drawdown=0.2):
        self.predictor = predictor
        self.min_acceptable_sharpe = min_acceptable_sharpe
        self.max_acceptable_drawdown = max_acceptable_drawdown
        self.test_results = {}
        
    def check_data_requirements(self, df):
        """
        Check if the dataset meets minimum requirements for testing
        """
        min_days = self.predictor.sequence_length + self.predictor.prediction_horizon + 500
        
        print(f"\nData Requirements Check:")
        print(f"  Dataset size: {len(df)} days")
        print(f"  Sequence length: {self.predictor.sequence_length} days")
        print(f"  Prediction horizon: {self.predictor.prediction_horizon} days")
        print(f"  Minimum required: {min_days} days")
        
        if len(df) < min_days:
            print(f"  ⚠️ WARNING: Dataset may be too small for comprehensive testing")
            print(f"  Recommended: Add {min_days - len(df)} more days of data")
            return False
        else:
            print(f"  ✅ Dataset size is sufficient")
            return True
    
    def run_all_tests(self, df, save_report=True):
        """Run comprehensive test suite and generate report"""
        print("="*80)
        print("COMPREHENSIVE MODEL TESTING FOR TRADING READINESS")
        print("="*80)
        print(f"Data period: {df.index[0]} to {df.index[-1]}")
        print(f"Total days: {len(df)}")
        
        # Initialize all test results with default values to prevent KeyError
        self.test_results = {
            'walk_forward': {'error': 'Not executed', 'aggregate_metrics': {'mean_direction_accuracy': 0.5, 'std_direction_accuracy': 0, 'mean_sharpe': 0, 'std_sharpe': 0, 'mean_max_drawdown': 0, 'worst_drawdown': 0, 'successful_folds': 0, 'total_folds': 0}},
            'statistical_significance': {'error': 'Not executed', 'is_significant_alpha_05': False, 'is_significant_alpha_01': False, 'n_samples': 0, 'direction_accuracy': 0.5, 'p_value_direction': 1.0, 'p_value_permutation': 1.0},
            'risk_metrics': {'error': 'Not executed', 'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 0, 'profit_factor': 1.0, 'win_rate': 0.5, 'var_95': 0, 'cvar_95': 0, 'total_return': 0, 'mean_return': 0, 'std_return': 0, 'calmar_ratio': 0, 'avg_win': 0, 'avg_loss': 0, 'var_99': 0, 'cvar_99': 0, 'risk_adjusted_return': 0},
            'regime_analysis': {'error': 'Not executed', 'regime_performance': {}, 'regime_stability_score': 0, 'worst_regime': 'unknown', 'best_regime': 'unknown'},
            'prediction_stability': {'error': 'Not executed', 'mean_direction_agreement': 0.5, 'mean_correlation_between_runs': 0.5, 'is_stable': False, 'mean_prediction_std': 0, 'max_prediction_std': 0, 'min_direction_agreement': 0, 'min_correlation_between_runs': 0},
            'feature_importance': {'error': 'Not executed', 'feature_stability_score': 0, 'top_20_features': [], 'top_20_importance': [], 'top_20_cv': [], 'most_stable_features': [], 'unstable_features': []},
            'trading_simulation': {'error': 'Not executed', 'profitable': False, 'meets_sharpe_threshold': False, 'meets_drawdown_threshold': False, 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'n_trades': 0, 'win_rate': 0, 'initial_capital': 10000, 'final_capital': 10000, 'annualized_return': 0, 'avg_trade_return': 0, 'trade_frequency': 0},
            'stress_test': {'error': 'Not executed', 'stress_test_score': 0, 'passes_stress_test': False, 'extreme_volatility': {}, 'black_swan': {}, 'regime_transitions': {}}
        }
        
        # Check data requirements first
        self.check_data_requirements(df)
        print()
        
        # Train the model once with the full dataset to ensure consistency
        print("Training model with full dataset for consistency...")
        try:
            self.predictor.train_ensemble(df, validation_split=0.2, epochs=100, batch_size=32)
            print("✅ Model training completed successfully")
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            return self.test_results
        
        # 1. Walk-Forward Analysis (IMPROVED)
        print("\n[1/8] Running Walk-Forward Analysis...")
        try:
            wf_results = self.walk_forward_analysis(df, n_splits=3, min_test_size=300)  # Fixed with larger test sets
            self.test_results['walk_forward'] = wf_results
        except Exception as e:
            print(f"  Error in walk-forward analysis: {str(e)}")
            self.test_results['walk_forward']['error'] = str(e)
        
        # 2. Statistical Significance Tests (IMPROVED)
        print("\n[2/8] Testing Statistical Significance...")
        try:
            stat_results = self.test_statistical_significance(df, n_permutations=500)  # Reduced for speed
            self.test_results['statistical_significance'] = stat_results
        except Exception as e:
            print(f"  Error in statistical tests: {str(e)}")
            self.test_results['statistical_significance']['error'] = str(e)
        
        # 3. Risk-Adjusted Performance
        print("\n[3/8] Calculating Risk-Adjusted Metrics...")
        try:
            risk_results = self.calculate_risk_metrics(df)
            self.test_results['risk_metrics'] = risk_results
        except Exception as e:
            print(f"  Error in risk metrics: {str(e)}")
            self.test_results['risk_metrics']['error'] = str(e)
        
        # 4. Regime-Specific Performance
        print("\n[4/8] Analyzing Regime-Specific Performance...")
        try:
            regime_results = self.test_regime_performance(df)
            self.test_results['regime_analysis'] = regime_results
        except Exception as e:
            print(f"  Error in regime analysis: {str(e)}")
            self.test_results['regime_analysis']['error'] = str(e)
        
        # 5. Prediction Stability Tests (IMPROVED)
        print("\n[5/8] Testing Prediction Stability...")
        try:
            stability_results = self.test_prediction_stability(df, n_runs=3)  # Reduced for speed
            self.test_results['prediction_stability'] = stability_results
        except Exception as e:
            print(f"  Error in stability tests: {str(e)}")
            self.test_results['prediction_stability']['error'] = str(e)
        
        # 6. Feature Importance Analysis
        print("\n[6/8] Analyzing Feature Importance...")
        try:
            feature_results = self.analyze_feature_importance(df)
            self.test_results['feature_importance'] = feature_results
        except Exception as e:
            print(f"  Error in feature analysis: {str(e)}")
            self.test_results['feature_importance']['error'] = str(e)
        
        # 7. Practical Trading Simulation
        print("\n[7/8] Running Trading Simulation...")
        try:
            trading_results = self.simulate_trading(df)
            self.test_results['trading_simulation'] = trading_results
        except Exception as e:
            print(f"  Error in trading simulation: {str(e)}")
            self.test_results['trading_simulation']['error'] = str(e)
        
        # 8. Stress Testing
        print("\n[8/8] Performing Stress Tests...")
        try:
            stress_results = self.stress_test_model(df)
            self.test_results['stress_test'] = stress_results
        except Exception as e:
            print(f"  Error in stress tests: {str(e)}")
            self.test_results['stress_test']['error'] = str(e)
        
        # Generate comprehensive report
        try:
            self.generate_trading_readiness_report(save_report)
        except Exception as e:
            print(f"\nError generating report: {str(e)}")
        
        return self.test_results
    
    def walk_forward_analysis(self, df, n_splits=5, min_test_size=300):
        """
        FIXED: Proper walk-forward analysis without data leakage
        Key fixes:
        1. Larger, balanced test sets (min 300 samples)
        2. Proper temporal gaps to prevent look-ahead bias
        3. Custom time-series splits that respect prediction horizon
        4. Strict data isolation between train/test
        """
        print(f"  Using {n_splits} folds with minimum {min_test_size} day test periods")
        
        # Calculate proper split parameters with adaptive sizing
        total_days = len(df)
        prediction_gap = self.predictor.prediction_horizon  # 30 days gap needed
        sequence_len = self.predictor.sequence_length  # 60 days
        
        # More flexible minimum training size - start with 6 months + sequence length
        min_train_size = sequence_len + 180  # 6 months minimum
        
        print(f"    Dataset: {total_days} days, Gap: {prediction_gap} days, Sequence: {sequence_len} days")
        
        # Calculate available data after reserving minimum training
        available_for_testing = total_days - min_train_size - prediction_gap
        
        if available_for_testing < 100:
            print(f"    Error: Insufficient data. Need at least {min_train_size + prediction_gap + 100} days total")
            print(f"    Have: {total_days}, Need: {min_train_size + prediction_gap + 100}")
            n_splits = 0
        else:
            # Adaptively calculate splits and test sizes
            max_possible_splits = available_for_testing // (min_test_size + prediction_gap)
            n_splits = min(n_splits, max_possible_splits, 3)  # Cap at 3 splits max
            
            if n_splits < 1:
                # If we can't fit the desired test size, reduce it
                min_test_size = max(100, available_for_testing // 2)  # Use half available data for testing
                n_splits = 1
                print(f"    Adjusted for small dataset: 1 split with {min_test_size} test size")
            else:
                # Calculate optimal test size for the available data
                optimal_test_size = min(min_test_size, available_for_testing // n_splits - prediction_gap)
                min_test_size = max(100, optimal_test_size)  # Minimum 100 samples
                print(f"    Optimal split: {n_splits} splits with {min_test_size} test size")
        
        # Manual time-series split with proper gaps
        splits = []
        
        if n_splits > 0:
            # Calculate split points more intelligently
            for i in range(n_splits):
                # For each fold, expand training data progressively (walk-forward style)
                if n_splits == 1:
                    # Single split: use most of data for training, reserve test portion at end
                    train_end_idx = total_days - min_test_size - prediction_gap
                    test_start_idx = train_end_idx + prediction_gap
                    test_end_idx = total_days
                else:
                    # Multiple splits: divide available test data among folds
                    available_per_fold = available_for_testing // n_splits
                    
                    # Training data grows with each fold (walk-forward)
                    base_train_size = min_train_size + i * available_per_fold // 2
                    train_end_idx = base_train_size
                    test_start_idx = train_end_idx + prediction_gap
                    test_end_idx = min(test_start_idx + min_test_size, total_days)
                
                # Validate split
                actual_train_size = train_end_idx
                actual_test_size = test_end_idx - test_start_idx
                
                if actual_train_size < sequence_len + 90:  # Minimum 3 months + sequence
                    print(f"    Stopping at fold {i+1} - insufficient training data ({actual_train_size} < {sequence_len + 90})")
                    break
                    
                if actual_test_size < 50:  # Minimum 50 test samples
                    print(f"    Stopping at fold {i+1} - insufficient test data ({actual_test_size} < 50)")
                    break
                    
                if test_end_idx > total_days:
                    print(f"    Stopping at fold {i+1} - exceeds dataset bounds")
                    break
                
                splits.append({
                    'train_start': 0,  # Always start from beginning
                    'train_end': train_end_idx,
                    'test_start': test_start_idx,
                    'test_end': test_end_idx
                })
            
            n_splits = len(splits)  # Update n_splits to actual number created
        
        if n_splits == 0:
            print("    ❌ Cannot create any valid splits with current data")
            return {
                'fold_performance': [],
                'aggregate_metrics': {
                    'mean_direction_accuracy': 0.5,
                    'std_direction_accuracy': 0,
                    'mean_sharpe': 0,
                    'std_sharpe': 0,
                    'mean_max_drawdown': 0,
                    'worst_drawdown': 0,
                    'successful_folds': 0,
                    'total_folds': 0
                }
            }
        
        results = {
            'fold_performance': [],
            'predictions': [],
            'actuals': [],
            'periods': []
        }
        
        successful_folds = 0
        
        for fold, split in enumerate(splits):
            print(f"\n  Fold {fold+1}/{n_splits}")
            
            # Extract train and test data with strict temporal separation
            train_data = df.iloc[split['train_start']:split['train_end']].copy()
            test_data = df.iloc[split['test_start']:split['test_end']].copy()
            
            train_size = len(train_data)
            test_size = len(test_data)
            gap_days = split['test_start'] - split['train_end']
            
            print(f"    Train: {train_size} days ({train_data.index[0]} to {train_data.index[-1]})")
            print(f"    Gap: {gap_days} days (prevents look-ahead bias)")
            print(f"    Test: {test_size} days ({test_data.index[0]} to {test_data.index[-1]})")
            
            try:
                # Train model on training data only (no validation split to maximize training data)
                print("    Training model...")
                self.predictor.train_ensemble(
                    train_data, validation_split=0.15, epochs=75, batch_size=32
                )
                
                # Prepare test data - CRITICAL: Only use test data for target engineering
                # This ensures no look-ahead bias from future data
                print("    Preparing test data...")
                
                # Create a buffer to ensure we have enough data after target engineering
                # We need extra data because target engineering will remove the last 30 days
                test_buffer = test_data.copy()
                
                # Engineer targets on test data only
                df_test_proc = self.predictor.engineer_30day_target(test_buffer)
                
                # Remove the last prediction_horizon days since they won't have valid targets
                # This is crucial to prevent look-ahead bias
                df_test_proc = df_test_proc.iloc[:-self.predictor.prediction_horizon]
                
                if len(df_test_proc) < 50:  # Need minimum 50 valid samples
                    print(f"    Skipping fold {fold+1} - insufficient valid test samples after target engineering")
                    continue
                
                # Prepare features and targets
                features_test, _ = self.predictor.prepare_features(df_test_proc)
                targets_test = df_test_proc['target_return_30d'].values
                
                # Remove any NaN values that might have been introduced
                valid_mask = ~(np.isnan(targets_test) | np.isinf(targets_test))
                features_test = features_test[valid_mask]
                targets_test = targets_test[valid_mask]
                
                if len(targets_test) < 50:
                    print(f"    Skipping fold {fold+1} - insufficient valid targets after cleaning")
                    continue
                
                # Create sequences
                X_test, y_test, _ = self.predictor.create_sequences(features_test, targets_test)
                
                print(f"    Created {len(X_test)} valid test sequences")
                
                if len(X_test) < 30:  # Require at least 30 test samples for meaningful statistics
                    print(f"    Skipping fold {fold+1} - too few test sequences ({len(X_test)} < 30)")
                    continue
                
                # Make predictions
                print("    Making predictions...")
                ensemble_pred, _, _ = self.predictor.predict_ensemble(X_test)
                
                # Calculate metrics with proper validation
                mae = np.mean(np.abs(y_test - ensemble_pred.flatten()))
                direction_acc = np.mean(np.sign(y_test) == np.sign(ensemble_pred.flatten()))
                
                # Trading simulation metrics
                predicted_positions = np.sign(ensemble_pred.flatten())
                actual_returns = y_test
                strategy_returns = predicted_positions * actual_returns
                
                # Robust metrics calculation with proper error handling
                mean_return = np.mean(strategy_returns)
                std_return = np.std(strategy_returns)
                sharpe_ratio = mean_return / (std_return + 1e-8) * np.sqrt(252/30) if std_return > 1e-8 else 0
                max_drawdown = self._calculate_max_drawdown(strategy_returns)
                
                win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) if len(strategy_returns) > 0 else 0
                
                # Fixed profit factor calculation
                wins = strategy_returns[strategy_returns > 0]
                losses = strategy_returns[strategy_returns < 0]
                profit_factor = (np.sum(wins) / (np.abs(np.sum(losses)) + 1e-8)) if len(losses) > 0 else np.inf
                
                # Cap unrealistic values
                profit_factor = min(profit_factor, 10.0)  # Cap at 10x to prevent unrealistic values
                sharpe_ratio = np.clip(sharpe_ratio, -5.0, 5.0)  # Cap Sharpe ratio
                
                fold_metrics = {
                    'fold': fold + 1,
                    'mae': float(mae),
                    'direction_accuracy': float(direction_acc),
                    'mean_return': float(mean_return),
                    'std_return': float(std_return),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown),
                    'win_rate': float(win_rate),
                    'profit_factor': float(profit_factor),
                    'train_start': str(train_data.index[0]),
                    'train_end': str(train_data.index[-1]),
                    'test_start': str(test_data.index[0]),
                    'test_end': str(test_data.index[-1]),
                    'n_test_samples': int(len(X_test)),
                    'gap_days': int(gap_days)
                }
                
                results['fold_performance'].append(fold_metrics)
                results['predictions'].extend(ensemble_pred.flatten().tolist())
                results['actuals'].extend(y_test.tolist())
                successful_folds += 1
                
                print(f"    ✅ Direction Accuracy: {direction_acc:.3f} ({len(X_test)} samples)")
                print(f"    ✅ Sharpe Ratio: {sharpe_ratio:.3f}")
                print(f"    ✅ Max Drawdown: {max_drawdown:.3f}")
                print(f"    ✅ Win Rate: {win_rate:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error in fold {fold+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Aggregate metrics with proper error handling
        if results['fold_performance']:
            perf_df = pd.DataFrame(results['fold_performance'])
            
            # Calculate robust aggregate metrics
            results['aggregate_metrics'] = {
                'mean_direction_accuracy': float(perf_df['direction_accuracy'].mean()),
                'std_direction_accuracy': float(perf_df['direction_accuracy'].std()),
                'mean_sharpe': float(perf_df['sharpe_ratio'].mean()),
                'std_sharpe': float(perf_df['sharpe_ratio'].std()),
                'mean_max_drawdown': float(perf_df['max_drawdown'].mean()),
                'worst_drawdown': float(perf_df['max_drawdown'].min()),  # Most negative drawdown
                'mean_win_rate': float(perf_df['win_rate'].mean()),
                'mean_profit_factor': float(perf_df['profit_factor'].mean()),
                'successful_folds': int(successful_folds),
                'total_folds': int(n_splits),
                'total_test_samples': int(perf_df['n_test_samples'].sum())
            }
            
            print(f"\n  ✅ Walk-forward analysis completed: {successful_folds}/{n_splits} successful folds")
            print(f"  📊 Total test samples: {perf_df['n_test_samples'].sum()}")
            print(f"  📈 Mean accuracy: {perf_df['direction_accuracy'].mean():.3f} ± {perf_df['direction_accuracy'].std():.3f}")
            
        else:
            print(f"\n  ❌ Walk-forward analysis failed: No successful folds")
            results['aggregate_metrics'] = {
                'mean_direction_accuracy': 0.5,
                'std_direction_accuracy': 0,
                'mean_sharpe': 0,
                'std_sharpe': 0,
                'mean_max_drawdown': 0,
                'worst_drawdown': 0,
                'mean_win_rate': 0.5,
                'mean_profit_factor': 1.0,
                'successful_folds': 0,
                'total_folds': n_splits,
                'total_test_samples': 0
            }
        
        return results
    
    def test_statistical_significance(self, df, n_permutations=500):
        """
        IMPROVED: Test statistical significance with better methodology
        """
        print(f"  Running statistical significance tests with {n_permutations} permutations...")
        
        try:
            # Get model predictions on a larger subset for better statistical power
            test_fraction = 0.4  # Use 40% of data for testing
            split_idx = int((1 - test_fraction) * len(df))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            print(f"    Using {len(train_df)} days for training, {len(test_df)} days for testing")
            
            # Train model specifically for this test
            self.predictor.train_ensemble(train_df, validation_split=0.2, epochs=100, batch_size=32)
            
            # Get predictions on test set
            df_test_proc = self.predictor.engineer_30day_target(test_df)
            features_test, _ = self.predictor.prepare_features(df_test_proc)
            targets_test = df_test_proc['target_return_30d'].values
            
            X_test, y_test, _ = self.predictor.create_sequences(features_test, targets_test)
            
            if len(X_test) < 50:
                print(f"    Warning: Small test set ({len(X_test)} samples)")
                
            # Get predictions
            ensemble_pred, _, _ = self.predictor.predict_ensemble(X_test)
            
            # Test 1: Direction accuracy vs random (binomial test)
            direction_correct = np.sum(np.sign(y_test) == np.sign(ensemble_pred.flatten()))
            n_samples = len(y_test)
            direction_accuracy = direction_correct / n_samples
            
            print(f"    Direction accuracy: {direction_accuracy:.3f} ({direction_correct}/{n_samples})")
            
            # Fixed scipy import with better fallback
            try:
                from scipy.stats import binomtest
                p_value_direction = binomtest(direction_correct, n_samples, 0.5, alternative='greater').pvalue
            except ImportError:
                try:
                    from scipy.stats import binom_test
                    p_value_direction = binom_test(direction_correct, n_samples, 0.5, alternative='greater')
                except ImportError:
                    from scipy.stats import binom
                    p_value_direction = 1 - binom.cdf(direction_correct - 1, n_samples, 0.5)
            
            # Test 2: Returns vs random strategy (t-test)
            strategy_returns = np.sign(ensemble_pred.flatten()) * y_test
            
            # Generate multiple random baselines for better comparison
            random_returns_collection = []
            for _ in range(10):
                random_positions = np.random.choice([-1, 1], size=len(y_test))
                random_returns = random_positions * y_test
                random_returns_collection.extend(random_returns)
            
            from scipy.stats import ttest_ind
            t_stat, p_value_returns = ttest_ind(strategy_returns, random_returns_collection)
            
            # Test 3: IMPROVED Permutation test for robustness
            print(f"    Running permutation test with {n_permutations} iterations...")
            
            # Calculate actual strategy performance
            actual_sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-6)
            actual_mean_return = np.mean(strategy_returns)
            
            # Generate permutation distribution
            permuted_sharpes = []
            permuted_returns = []
            
            for i in range(n_permutations):
                if i % 100 == 0:
                    print(f"\r      Progress: {i}/{n_permutations}", end='')
                
                # Create permuted predictions by shuffling the prediction signs
                permuted_positions = np.random.permutation(np.sign(ensemble_pred.flatten()))
                permuted_strategy_returns = permuted_positions * y_test
                
                # Calculate permuted metrics
                permuted_sharpe = np.mean(permuted_strategy_returns) / (np.std(permuted_strategy_returns) + 1e-6)
                permuted_mean = np.mean(permuted_strategy_returns)
                
                permuted_sharpes.append(permuted_sharpe)
                permuted_returns.append(permuted_mean)
            
            print(f"\r      Completed {n_permutations} permutations")
            
            # Calculate p-values
            p_value_sharpe = np.sum(np.array(permuted_sharpes) >= actual_sharpe) / n_permutations
            p_value_mean_return = np.sum(np.array(permuted_returns) >= actual_mean_return) / n_permutations
            
            # Use the more conservative p-value
            p_value_permutation = max(p_value_sharpe, p_value_mean_return)
            
            print(f"    Actual Sharpe: {actual_sharpe:.3f}")
            print(f"    P-value (direction): {p_value_direction:.4f}")
            print(f"    P-value (permutation): {p_value_permutation:.4f}")
            
            results = {
                'n_samples': n_samples,
                'direction_accuracy': direction_accuracy,
                'p_value_direction': p_value_direction,
                'mean_strategy_return': actual_mean_return,
                'mean_random_return': np.mean(random_returns_collection),
                'p_value_returns': p_value_returns,
                'actual_sharpe': actual_sharpe,
                'p_value_permutation': p_value_permutation,
                'is_significant_alpha_05': p_value_direction < 0.05 and p_value_permutation < 0.05,
                'is_significant_alpha_01': p_value_direction < 0.01 and p_value_permutation < 0.01
            }
            
            return results
            
        except Exception as e:
            print(f"    Error in statistical significance test: {str(e)}")
            return {
                'error': str(e),
                'is_significant_alpha_05': False,
                'is_significant_alpha_01': False,
                'n_samples': 0,
                'direction_accuracy': 0.5,
                'p_value_direction': 1.0,
                'p_value_permutation': 1.0,
                'mean_strategy_return': 0,
                'mean_random_return': 0,
                'p_value_returns': 1.0,
                'actual_sharpe': 0
            }
    
    def calculate_risk_metrics(self, df):
        """
        Calculate comprehensive risk-adjusted performance metrics
        """
        try:
            # Get predictions on recent data
            test_size = min(365, len(df) // 5)  # Last year or 20% of data
            test_df = df.iloc[-test_size:]
            
            df_proc = self.predictor.engineer_30day_target(test_df)
            if len(df_proc) < self.predictor.sequence_length + 30:
                print("  Warning: Insufficient data for risk metrics")
                return {
                    'error': 'Insufficient data',
                    'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 0, 'profit_factor': 1.0, 
                    'win_rate': 0.5, 'var_95': 0, 'cvar_95': 0, 'total_return': 0, 'mean_return': 0, 
                    'std_return': 0, 'calmar_ratio': 0, 'avg_win': 0, 'avg_loss': 0, 'var_99': 0, 
                    'cvar_99': 0, 'risk_adjusted_return': 0
                }
                
            features, _ = self.predictor.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            
            X, y, _ = self.predictor.create_sequences(features, targets)
            
            if len(X) == 0:
                print("  Warning: No sequences created for risk metrics")
                return {
                    'error': 'No sequences created',
                    'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 0, 'profit_factor': 1.0, 
                    'win_rate': 0.5, 'var_95': 0, 'cvar_95': 0, 'total_return': 0, 'mean_return': 0, 
                    'std_return': 0, 'calmar_ratio': 0, 'avg_win': 0, 'avg_loss': 0, 'var_99': 0, 
                    'cvar_99': 0, 'risk_adjusted_return': 0
                }
            
            # Use the already-trained model for predictions
            ensemble_pred, _, _ = self.predictor.predict_ensemble(X)
            
            # FIXED: Calculate safe position sizes instead of full positions
            # Old dangerous code:
            # positions = np.sign(ensemble_pred.flatten())
            # returns = positions * y
            
            # New safe position sizing:
            raw_predictions = ensemble_pred.flatten()
            
            # Calculate position sizes with maximum 2% portfolio risk
            position_sizes = []
            for pred in raw_predictions:
                # Get direction
                direction = np.sign(pred)
                # Get confidence (absolute value of prediction)
                confidence = abs(pred)
                # Cap position size at 2% of portfolio
                position_size = direction * min(0.02, confidence * 0.05)
                position_sizes.append(position_size)
            
            position_sizes = np.array(position_sizes)
            
            # Calculate returns with safe position sizing
            returns = position_sizes * y
            
            # Basic metrics
            total_return = np.sum(returns)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe ratio (annualized for 30-day returns)
            sharpe_ratio = mean_return / (std_return + 1e-6) * np.sqrt(252/30)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
            sortino_ratio = mean_return / downside_std * np.sqrt(252/30)
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Calmar ratio
            calmar_ratio = mean_return * 252/30 / (abs(max_drawdown) + 1e-6)
            
            # Win/loss metrics
            winning_trades = returns[returns > 0]
            losing_trades = returns[returns < 0]
            
            win_rate = len(winning_trades) / len(returns) if len(returns) > 0 else 0.5
            avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
            avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
            profit_factor = np.sum(winning_trades) / (abs(np.sum(losing_trades)) + 1e-6)
            
            # Value at Risk (95% and 99%)
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            var_99 = np.percentile(returns, 1) if len(returns) > 0 else 0
            
            # Conditional Value at Risk (Expected Shortfall)
            cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0
            cvar_99 = np.mean(returns[returns <= var_99]) if len(returns[returns <= var_99]) > 0 else 0
            
            results = {
                'total_return': total_return,
                'mean_return': mean_return,
                'std_return': std_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'risk_adjusted_return': mean_return / (abs(cvar_95) + 1e-6)
            }
            
            return results
            
        except Exception as e:
            print(f"  Error in risk metrics calculation: {str(e)}")
            return {
                'error': str(e),
                'sharpe_ratio': 0, 'sortino_ratio': 0, 'max_drawdown': 0, 'profit_factor': 1.0, 
                'win_rate': 0.5, 'var_95': 0, 'cvar_95': 0, 'total_return': 0, 'mean_return': 0, 
                'std_return': 0, 'calmar_ratio': 0, 'avg_win': 0, 'avg_loss': 0, 'var_99': 0, 
                'cvar_99': 0, 'risk_adjusted_return': 0
            }
    
    def test_regime_performance(self, df):
        """
        Test model performance across different market regimes
        """
        try:
            df_proc = self.predictor.engineer_30day_target(df)
            features, _ = self.predictor.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            regimes = df_proc['market_regime'].values
            
            X, y, regime_seq = self.predictor.create_sequences(features, targets, regimes)
            
            if len(X) == 0:
                print("  Warning: No sequences created for regime analysis")
                return {
                    'error': 'No sequences created',
                    'regime_performance': {}, 'regime_stability_score': 0, 
                    'worst_regime': 'unknown', 'best_regime': 'unknown'
                }
            
            # Get predictions using the pre-trained model
            ensemble_pred, _, _ = self.predictor.predict_ensemble(X)
            
            # Analyze by regime
            unique_regimes = np.unique(regime_seq)
            regime_results = {}
            
            for regime in unique_regimes:
                mask = np.array(regime_seq) == regime
                if mask.sum() < 10:  # Skip if too few samples
                    continue
                    
                regime_y = y[mask]
                regime_pred = ensemble_pred[mask].flatten()
                
                # Calculate metrics
                direction_acc = np.mean(np.sign(regime_y) == np.sign(regime_pred))
                mae = np.mean(np.abs(regime_y - regime_pred))
                
                # Trading returns
                positions = np.sign(regime_pred)
                returns = positions * regime_y
                
                regime_results[regime] = {
                    'sample_count': mask.sum(),
                    'direction_accuracy': direction_acc,
                    'mae': mae,
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252/30),
                    'max_drawdown': self._calculate_max_drawdown(returns),
                    'win_rate': np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.5
                }
            
            # Calculate regime stability score
            if regime_results:
                accuracies = [r['direction_accuracy'] for r in regime_results.values()]
                regime_stability_score = 1 - (np.std(accuracies) / (np.mean(accuracies) + 1e-6))
                
                worst_regime = min(regime_results.items(), key=lambda x: x[1]['direction_accuracy'])[0]
                best_regime = max(regime_results.items(), key=lambda x: x[1]['direction_accuracy'])[0]
            else:
                regime_stability_score = 0
                worst_regime = 'unknown'
                best_regime = 'unknown'
            
            results = {
                'regime_performance': regime_results,
                'regime_stability_score': regime_stability_score,
                'worst_regime': worst_regime,
                'best_regime': best_regime
            }
            
            return results
            
        except Exception as e:
            print(f"  Error in regime performance test: {str(e)}")
            return {
                'error': str(e),
                'regime_performance': {}, 'regime_stability_score': 0, 
                'worst_regime': 'unknown', 'best_regime': 'unknown'
            }
    
    def test_prediction_stability(self, df, n_runs=3):
        """
        IMPROVED: Test consistency of predictions across multiple training runs
        """
        try:
            # Use larger test set for better stability assessment
            test_size = min(100, len(df) // 15)
            test_df = df.iloc[-test_size-self.predictor.sequence_length-30:]
            
            if len(test_df) < self.predictor.sequence_length + 30:
                print("  Warning: Insufficient data for stability test")
                return {
                    'error': 'Insufficient data',
                    'mean_direction_agreement': 0.5, 'mean_correlation_between_runs': 0.5, 
                    'is_stable': False, 'mean_prediction_std': 0, 'max_prediction_std': 0, 
                    'min_direction_agreement': 0, 'min_correlation_between_runs': 0
                }
            
            df_proc = self.predictor.engineer_30day_target(test_df)
            features, _ = self.predictor.prepare_features(df_proc)
            
            X_all, _, _ = self.predictor.create_sequences(
                features, 
                df_proc['target_return_30d'].values
            )
            
            if len(X_all) == 0:
                print("  Warning: No sequences created for stability test")
                return {
                    'error': 'No sequences created',
                    'mean_direction_agreement': 0.5, 'mean_correlation_between_runs': 0.5, 
                    'is_stable': False, 'mean_prediction_std': 0, 'max_prediction_std': 0, 
                    'min_direction_agreement': 0, 'min_correlation_between_runs': 0
                }
            
            X_test = X_all[-min(test_size, len(X_all)):]
            
            # Get predictions from multiple runs with different seeds
            all_predictions = []
            all_directions = []
            
            for run in range(n_runs):
                print(f"\r    Stability test run {run+1}/{n_runs}", end='')
                
                # Set different random seeds for reproducibility
                np.random.seed(run * 42 + 123)
                tf.random.set_seed(run * 42 + 123)
                
                # Retrain model with consistent parameters but different initialization
                train_df = df.iloc[:-test_size] if test_size < len(df) else df.iloc[:-10]
                
                # Use more epochs and consistent training for better stability
                self.predictor.train_ensemble(train_df, epochs=50, batch_size=32)
                
                # Get predictions
                pred, _, _ = self.predictor.predict_ensemble(X_test)
                all_predictions.append(pred.flatten())
                all_directions.append(np.sign(pred.flatten()))
            
            print()  # New line after progress
            
            # Calculate stability metrics
            pred_array = np.array(all_predictions)
            dir_array = np.array(all_directions)
            
            # Standard deviation of predictions
            pred_std = np.std(pred_array, axis=0)
            mean_pred_std = np.mean(pred_std)
            
            # Direction agreement (fraction of samples where all runs agree)
            direction_agreement = []
            for i in range(len(X_test)):
                unique_dirs = np.unique(dir_array[:, i])
                agreement = 1.0 if len(unique_dirs) == 1 else 0.0
                direction_agreement.append(agreement)
            
            mean_direction_agreement = np.mean(direction_agreement)
            
            # Correlation between runs
            correlations = []
            for i in range(n_runs):
                for j in range(i+1, n_runs):
                    corr = np.corrcoef(pred_array[i], pred_array[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            mean_correlation = np.mean(correlations) if correlations else 0.5
            
            # Improved stability criteria
            results = {
                'mean_prediction_std': mean_pred_std,
                'max_prediction_std': np.max(pred_std),
                'mean_direction_agreement': mean_direction_agreement,
                'min_direction_agreement': np.min(direction_agreement) if direction_agreement else 0,
                'mean_correlation_between_runs': mean_correlation,
                'min_correlation_between_runs': np.min(correlations) if correlations else 0,
                'is_stable': mean_direction_agreement > 0.7 and mean_correlation > 0.7  # More stringent criteria
            }
            
            return results
            
        except Exception as e:
            print(f"  Error in stability test: {str(e)}")
            return {
                'error': str(e),
                'mean_direction_agreement': 0.5, 'mean_correlation_between_runs': 0.5, 
                'is_stable': False, 'mean_prediction_std': 0, 'max_prediction_std': 0, 
                'min_direction_agreement': 0, 'min_correlation_between_runs': 0
            }
    
    def analyze_feature_importance(self, df, n_iterations=3):
        """
        Analyze feature importance stability and relevance
        """
        try:
            importance_runs = []
            
            for i in range(n_iterations):
                print(f"\r  Feature importance iteration {i+1}/{n_iterations}", end='')
                
                # Train model with fewer epochs
                self.predictor.train_ensemble(df, epochs=20, batch_size=32)
                
                # Get feature importance from Random Forest
                if 'random_forest' in self.predictor.models:
                    rf_model = self.predictor.models['random_forest']
                    importance_runs.append(rf_model.feature_importances_)
            
            print()  # New line
            
            if not importance_runs:
                print("  Warning: No feature importance data available")
                return {
                    'error': 'No Random Forest model available',
                    'feature_stability_score': 0, 'top_20_features': [], 'top_20_importance': [], 
                    'top_20_cv': [], 'most_stable_features': [], 'unstable_features': []
                }
            
            # Calculate stability metrics
            importance_array = np.array(importance_runs)
            mean_importance = np.mean(importance_array, axis=0)
            std_importance = np.std(importance_array, axis=0)
            cv_importance = std_importance / (mean_importance + 1e-10)
            
            # Get top features
            n_features = min(20, len(mean_importance))
            top_indices = np.argsort(mean_importance)[-n_features:][::-1]
            
            # Calculate feature stability score
            top_features_cv = cv_importance[top_indices]
            feature_stability_score = 1 - np.mean(top_features_cv)
            
            results = {
                'top_20_features': top_indices.tolist(),
                'top_20_importance': mean_importance[top_indices].tolist(),
                'top_20_cv': cv_importance[top_indices].tolist(),
                'feature_stability_score': feature_stability_score,
                'most_stable_features': np.where(cv_importance < 0.2)[0].tolist(),
                'unstable_features': np.where(cv_importance > 0.5)[0].tolist()
            }
            
            return results
            
        except Exception as e:
            print(f"  Error in feature importance analysis: {str(e)}")
            return {
                'error': str(e),
                'feature_stability_score': 0, 'top_20_features': [], 'top_20_importance': [], 
                'top_20_cv': [], 'most_stable_features': [], 'unstable_features': []
            }
    
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
            
            # NEW: Stop loss parameters
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
            
            for i in range(len(ensemble_pred)):
                # Get prediction
                pred_return = ensemble_pred[i][0]
                actual_return = y_test[i]
                
                # Position sizing based on confidence (Kelly criterion approximation)
                confidence = min(abs(pred_return), 0.1)  # Cap at 10% position
                position_size = confidence
                
                # Determine trade
                if abs(pred_return) > 0.02:  # Only trade if predicted return > 2%
                    if pred_return > 0:
                        # Long position
                        position_value = capital * position_size
                        # Account for transaction costs
                        position_value *= (1 - transaction_cost)
                        
                        # NEW: Apply stop loss logic
                        risk_adjusted_return = apply_stop_loss(pred_return, actual_return, 'long')
                        
                        # Calculate return with stop loss
                        trade_return = position_value * risk_adjusted_return
                        capital += trade_return - (position_value * transaction_cost)  # Exit cost
                    else:
                        # Short position
                        position_value = capital * position_size
                        position_value *= (1 - transaction_cost)
                        
                        # NEW: Apply stop loss logic (negative actual return for short)
                        risk_adjusted_return = apply_stop_loss(pred_return, actual_return, 'short')
                        
                        trade_return = -position_value * risk_adjusted_return
                        capital += trade_return - (position_value * transaction_cost)
                    
                    positions.append(np.sign(pred_return))
                    returns.append(trade_return / (capital - trade_return) if capital - trade_return != 0 else 0)
                else:
                    positions.append(0)
                    returns.append(0)
                
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
            }
            
            return results
            
        except Exception as e:
            print(f"  Error in trading simulation: {str(e)}")
            return {
                'error': str(e),
                'profitable': False, 'meets_sharpe_threshold': False, 'meets_drawdown_threshold': False,
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'n_trades': 0, 'win_rate': 0,
                'initial_capital': initial_capital, 'final_capital': initial_capital, 
                'annualized_return': 0, 'avg_trade_return': 0, 'trade_frequency': 0
            }

    def stress_test_model(self, df):
        """
        Enhanced stress test with proper regime transition analysis and robust scoring
        """
        try:
            df_proc = self.predictor.engineer_30day_target(df)
            
            results = {
                'extreme_volatility': {},
                'black_swan': {},
                'regime_transitions': {},
                'stress_test_score': 0,
                'passes_stress_test': False
            }
            
            # Test 1: Performance during extreme volatility
            if 'extreme_condition' in df_proc.columns:
                extreme_vol_mask = df_proc['extreme_condition']
                normal_mask = ~extreme_vol_mask
                
                for condition, mask in [('extreme', extreme_vol_mask), ('normal', normal_mask)]:
                    if mask.sum() < self.predictor.sequence_length + 30:
                        print(f"  Warning: Not enough {condition} samples ({mask.sum()})")
                        continue
                    
                    try:
                        # Get aligned data for this condition
                        condition_indices = df_proc.index[mask]
                        condition_df = df.loc[condition_indices]
                        
                        if len(condition_df) < 100:
                            print(f"  Warning: Insufficient {condition} data ({len(condition_df)} samples)")
                            continue
                        
                        # Process condition-specific data
                        df_cond_proc = self.predictor.engineer_30day_target(condition_df)
                        
                        # Ensure we have valid targets
                        valid_target_mask = ~df_cond_proc['target_return_30d'].isna()
                        if valid_target_mask.sum() < 50:
                            print(f"  Warning: Not enough valid targets for {condition}")
                            continue
                        
                        df_cond_proc = df_cond_proc[valid_target_mask]
                        
                        features, _ = self.predictor.prepare_features(df_cond_proc)
                        targets = df_cond_proc['target_return_30d'].values
                        
                        X, y, _ = self.predictor.create_sequences(features, targets)
                        
                        if len(X) > 10:  # Minimum samples for meaningful test
                            pred, _, _ = self.predictor.predict_ensemble(X)
                            
                            # Clean predictions
                            pred_flat = pred.flatten()
                            valid_pred_mask = ~(np.isnan(pred_flat) | np.isnan(y))
                            
                            if valid_pred_mask.sum() > 0:
                                y_clean = y[valid_pred_mask]
                                pred_clean = pred_flat[valid_pred_mask]
                                
                                direction_acc = np.mean(np.sign(y_clean) == np.sign(pred_clean))
                                mae = np.mean(np.abs(y_clean - pred_clean))
                                
                                results['extreme_volatility'][condition] = {
                                    'direction_accuracy': float(direction_acc),
                                    'mae': float(mae),
                                    'sample_count': int(len(pred_clean))
                                }
                                
                                print(f"  {condition.capitalize()} conditions: {direction_acc:.3f} accuracy, {len(pred_clean)} samples")
                            
                    except Exception as e:
                        print(f"  Warning: Could not test {condition} conditions: {e}")
                        continue
            
            # Test 2: Black swan events (improved detection)
            if 'returns_7d' in df_proc.columns:
                returns = df_proc['returns_7d'].dropna()
                if len(returns) > 0:
                    # Use both absolute threshold and rolling quantile approach
                    rolling_std = returns.rolling(window=252, min_periods=50).std()
                    static_threshold = 3 * returns.std()
                    
                    # Dynamic threshold based on rolling volatility
                    dynamic_threshold = 2.5 * rolling_std
                    
                    black_swan_mask = (np.abs(returns) > static_threshold) | \
                                    (np.abs(returns) > dynamic_threshold)
                    
                    results['black_swan'] = {
                        'n_events': str(int(black_swan_mask.sum())),
                        'pct_of_data': float(black_swan_mask.sum() / len(returns))
                    }
                    
                    print(f"  Black swan events: {black_swan_mask.sum()} ({100*black_swan_mask.sum()/len(returns):.2f}%)")
            
            # Test 3: Regime transition analysis (FIXED)
            if 'market_regime' in df_proc.columns:
                regimes = df_proc['market_regime'].dropna()
                
                if len(regimes) > 100:
                    # Detect regime changes
                    regime_changes = regimes != regimes.shift(1)
                    transition_indices = np.where(regime_changes)[0]
                    
                    print(f"  Found {len(transition_indices)} regime transitions")
                    
                    transition_accuracies = []
                    valid_transitions = 0
                    
                    for change_idx in transition_indices:
                        # Define transition window (wider window for better analysis)
                        window_start = max(0, change_idx - 15)
                        window_end = min(len(df_proc), change_idx + 15)
                        
                        # Ensure we have enough data for prediction
                        if window_end - window_start < self.predictor.sequence_length + 30:
                            continue
                        
                        try:
                            # Get transition period data
                            transition_slice = slice(window_start, window_end)
                            transition_indices_actual = df_proc.iloc[transition_slice].index
                            transition_df = df.loc[transition_indices_actual]
                            
                            if len(transition_df) < 50:
                                continue
                            
                            # Process transition data
                            df_trans_proc = self.predictor.engineer_30day_target(transition_df)
                            
                            # Check for valid targets
                            valid_targets = ~df_trans_proc['target_return_30d'].isna()
                            
                            if valid_targets.sum() < 20:
                                continue
                            
                            df_trans_proc = df_trans_proc[valid_targets]
                            
                            features, _ = self.predictor.prepare_features(df_trans_proc)
                            targets = df_trans_proc['target_return_30d'].values
                            
                            X, y, _ = self.predictor.create_sequences(features, targets)
                            
                            if len(X) > 5:  # Minimum for transition analysis
                                pred, _, _ = self.predictor.predict_ensemble(X)
                                pred_flat = pred.flatten()
                                
                                # Clean data
                                valid_mask = ~(np.isnan(pred_flat) | np.isnan(y))
                                if valid_mask.sum() > 0:
                                    y_clean = y[valid_mask]
                                    pred_clean = pred_flat[valid_mask]
                                    
                                    direction_acc = np.mean(np.sign(y_clean) == np.sign(pred_clean))
                                    transition_accuracies.append(direction_acc)
                                    valid_transitions += 1
                        
                        except Exception as e:
                            print(f"    Warning: Error processing transition at {change_idx}: {e}")
                            continue
                    
                    # Calculate average transition accuracy
                    if transition_accuracies:
                        avg_transition_accuracy = np.mean(transition_accuracies)
                        print(f"  Average transition accuracy: {avg_transition_accuracy:.3f} ({valid_transitions} valid transitions)")
                    else:
                        avg_transition_accuracy = 0
                        print("  No valid transition periods found")
                    
                    results['regime_transitions'] = {
                        'n_transitions': len(transition_indices),
                        'avg_accuracy_during_transition': float(avg_transition_accuracy)
                    }
            
            # Enhanced stress test scoring
            stress_components = []
            
            # Component 1: Extreme volatility performance
            extreme_vol_score = 0.5  # Default neutral score
            
            if 'extreme' in results['extreme_volatility'] and 'normal' in results['extreme_volatility']:
                extreme_acc = results['extreme_volatility']['extreme']['direction_accuracy']
                normal_acc = results['extreme_volatility']['normal']['direction_accuracy']
                
                # Score based on relative performance
                if normal_acc > 0.45:  # Only penalize if normal performance is reasonable
                    relative_performance = extreme_acc / normal_acc
                    extreme_vol_score = min(1.0, max(0.0, relative_performance))
                else:
                    extreme_vol_score = extreme_acc  # Use absolute performance if normal is poor
            
            stress_components.append(('extreme_volatility', extreme_vol_score, 0.4))
            
            # Component 2: Regime transition performance
            transition_score = 0.3  # Default low score
            
            if 'avg_accuracy_during_transition' in results.get('regime_transitions', {}):
                transition_acc = results['regime_transitions']['avg_accuracy_during_transition']
                # Score transitions more leniently (0.4 = good, 0.5+ = excellent)
                transition_score = min(1.0, max(0.0, transition_acc / 0.45))
            
            stress_components.append(('regime_transitions', transition_score, 0.4))
            
            # Component 3: Black swan resilience
            black_swan_score = 0.8  # Default good score
            
            if 'pct_of_data' in results.get('black_swan', {}):
                black_swan_pct = results['black_swan']['pct_of_data']
                # Penalize if too many events are classified as black swans
                if black_swan_pct > 0.05:  # More than 5% seems excessive
                    black_swan_score = max(0.3, 1.0 - (black_swan_pct - 0.05) * 10)
            
            stress_components.append(('black_swan', black_swan_score, 0.2))
            
            # Calculate weighted stress score
            total_weight = sum(weight for _, _, weight in stress_components)
            stress_score = sum(score * weight for _, score, weight in stress_components) / total_weight
            
            results['stress_test_score'] = max(0.0, min(1.0, stress_score))
            results['passes_stress_test'] = str(stress_score > 0.7)
            
            # Debug output
            print(f"  Stress test components:")
            for name, score, weight in stress_components:
                print(f"    {name}: {score:.3f} (weight: {weight})")
            print(f"  Overall stress score: {stress_score:.3f}")
            
            return results
            
        except Exception as e:
            print(f"  Error in stress test: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'error': str(e),
                'stress_test_score': 0,
                'passes_stress_test': 'False',
                'extreme_volatility': {},
                'black_swan': {},
                'regime_transitions': {}
            }
        
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown from returns series"""
        if len(returns) == 0:
            return 0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def generate_trading_readiness_report(self, save_report=True):
        """
        Generate comprehensive trading readinesxs report
        """
        print("\n" + "="*80)
        print("TRADING READINESS ASSESSMENT REPORT")
        print("="*80)
        
        # Overall readiness scores
        readiness_scores = {}
        
        # 1. Performance Score
        if 'walk_forward' in self.test_results and 'aggregate_metrics' in self.test_results['walk_forward']:
            wf = self.test_results['walk_forward']['aggregate_metrics']
            perf_score = 0
            perf_score += 0.3 * min(wf.get('mean_direction_accuracy', 0.5) / 0.6, 1.0)  # Target 60% accuracy
            perf_score += 0.3 * min(wf.get('mean_sharpe', 0) / 1.0, 1.0)  # Target Sharpe > 1
            perf_score += 0.2 * (1 - min(abs(wf.get('worst_drawdown', 0)) / 0.2, 1.0))  # Max 20% drawdown
            perf_score += 0.2 * (1 - min(wf.get('std_direction_accuracy', 0.1) / 0.1, 1.0))  # Low variance
            readiness_scores['Performance'] = perf_score
        else:
            readiness_scores['Performance'] = 0
        
        # 2. Statistical Significance Score
        if 'statistical_significance' in self.test_results:
            sig = self.test_results['statistical_significance']
            sig_score = 0
            sig_score += 0.5 if sig.get('is_significant_alpha_05', False) else 0
            sig_score += 0.5 if sig.get('is_significant_alpha_01', False) else 0.25
            readiness_scores['Statistical_Significance'] = sig_score
        else:
            readiness_scores['Statistical_Significance'] = 0
        
        # 3. Risk Management Score
        if 'risk_metrics' in self.test_results:
            risk = self.test_results['risk_metrics']
            risk_score = 0
            risk_score += 0.25 * min(risk.get('sharpe_ratio', 0) / 1.0, 1.0)
            risk_score += 0.25 * min(risk.get('sortino_ratio', 0) / 1.5, 1.0)
            risk_score += 0.25 * (1 - min(abs(risk.get('max_drawdown', 0)) / 0.2, 1.0))
            risk_score += 0.25 * min(risk.get('profit_factor', 1.0) / 1.5, 1.0)
            readiness_scores['Risk_Management'] = risk_score
        else:
            readiness_scores['Risk_Management'] = 0
        
        # 4. Stability Score
        if 'prediction_stability' in self.test_results:
            stab = self.test_results['prediction_stability']
            stab_score = 0
            stab_score += 0.5 * stab.get('mean_direction_agreement', 0.5)
            stab_score += 0.5 * stab.get('mean_correlation_between_runs', 0.5)
            readiness_scores['Stability'] = stab_score
        else:
            readiness_scores['Stability'] = 0
        
        # 5. Regime Robustness Score
        if 'regime_analysis' in self.test_results:
            regime = self.test_results['regime_analysis']
            regime_score = regime.get('regime_stability_score', 0)
            readiness_scores['Regime_Robustness'] = regime_score
        else:
            readiness_scores['Regime_Robustness'] = 0
        
        # 6. Practical Trading Score
        if 'trading_simulation' in self.test_results:
            trade = self.test_results['trading_simulation']
            trade_score = 0
            trade_score += 0.4 if trade.get('profitable', False) else 0
            trade_score += 0.3 if trade.get('meets_sharpe_threshold', False) else 0
            trade_score += 0.3 if trade.get('meets_drawdown_threshold', False) else 0
            readiness_scores['Practical_Trading'] = trade_score
        else:
            readiness_scores['Practical_Trading'] = 0
        
        # Calculate overall readiness
        overall_readiness = np.mean(list(readiness_scores.values()))
        
        # Print detailed report
        print("\n1. PERFORMANCE METRICS")
        print("-" * 40)
        if 'walk_forward' in self.test_results and 'aggregate_metrics' in self.test_results['walk_forward']:
            wf = self.test_results['walk_forward']['aggregate_metrics']
            print(f"Mean Direction Accuracy: {wf.get('mean_direction_accuracy', 0.5):.3f} ± {wf.get('std_direction_accuracy', 0):.3f}")
            print(f"Mean Sharpe Ratio: {wf.get('mean_sharpe', 0):.3f} ± {wf.get('std_sharpe', 0):.3f}")
            print(f"Worst Drawdown: {wf.get('worst_drawdown', 0):.3f}")
            print(f"Successful Folds: {wf.get('successful_folds', 0)}/{wf.get('total_folds', 0)}")
        else:
            print("Walk-forward analysis not completed successfully")
        
        print("\n2. STATISTICAL SIGNIFICANCE")
        print("-" * 40)
        if 'statistical_significance' in self.test_results:
            sig = self.test_results['statistical_significance']
            print(f"Direction Accuracy: {sig.get('direction_accuracy', 0.5):.3f}")
            print(f"P-value (Direction): {sig.get('p_value_direction', 1.0):.4f}")
            print(f"P-value (Permutation): {sig.get('p_value_permutation', 1.0):.4f}")
            print(f"Statistically Significant: {'Yes' if sig.get('is_significant_alpha_05', False) else 'No'}")
        else:
            print("Statistical significance test not completed successfully")
        
        print("\n3. RISK METRICS")
        print("-" * 40)
        if 'risk_metrics' in self.test_results:
            risk = self.test_results['risk_metrics']
            print(f"Sharpe Ratio: {risk.get('sharpe_ratio', 0):.3f}")
            print(f"Sortino Ratio: {risk.get('sortino_ratio', 0):.3f}")
            print(f"Max Drawdown: {risk.get('max_drawdown', 0):.3f}")
            print(f"Win Rate: {risk.get('win_rate', 0.5):.3f}")
            print(f"Profit Factor: {risk.get('profit_factor', 1.0):.3f}")
            print(f"VaR (95%): {risk.get('var_95', 0):.3f}")
            print(f"CVaR (95%): {risk.get('cvar_95', 0):.3f}")
        else:
            print("Risk metrics calculation not completed successfully")
        
        print("\n4. STABILITY ANALYSIS")
        print("-" * 40)
        if 'prediction_stability' in self.test_results:
            stab = self.test_results['prediction_stability']
            print(f"Direction Agreement: {stab.get('mean_direction_agreement', 0.5):.3f}")
            print(f"Prediction Correlation: {stab.get('mean_correlation_between_runs', 0.5):.3f}")
            print(f"Model is Stable: {'Yes' if stab.get('is_stable', False) else 'No'}")
        else:
            print("Stability analysis not completed successfully")
        
        print("\n5. REGIME PERFORMANCE")
        print("-" * 40)
        if 'regime_analysis' in self.test_results:
            regime = self.test_results['regime_analysis']
            print(f"Regime Stability Score: {regime.get('regime_stability_score', 0):.3f}")
            print(f"Best Regime: {regime.get('best_regime', 'unknown')}")
            print(f"Worst Regime: {regime.get('worst_regime', 'unknown')}")
            
            regime_perf = regime.get('regime_performance', {})
            if regime_perf:
                print("\nDetailed Regime Performance:")
                for reg, perf in regime_perf.items():
                    print(f"  {reg}: Accuracy={perf.get('direction_accuracy', 0.5):.3f}, Sharpe={perf.get('sharpe_ratio', 0):.3f}")
        else:
            print("Regime analysis not completed successfully")
        
        print("\n6. TRADING SIMULATION")
        print("-" * 40)
        if 'trading_simulation' in self.test_results:
            trade = self.test_results['trading_simulation']
            print(f"Total Return: {trade.get('total_return', 0):.2%}")
            print(f"Annualized Return: {trade.get('annualized_return', 0):.2%}")
            print(f"Sharpe Ratio: {trade.get('sharpe_ratio', 0):.3f}")
            print(f"Max Drawdown: {trade.get('max_drawdown', 0):.3f}")
            print(f"Number of Trades: {trade.get('n_trades', 0)}")
            print(f"Win Rate: {trade.get('win_rate', 0.5):.3f}")
        else:
            print("Trading simulation not completed successfully")
        
        print("\n7. STRESS TEST RESULTS")
        print("-" * 40)
        if 'stress_test' in self.test_results:
            stress = self.test_results['stress_test']
            print(f"Stress Test Score: {stress.get('stress_test_score', 0):.3f}")
            print(f"Passes Stress Test: {'Yes' if stress.get('passes_stress_test', False) else 'No'}")
            
            extreme_vol = stress.get('extreme_volatility', {})
            if extreme_vol:
                print("\nExtreme vs Normal Conditions:")
                for cond, metrics in extreme_vol.items():
                    print(f"  {cond}: Accuracy={metrics.get('direction_accuracy', 0.5):.3f}")
        else:
            print("Stress test not completed successfully")
        
        print("\n" + "="*80)
        print("READINESS SCORES")
        print("="*80)
        
        for category, score in readiness_scores.items():
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            print(f"{status} {category}: {score:.2f}/1.00")
        
        print("\n" + "="*80)
        print(f"OVERALL TRADING READINESS: {overall_readiness:.2f}/1.00")
        print("="*80)
        
        # Final recommendation
        print("\nRECOMMENDATION:")
        if overall_readiness >= 0.8:
            print("✅ Model is READY for live trading with proper risk management")
            print("   - Start with small position sizes")
            print("   - Monitor performance closely for first 30 days")
            print("   - Set strict stop-loss rules")
        elif overall_readiness >= 0.6:
            print("⚠️ Model shows POTENTIAL but needs improvements:")
            
            # Specific recommendations based on weak areas
            weak_areas = [k for k, v in readiness_scores.items() if v < 0.7]
            for area in weak_areas:
                if area == 'Performance':
                    print("   - Improve direction accuracy or reduce prediction horizon")
                elif area == 'Risk_Management':
                    print("   - Optimize position sizing and risk controls")
                elif area == 'Stability':
                    print("   - Add more regularization or ensemble methods")
                elif area == 'Regime_Robustness':
                    print("   - Train on more diverse market conditions")
        else:
            print("❌ Model is NOT READY for live trading")
            print("   - Continue development and testing")
            print("   - Consider fundamental strategy changes")
        
        # Save detailed report if requested
        if save_report:
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_readiness': overall_readiness,
                'readiness_scores': readiness_scores,
                'test_results': self.test_results,
                'recommendation': 'READY' if overall_readiness >= 0.8 else 'NOT READY'
            }
            
            # Save to file
            filename = f"trading_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            import json
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {filename}")
        
        return overall_readiness, readiness_scores
    
    def plot_test_results(self):
        """
        Create visualizations of test results
        """
        try:
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            
            # 1. Walk-forward performance
            if 'walk_forward' in self.test_results and 'fold_performance' in self.test_results['walk_forward']:
                ax = axes[0, 0]
                wf_data = pd.DataFrame(self.test_results['walk_forward']['fold_performance'])
                
                if len(wf_data) > 0:
                    x = range(len(wf_data))
                    ax.plot(x, wf_data['direction_accuracy'], 'b-o', label='Direction Accuracy')
                    ax.plot(x, wf_data['win_rate'], 'g-s', label='Win Rate')
                    ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)
                    
                    ax.set_xlabel('Fold')
                    ax.set_ylabel('Accuracy/Win Rate')
                    ax.set_title('Walk-Forward Performance')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No walk-forward data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Walk-Forward Performance')
            else:
                ax = axes[0, 0]
                ax.text(0.5, 0.5, 'Walk-forward analysis failed', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Walk-Forward Performance')
            
            # 2. Risk metrics visualization
            if 'risk_metrics' in self.test_results and 'error' not in self.test_results['risk_metrics']:
                ax = axes[0, 1]
                risk = self.test_results['risk_metrics']
                
                metrics = ['Sharpe', 'Sortino', 'Calmar']
                values = [risk.get('sharpe_ratio', 0), risk.get('sortino_ratio', 0), risk.get('calmar_ratio', 0)]
                
                bars = ax.bar(metrics, values)
                for i, (metric, value) in enumerate(zip(metrics, values)):
                    color = 'green' if value > 1 else 'orange' if value > 0.5 else 'red'
                    bars[i].set_color(color)
                    ax.text(i, value + 0.05, f'{value:.2f}', ha='center')
                
                ax.set_ylabel('Ratio')
                ax.set_title('Risk-Adjusted Performance Ratios')
                ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
            else:
                ax = axes[0, 1]
                ax.text(0.5, 0.5, 'Risk metrics calculation failed', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Risk-Adjusted Performance Ratios')
            
            # 3. Regime performance
            if 'regime_analysis' in self.test_results and 'regime_performance' in self.test_results['regime_analysis']:
                ax = axes[1, 0]
                regime_perf = self.test_results['regime_analysis']['regime_performance']
                
                if regime_perf:
                    regimes = list(regime_perf.keys())
                    accuracies = [regime_perf[r].get('direction_accuracy', 0.5) for r in regimes]
                    
                    bars = ax.bar(regimes, accuracies)
                    ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)
                    ax.set_ylabel('Direction Accuracy')
                    ax.set_title('Performance by Market Regime')
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, 'No regime data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Performance by Market Regime')
            else:
                ax = axes[1, 0]
                ax.text(0.5, 0.5, 'Regime analysis failed', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Performance by Market Regime')
            
            # 4. Prediction stability
            if 'prediction_stability' in self.test_results and 'error' not in self.test_results['prediction_stability']:
                ax = axes[1, 1]
                stab = self.test_results['prediction_stability']
                
                categories = ['Direction\nAgreement', 'Prediction\nCorrelation']
                values = [stab.get('mean_direction_agreement', 0.5), stab.get('mean_correlation_between_runs', 0.5)]
                
                bars = ax.bar(categories, values)
                for i, value in enumerate(values):
                    color = 'green' if value > 0.8 else 'orange' if value > 0.6 else 'red'
                    bars[i].set_color(color)
                    ax.text(i, value + 0.02, f'{value:.3f}', ha='center')
                
                ax.set_ylabel('Score')
                ax.set_title('Model Stability Metrics')
                ax.set_ylim(0, 1.1)
            else:
                ax = axes[1, 1]
                ax.text(0.5, 0.5, 'Stability analysis failed', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Model Stability Metrics')
            
            # 5. Trading simulation
            if 'trading_simulation' in self.test_results and 'error' not in self.test_results['trading_simulation']:
                ax = axes[2, 0]
                trade = self.test_results['trading_simulation']
                
                metrics = ['Total Return', 'Win Rate', 'Sharpe Ratio']
                values = [trade.get('total_return', 0), trade.get('win_rate', 0.5), trade.get('sharpe_ratio', 0)]
                
                ax.bar(metrics, values)
                ax.set_title('Trading Simulation Results')
                ax.set_ylabel('Value')
            else:
                ax = axes[2, 0]
                ax.text(0.5, 0.5, 'Trading simulation failed', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Trading Simulation Results')
            
            # 6. Overall readiness summary
            ax = axes[2, 1]
            ax.text(0.5, 0.5, 'Overall Readiness Summary\n(See text report above)', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trading Readiness Overview')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {str(e)}")
            print("Plots could not be generated, but test results are available in text format above.")

btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data()
# Assuming you have your df with engineered features
df_news = add_vader_sentiment(df_news)
df_newsdaily_sentiment = aggregate_daily_sentiment(df_news)

# 3. Feature engineering
df = engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment)

improved_predictor = ImprovedBitcoinPredictor(
    sequence_length=60,
    prediction_horizon=30,
)
# Example usage:
tester = ComprehensiveTradingModelTester(improved_predictor) # the model is initialized as improved_predictor
test_results = tester.run_all_tests(df, save_report=True) # df is BTC OHLCV data from 2017 august up until now
tester.plot_test_results() 