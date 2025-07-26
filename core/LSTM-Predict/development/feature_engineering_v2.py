import numpy as np
import pandas as pd

def engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment):
    df_daily = (
        btc_ohlcv
        #   .join(daily_oi, how='left')
        #   .join(daily_funding_rate, how='left')
        #   .join(df_newsdaily_sentiment, how='left')
    )
    df = df_daily.copy()
    
    # Fix index type consistency
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if len(df) < 100:
        raise ValueError("Need at least 100 data points for proper LSTM training")
    
    # ==================== ORIGINAL FEATURES ====================
    df['high_close_ratio'] = df['high'] / df['close']
    df['low_close_ratio'] = df['low'] / df['close']
    df['open_close_ratio'] = df['open'] / df['close']
    df['volume_avg_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Returns
    df['returns_1d'] = df['close'].pct_change()
    df['returns_3d'] = df['close'].pct_change(3)
    df['returns_7d'] = df['close'].pct_change(7)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20]:
        df[f'ma_{window}'] = df['close'].rolling(window).mean()
        df[f'price_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_normalized'] = df['macd'] / df['close']
    df['macd_signal_normalized'] = df['macd_signal'] / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_normalized'] = df['rsi'] / 100
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Basic volatility
    df['volatility_10'] = df['returns_1d'].rolling(10).std()
    df['volatility_20'] = df['returns_1d'].rolling(20).std()
    
    # Volume features
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_change'] = df['volume'].pct_change()
    
    # Momentum
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    
    # Time features
    df['day_of_week'] = df.index.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # ==================== NEW VOLATILITY-AWARE FEATURES ====================
    
    # Multi-timeframe volatility regimes
    df['volatility_30'] = df['returns_1d'].rolling(30).std()
    df['volatility_60'] = df['returns_1d'].rolling(60).std()
    
    # Volatility percentiles (regime detection)
    df['vol_percentile_30d'] = df['volatility_20'].rolling(252).rank(pct=True)
    
    # Volatility regime classification
    df['vol_regime_numeric'] = pd.cut(df['vol_percentile_30d'], 
                                     bins=[0, 0.33, 0.67, 1.0], 
                                     labels=[0, 1, 2]).astype(float)
    
    # Volatility breakouts
    df['vol_breakout'] = (df['volatility_10'] > df['volatility_10'].rolling(60).quantile(0.8)).astype(int)
    
    # Volatility-of-Volatility (second-order volatility)
    df['vol_of_vol'] = df['volatility_20'].rolling(20).std()
    df['vol_acceleration'] = df['volatility_20'].diff()
    df['vol_momentum'] = df['volatility_20'].pct_change(5)
    
    # Volatility clustering detection
    df['vol_cluster_strength'] = df['volatility_10'].rolling(10).std() / df['volatility_10'].rolling(10).mean()
    
    # GARCH-style features
    df['vol_ewm_fast'] = df['returns_1d'].ewm(alpha=0.1).std()
    df['vol_ewm_slow'] = df['returns_1d'].ewm(alpha=0.05).std()
    df['vol_ratio_fast_slow'] = df['vol_ewm_fast'] / df['vol_ewm_slow']
    
    # Asymmetric volatility (volatility skew)
    positive_returns = df['returns_1d'].where(df['returns_1d'] > 0, 0)
    negative_returns = df['returns_1d'].where(df['returns_1d'] < 0, 0)
    df['upside_volatility'] = positive_returns.rolling(20).std()
    df['downside_volatility'] = negative_returns.rolling(20).std()
    df['volatility_skew'] = df['downside_volatility'] / df['upside_volatility']
    
    # Extreme volatility indicators
    df['vol_spike'] = (df['volatility_10'] > df['volatility_10'].rolling(252).quantile(0.95)).astype(int)
    
    # Rolling maximum volatility (stress indicator)
    df['max_vol_30d'] = df['volatility_10'].rolling(30).max()
    df['vol_stress_ratio'] = df['volatility_10'] / df['max_vol_30d']
    
    # Volatility mean reversion indicator
    df['vol_mean_reversion'] = (df['volatility_20'] - df['volatility_20'].rolling(60).mean()) / df['volatility_20'].rolling(60).std()
    
    # Market microstructure volatility
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_14'] = df['true_range'].rolling(14).mean()
    df['atr_normalized'] = df['atr_14'] / df['close']
    
    # Volatility efficiency
    df['price_efficiency'] = abs(df['close'] - df['open']) / df['true_range']
    
    # Volatility trend features
    df['vol_trend_5d'] = df['volatility_20'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0, raw=False)
    df['vol_trend_strength'] = df['volatility_20'].rolling(10).apply(
        lambda x: abs(np.corrcoef(range(len(x)), x)[0,1]) if len(x) == 10 else 0, raw=False
    )
    
    # Volatility relative to historical levels
    df['vol_zscore'] = (df['volatility_20'] - df['volatility_20'].rolling(252).mean()) / df['volatility_20'].rolling(252).std()
    
    # Extreme condition detection (for stress testing)
    df['extreme_condition'] = (
        (df['vol_percentile_30d'] > 0.8) |  # High volatility regime
        (df['vol_spike'] == 1) |            # Volatility spikes
        (abs(df['vol_zscore']) > 2)         # Extreme volatility z-score
    )
    
    # Market regime classification (for regime transition testing)
    def classify_regime(row):
        if pd.isna(row['vol_percentile_30d']) or pd.isna(row['returns_7d']):
            return 'unknown'
        
        if row['vol_percentile_30d'] > 0.7 and row['returns_7d'] > 0.05:
            return 'bull_volatile'
        elif row['vol_percentile_30d'] > 0.7 and row['returns_7d'] < -0.05:
            return 'bear_volatile'
        elif row['vol_percentile_30d'] < 0.3 and abs(row['returns_7d']) < 0.02:
            return 'sideways_stable'
        elif row['returns_7d'] > 0.02:
            return 'bull_stable'
        elif row['returns_7d'] < -0.02:
            return 'bear_stable'
        else:
            return 'sideways'
    
    df['market_regime'] = df.apply(classify_regime, axis=1)
    
    # Volatility-adjusted technical indicators
    df['rsi_vol_adjusted'] = df['rsi'] * (1 + df['vol_zscore'] * 0.1)  # Adjust RSI for volatility regime
    df['bb_position_vol_adjusted'] = df['bb_position'] * df['vol_ratio_fast_slow']  # Vol-aware BB position
    df['macd_vol_normalized'] = df['macd_normalized'] / (1 + df['volatility_20'])  # Vol-normalized MACD
    
    # Volatility momentum indicators
    df['vol_rsi'] = df['volatility_20'].rolling(14).apply(
        lambda x: 100 - (100 / (1 + (x.diff().where(x.diff() > 0, 0).mean() / 
                                    (-x.diff().where(x.diff() < 0, 0).mean())))) if len(x) == 14 else 50, 
        raw=False
    )
    
    # Cross-timeframe volatility ratios
    df['vol_ratio_10_30'] = df['volatility_10'] / df['volatility_30']
    df['vol_ratio_20_60'] = df['volatility_20'] / df['volatility_60']
    
    # Volatility persistence
    df['vol_persistence'] = df['volatility_20'].rolling(5).apply(
        lambda x: (x > x.mean()).sum() / len(x), raw=False
    )
    
    # ==================== TARGET VARIABLES ====================
    df['next_close'] = df['close'].shift(-1)
    df['target_return'] = (df['next_close'] - df['close']) / df['close']
    df['target_direction'] = (df['target_return'] > 0).astype(int)
    
    # ==================== CLEANUP ====================
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Convert object columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'market_regime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna()
    
    if len(df) < 50:
        raise ValueError("Not enough clean data after preprocessing")
    
    return df