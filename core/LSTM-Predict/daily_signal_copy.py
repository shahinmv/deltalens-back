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
from typing import Dict, Optional, Tuple, List
import requests
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from main_ml import ImprovedBitcoinPredictor

# Import the required modules using relative imports
base_path = os.path.dirname(os.path.abspath(__file__))
dev_path = os.path.join(base_path, 'development')
if dev_path not in sys.path:
    sys.path.insert(0, dev_path)
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from data_loader import load_all_data
from sentiment import add_vader_sentiment, aggregate_daily_sentiment
from feature_engineering import engineer_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_trading.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # 'bull', 'bear', 'sideways', 'high_vol', 'low_vol'
    confidence: float
    volatility_regime: str  # 'low', 'medium', 'high', 'extreme'
    trend_strength: float
    momentum_score: float
    fear_greed_level: float

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_1d: float  # 1-day Value at Risk
    cvar_1d: float  # 1-day Conditional VaR
    max_drawdown: float
    volatility: float
    beta_to_market: float
    sharpe_ratio: float
    calmar_ratio: float
    skewness: float
    kurtosis: float
    correlation_btc: float

@dataclass
class AdvancedTradeSignal:
    """Enhanced trade signal with multiple strategies"""
    symbol: str
    primary_signal: str  # 'LONG', 'SHORT', 'HOLD'
    signal_strength: float  # 0-1 confidence
    strategy_components: Dict[str, float]  # Individual strategy contributions
    
    # Position sizing
    base_position_size: float
    volatility_adjusted_size: float
    risk_adjusted_size: float
    final_position_size: float
    
    # Execution parameters
    entry_price: float
    target_prices: List[float]  # Multiple targets
    stop_loss_price: float
    trailing_stop: bool
    
    # Risk management
    max_position_risk: float
    portfolio_heat: float
    correlation_adjustment: float
    
    # Market timing
    optimal_execution_time: str
    market_impact_cost: float
    funding_rate_impact: float
    
    # Multi-timeframe
    short_term_signal: str  # 1h-4h signals
    medium_term_signal: str  # daily signals
    long_term_signal: str  # weekly signals
    
    timestamp: datetime
    expires_at: datetime
    regime: MarketRegime

class AdvancedRiskManager:
    """Institutional-grade risk management system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_portfolio_var = config.get('max_portfolio_var', 0.02)  # 2% daily VaR
        self.max_concentration = config.get('max_concentration', 0.3)   # 30% max in one position
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.volatility_lookback = config.get('volatility_lookback', 20)
        
    def calculate_var_cvar(self, returns: np.ndarray, confidence: float = 0.05) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional VaR"""
        if len(returns) < 10:
            return 0.05, 0.07  # Conservative defaults
            
        # Handle outliers
        returns_clean = self._winsorize_returns(returns)
        
        # Parametric VaR (assumes normal distribution)
        mean_return = np.mean(returns_clean)
        std_return = np.std(returns_clean)
        var_parametric = mean_return - stats.norm.ppf(1 - confidence) * std_return
        
        # Historical VaR
        var_historical = np.percentile(returns_clean, confidence * 100)
        
        # Use the more conservative estimate
        var = min(var_parametric, var_historical)
        
        # Conditional VaR (Expected Shortfall)
        cvar = np.mean(returns_clean[returns_clean <= var])
        
        return abs(var), abs(cvar)
    
    def _winsorize_returns(self, returns: np.ndarray, limits: Tuple[float, float] = (0.01, 0.01)) -> np.ndarray:
        """Winsorize returns to handle outliers"""
        return stats.mstats.winsorize(returns, limits=limits)
    
    def calculate_position_size_kelly_refined(self, expected_return: float, variance: float, 
                                            win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Refined Kelly criterion with additional constraints"""
        if variance <= 0 or avg_loss >= 0:
            return 0.0
            
        # Traditional Kelly
        kelly_simple = expected_return / variance
        
        # Kelly with win/loss ratio
        if avg_loss != 0:
            win_loss_ratio = abs(avg_win / avg_loss)
            kelly_refined = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        else:
            kelly_refined = kelly_simple
            
        # Apply conservative constraints
        kelly_constrained = min(kelly_refined, 0.25)  # Never more than 25%
        kelly_constrained = max(kelly_constrained, 0.0)  # Never negative
        
        # Apply additional risk scaling
        if expected_return < 0:
            return 0.0
            
        return kelly_constrained * 0.5  # 50% Kelly for additional safety

class RegimeDetector:
    """Advanced market regime detection system"""
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        self.lookback_periods = lookback_periods or {
            'short': 10, 'medium': 30, 'long': 60
        }
        
    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Comprehensive regime detection"""
        
        # Calculate returns
        returns = data['close'].pct_change().dropna()
        
        # Trend analysis
        trend_strength = self._calculate_trend_strength(data)
        
        # Volatility regime
        volatility_regime, current_vol = self._detect_volatility_regime(returns)
        
        # Momentum analysis
        momentum_score = self._calculate_momentum_score(data)
        
        # Market structure analysis
        market_structure = self._analyze_market_structure(data)
        
        # Fear & Greed estimation (simplified)
        fear_greed_level = self._estimate_fear_greed(data, returns)
        
        # Combine indicators to determine primary regime
        regime_type = self._classify_primary_regime(
            trend_strength, momentum_score, current_vol, market_structure
        )
        
        # Calculate confidence based on signal alignment
        confidence = self._calculate_regime_confidence(
            trend_strength, momentum_score, volatility_regime
        )
        
        return MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            volatility_regime=volatility_regime,
            trend_strength=trend_strength,
            momentum_score=momentum_score,
            fear_greed_level=fear_greed_level
        )
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        close = data['close']
        
        # Moving averages
        ma_short = close.rolling(10).mean()
        ma_medium = close.rolling(20).mean()
        ma_long = close.rolling(50).mean()
        
        # Trend score based on MA alignment
        current_price = close.iloc[-1]
        trend_score = 0
        
        if current_price > ma_short.iloc[-1] > ma_medium.iloc[-1] > ma_long.iloc[-1]:
            trend_score = 1.0  # Strong uptrend
        elif current_price < ma_short.iloc[-1] < ma_medium.iloc[-1] < ma_long.iloc[-1]:
            trend_score = -1.0  # Strong downtrend
        else:
            # Partial alignment
            alignments = [
                current_price > ma_short.iloc[-1],
                ma_short.iloc[-1] > ma_medium.iloc[-1],
                ma_medium.iloc[-1] > ma_long.iloc[-1]
            ]
            trend_score = (sum(alignments) - 1.5) / 1.5  # Normalize to [-1, 1]
        
        return trend_score
    
    def _detect_volatility_regime(self, returns: pd.Series) -> Tuple[str, float]:
        """Detect volatility regime using GARCH-style analysis"""
        vol_short = returns.rolling(10).std() * np.sqrt(365)
        vol_medium = returns.rolling(30).std() * np.sqrt(365)
        vol_long = returns.rolling(60).std() * np.sqrt(365)
        
        current_vol = vol_short.iloc[-1]
        historical_vol = vol_long.mean()
        
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
        
        if vol_ratio < 0.7:
            regime = 'low'
        elif vol_ratio < 1.3:
            regime = 'medium'
        elif vol_ratio < 2.0:
            regime = 'high'
        else:
            regime = 'extreme'
            
        return regime, current_vol
    
    def _calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate multi-timeframe momentum score"""
        close = data['close']
        
        # Different timeframe returns
        ret_1d = close.pct_change(1).iloc[-1]
        ret_3d = close.pct_change(3).iloc[-1]
        ret_7d = close.pct_change(7).iloc[-1]
        ret_14d = close.pct_change(14).iloc[-1]
        
        # Weighted momentum score
        weights = [0.1, 0.2, 0.3, 0.4]  # More weight on longer-term momentum
        momentum_score = (
            weights[0] * ret_1d + 
            weights[1] * ret_3d + 
            weights[2] * ret_7d + 
            weights[3] * ret_14d
        )
        
        # Normalize to [-1, 1] range
        return np.tanh(momentum_score * 10)
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> float:
        """Analyze market microstructure indicators"""
        # Volume analysis
        volume = data['volume']
        price = data['close']
        
        # Volume-price relationship
        price_change = price.pct_change()
        volume_change = volume.pct_change()
        
        # Correlation between volume and absolute price change
        vol_price_corr = price_change.abs().rolling(20).corr(volume_change).iloc[-1]
        
        # Price efficiency (mean reversion vs trend)
        returns = price.pct_change()
        autocorr = returns.rolling(20).apply(lambda x: x.autocorr(lag=1)).iloc[-1]
        
        # Combine indicators
        structure_score = (vol_price_corr * 0.6 + (1 - abs(autocorr)) * 0.4)
        
        return structure_score if not np.isnan(structure_score) else 0.5
    
    def _estimate_fear_greed(self, data: pd.DataFrame, returns: pd.Series) -> float:
        """Estimate fear & greed level"""
        # Volatility component
        current_vol = returns.rolling(10).std().iloc[-1]
        historical_vol = returns.rolling(60).std().mean()
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
        
        # Price momentum component
        price_change_7d = data['close'].pct_change(7).iloc[-1]
        
        # Volume component (higher volume in up moves = greed)
        volume_weighted_return = (returns * data['volume'].pct_change()).rolling(10).mean().iloc[-1]
        
        # Combine components (0 = extreme fear, 1 = extreme greed)
        fear_greed = 0.5 + (price_change_7d * 0.4 + volume_weighted_return * 0.3 - (vol_ratio - 1) * 0.3)
        
        return np.clip(fear_greed, 0, 1)
    
    def _classify_primary_regime(self, trend_strength: float, momentum_score: float, 
                               volatility: float, market_structure: float) -> str:
        """Classify the primary market regime"""
        
        if trend_strength > 0.3 and momentum_score > 0.2:
            return 'bull'
        elif trend_strength < -0.3 and momentum_score < -0.2:
            return 'bear'
        elif volatility > 0.8:  # High volatility
            return 'high_vol'
        elif abs(trend_strength) < 0.2 and abs(momentum_score) < 0.1:
            return 'sideways'
        else:
            return 'mixed'
    
    def _calculate_regime_confidence(self, trend_strength: float, momentum_score: float, 
                                   volatility_regime: str) -> float:
        """Calculate confidence in regime classification"""
        
        # Alignment between trend and momentum
        trend_momentum_alignment = 1 - abs(trend_strength - momentum_score) / 2
        
        # Volatility penalty (high vol = lower confidence)
        vol_penalty = {'low': 0.0, 'medium': 0.1, 'high': 0.3, 'extreme': 0.5}
        
        confidence = trend_momentum_alignment - vol_penalty.get(volatility_regime, 0.2)
        
        return np.clip(confidence, 0.1, 0.95)

class StrategyEnsemble:
    """Ensemble of different trading strategies"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategies = {
            'ml_momentum': {'weight': 0.3, 'enabled': True},
            'mean_reversion': {'weight': 0.2, 'enabled': True},
            'breakout': {'weight': 0.2, 'enabled': True},
            'volatility_trading': {'weight': 0.15, 'enabled': True},
            'regime_following': {'weight': 0.15, 'enabled': True}
        }
        
    def generate_ensemble_signal(self, data: pd.DataFrame, ml_prediction: float, 
                               regime: MarketRegime) -> Dict[str, float]:
        """Generate signals from multiple strategies"""
        
        signals = {}
        
        # ML Momentum Strategy
        if self.strategies['ml_momentum']['enabled']:
            signals['ml_momentum'] = self._ml_momentum_signal(ml_prediction, regime)
        
        # Mean Reversion Strategy
        if self.strategies['mean_reversion']['enabled']:
            signals['mean_reversion'] = self._mean_reversion_signal(data, regime)
            
        # Breakout Strategy
        if self.strategies['breakout']['enabled']:
            signals['breakout'] = self._breakout_signal(data, regime)
            
        # Volatility Trading Strategy
        if self.strategies['volatility_trading']['enabled']:
            signals['volatility_trading'] = self._volatility_trading_signal(data, regime)
            
        # Regime Following Strategy
        if self.strategies['regime_following']['enabled']:
            signals['regime_following'] = self._regime_following_signal(regime)
        
        return signals
    
    def _ml_momentum_signal(self, ml_prediction: float, regime: MarketRegime) -> float:
        """ML-based momentum signal with regime adjustment"""
        base_signal = np.tanh(ml_prediction * 5)  # Scale and bound
        
        # Adjust based on regime
        if regime.regime_type == 'bull':
            regime_multiplier = 1.2
        elif regime.regime_type == 'bear':
            regime_multiplier = 1.1  # Still useful in bear markets
        elif regime.regime_type == 'high_vol':
            regime_multiplier = 0.7  # Reduce during high volatility
        else:
            regime_multiplier = 1.0
            
        return base_signal * regime_multiplier * regime.confidence
    
    def _mean_reversion_signal(self, data: pd.DataFrame, regime: MarketRegime) -> float:
        """Mean reversion signal"""
        close = data['close']
        
        # Bollinger Bands-based reversion
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        
        current_price = close.iloc[-1]
        upper_band = ma_20.iloc[-1] + 2 * std_20.iloc[-1]
        lower_band = ma_20.iloc[-1] - 2 * std_20.iloc[-1]
        
        # Mean reversion signal
        if current_price > upper_band:
            signal = -1  # Sell signal
        elif current_price < lower_band:
            signal = 1   # Buy signal
        else:
            # Gradual signal based on position within bands
            band_position = (current_price - ma_20.iloc[-1]) / std_20.iloc[-1]
            signal = -np.tanh(band_position)  # Negative means sell when high
        
        # Reduce mean reversion in strong trending markets
        if abs(regime.trend_strength) > 0.5:
            signal *= 0.3
            
        return signal
    
    def _breakout_signal(self, data: pd.DataFrame, regime: MarketRegime) -> float:
        """Breakout strategy signal"""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Donchian Channel breakout
        period = 20
        high_channel = high.rolling(period).max()
        low_channel = low.rolling(period).min()
        
        current_price = close.iloc[-1]
        
        # Volume confirmation
        avg_volume = volume.rolling(period).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        volume_multiplier = min(current_volume / avg_volume, 2.0) if avg_volume > 0 else 1.0
        
        if current_price > high_channel.iloc[-2]:  # Upside breakout
            signal = 1.0 * volume_multiplier
        elif current_price < low_channel.iloc[-2]:  # Downside breakout
            signal = -1.0 * volume_multiplier
        else:
            # Near breakout levels
            upper_proximity = (current_price - high_channel.iloc[-2]) / high_channel.iloc[-2]
            lower_proximity = (low_channel.iloc[-2] - current_price) / low_channel.iloc[-2]
            
            if upper_proximity > -0.02:  # Within 2% of upper breakout
                signal = 0.5 * volume_multiplier
            elif lower_proximity > -0.02:  # Within 2% of lower breakout
                signal = -0.5 * volume_multiplier
            else:
                signal = 0.0
        
        # Enhance signal in trending regimes
        if regime.regime_type in ['bull', 'bear']:
            signal *= 1.3
            
        return np.clip(signal, -1, 1)
    
    def _volatility_trading_signal(self, data: pd.DataFrame, regime: MarketRegime) -> float:
        """Volatility trading signal (buy low vol, sell high vol)"""
        returns = data['close'].pct_change()
        
        # Realized volatility
        current_vol = returns.rolling(10).std() * np.sqrt(365)
        historical_vol = returns.rolling(60).std() * np.sqrt(365)
        
        vol_ratio = current_vol.iloc[-1] / historical_vol.mean() if historical_vol.mean() > 0 else 1
        
        # Volatility mean reversion signal
        if vol_ratio < 0.7:  # Low volatility
            signal = 0.5  # Expect volatility to increase
        elif vol_ratio > 1.5:  # High volatility
            signal = -0.3  # Expect volatility to decrease
        else:
            signal = 0.0
            
        # Adjust based on regime volatility
        if regime.volatility_regime == 'extreme':
            signal *= 1.5  # Stronger signal in extreme vol
            
        return signal
    
    def _regime_following_signal(self, regime: MarketRegime) -> float:
        """Pure regime-following signal"""
        base_signal = regime.trend_strength * regime.momentum_score
        
        # Confidence weighting
        confidence_weighted = base_signal * regime.confidence
        
        # Fear/greed adjustment
        if regime.fear_greed_level > 0.8:  # Extreme greed
            confidence_weighted *= 0.7  # Reduce bullish signals
        elif regime.fear_greed_level < 0.2:  # Extreme fear
            if confidence_weighted < 0:
                confidence_weighted *= 0.7  # Reduce bearish signals
        
        return confidence_weighted
    
    def combine_signals(self, individual_signals: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Combine individual strategy signals into ensemble signal"""
        
        total_weight = 0
        weighted_signal = 0
        
        active_signals = {}
        
        for strategy, signal in individual_signals.items():
            if strategy in self.strategies and self.strategies[strategy]['enabled']:
                weight = self.strategies[strategy]['weight']
                weighted_signal += signal * weight
                total_weight += weight
                active_signals[strategy] = signal
        
        # Normalize
        if total_weight > 0:
            ensemble_signal = weighted_signal / total_weight
        else:
            ensemble_signal = 0.0
            
        return ensemble_signal, active_signals

class AdvancedExecutionEngine:
    """Advanced execution engine with market microstructure awareness"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.slippage_model = config.get('slippage_model', 'linear')
        self.market_impact_factor = config.get('market_impact_factor', 0.1)
        
    def calculate_optimal_execution(self, signal: AdvancedTradeSignal, 
                                  market_data: pd.DataFrame) -> Dict:
        """Calculate optimal execution parameters"""
        
        # Market impact estimation
        market_impact = self._estimate_market_impact(
            signal.final_position_size, market_data
        )
        
        # Timing optimization
        optimal_timing = self._optimize_execution_timing(market_data)
        
        # Order splitting strategy
        order_schedule = self._create_order_schedule(
            signal.final_position_size, market_impact
        )
        
        return {
            'market_impact_bps': market_impact * 10000,
            'optimal_timing': optimal_timing,
            'order_schedule': order_schedule,
            'expected_slippage': market_impact * 0.5  # Conservative estimate
        }
    
    def _estimate_market_impact(self, position_size: float, data: pd.DataFrame) -> float:
        """Estimate market impact based on position size and liquidity"""
        
        # Simple square-root model: impact ∝ sqrt(size/volume)
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        notional_size = position_size * data['close'].iloc[-1]
        
        if avg_volume > 0:
            volume_ratio = notional_size / avg_volume
            impact = self.market_impact_factor * np.sqrt(volume_ratio)
        else:
            impact = 0.01  # 1% default impact
            
        return min(impact, 0.05)  # Cap at 5%
    
    def _optimize_execution_timing(self, data: pd.DataFrame) -> str:
        """Determine optimal execution timing"""
        
        # Simple heuristic based on volatility patterns
        returns = data['close'].pct_change()
        hourly_vol = returns.rolling(24).std()  # Assuming hourly data
        
        current_vol = hourly_vol.iloc[-1]
        avg_vol = hourly_vol.mean()
        
        if current_vol < avg_vol * 0.8:
            return 'immediate'  # Low volatility, execute now
        elif current_vol > avg_vol * 1.5:
            return 'wait_for_calm'  # High volatility, wait
        else:
            return 'VWAP'  # Normal volatility, use VWAP
    
    def _create_order_schedule(self, total_size: float, impact: float) -> List[Dict]:
        """Create order schedule to minimize market impact"""
        
        if total_size <= 0.01:  # Small order
            return [{'size': total_size, 'timing': 'immediate'}]
        
        # Split large orders
        if impact > 0.02:  # High impact expected
            num_splits = min(int(total_size / 0.01), 10)  # Max 10 splits
            split_size = total_size / num_splits
            
            schedule = []
            for i in range(num_splits):
                schedule.append({
                    'size': split_size,
                    'timing': f'interval_{i+1}',
                    'delay_minutes': i * 15  # 15-minute intervals
                })
            return schedule
        else:
            return [{'size': total_size, 'timing': 'immediate'}]

class InstitutionalTradingSystem:
    """Advanced institutional-grade trading system"""
    
    def __init__(self, config_path: str = "institutional_config.json"):
        
        # Load enhanced configuration
        self.config = self._load_enhanced_config(config_path)
        
        # Initialize components
        self.risk_manager = AdvancedRiskManager(self.config['risk_management'])
        self.regime_detector = RegimeDetector(self.config['regime_detection'])
        self.strategy_ensemble = StrategyEnsemble(self.config['strategies'])
        self.execution_engine = AdvancedExecutionEngine(self.config['execution'])
        
        # ML predictor
        self.predictor = None
        
        # Performance tracking
        self.performance_metrics = {}
        self.portfolio_state = {
            'capital': self.config['trading']['initial_capital'],
            'positions': {},
            'daily_returns': [],
            'drawdown_periods': []
        }
        
        # Risk limits
        self.risk_limits = self.config['risk_management']
        
    def _load_enhanced_config(self, config_path: str) -> Dict:
        """Load comprehensive configuration"""
        default_config = {
            'trading': {
                'initial_capital': 100000,
                'max_leverage': 1.0,
                'rebalance_frequency': 'daily',
                'position_limits': {
                    'max_single_position': 0.2,  # 20% max
                    'max_sector_exposure': 0.5,   # 50% max
                    'max_correlation_exposure': 0.4
                }
            },
            'risk_management': {
                'max_portfolio_var': 0.02,
                'max_drawdown': 0.15,
                'volatility_target': 0.15,
                'correlation_threshold': 0.7,
                'stop_loss_method': 'dynamic',
                'position_sizing_method': 'kelly_refined'
            },
            'regime_detection': {
                'lookback_periods': {'short': 10, 'medium': 30, 'long': 60},
                'regime_change_threshold': 0.3,
                'min_regime_confidence': 0.6
            },
            'strategies': {
                'ensemble_weights': {
                    'ml_momentum': 0.3,
                    'mean_reversion': 0.2,
                    'breakout': 0.2,
                    'volatility_trading': 0.15,
                    'regime_following': 0.15
                },
                'strategy_allocation_method': 'risk_parity'
            },
            'execution': {
                'slippage_model': 'square_root',
                'market_impact_factor': 0.1,
                'min_order_size': 0.001,
                'max_order_size': 0.05
            },
            'data_sources': {
                'primary': 'binance',
                'alternatives': ['coinbase', 'kraken'],
                'macro_data': True,
                'options_data': False
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Deep merge configurations
                self._deep_merge_config(default_config, user_config)
        except Exception as e:
            logging.warning(f"Could not load config file: {e}. Using defaults.")
            
        return default_config
    
    def _deep_merge_config(self, base: Dict, overlay: Dict) -> None:
        """Deep merge configuration dictionaries"""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_config(base[key], value)
            else:
                base[key] = value
    
    def generate_advanced_signal(self, data: pd.DataFrame) -> Optional[AdvancedTradeSignal]:
        """Generate advanced trading signal with institutional features"""
        
        try:
            # 1. Detect market regime
            regime = self.regime_detector.detect_regime(data)
            logging.info(f"Market regime: {regime.regime_type} (confidence: {regime.confidence:.2f})")
            
            # 2. Get ML prediction
            if self.predictor is None:
                logging.warning("ML predictor not trained")
                return None
                
            ml_result = self.predictor.predict_next_30d(data)
            ml_prediction = ml_result['predicted_return']
            
            # 3. Generate ensemble signals
            individual_signals = self.strategy_ensemble.generate_ensemble_signal(
                data, ml_prediction, regime
            )
            
            ensemble_signal, active_signals = self.strategy_ensemble.combine_signals(
                individual_signals
            )
            
            logging.info(f"Ensemble signal: {ensemble_signal:.3f}")
            logging.info(f"Active signals: {active_signals}")
            
            # 4. Risk-adjusted position sizing
            returns = data['close'].pct_change().dropna()
            
            # Calculate various position sizing methods
            var_1d, cvar_1d = self.risk_manager.calculate_var_cvar(returns)
            
            # Kelly criterion (refined)
            kelly_size = self.risk_manager.calculate_position_size_kelly_refined(
                ml_prediction, 
                returns.var(),
                0.55,  # Estimated win rate
                0.03,  # Average win
                -0.02  # Average loss
            )
            
            # Volatility-adjusted sizing
            current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(365)
            target_vol = self.config['risk_management']['volatility_target']
            vol_adjustment = target_vol / current_vol if current_vol > 0 else 1.0
            vol_adjusted_size = abs(ensemble_signal) * vol_adjustment
            
            # Risk-adjusted sizing (based on VaR)
            risk_adjusted_size = min(
                vol_adjusted_size,
                self.risk_limits['max_portfolio_var'] / var_1d if var_1d > 0 else 0.1
            )
            
            # Final position size
            final_size = min(
                kelly_size,
                risk_adjusted_size,
                self.config['trading']['position_limits']['max_single_position']
            )
            
            # Apply signal direction
            if ensemble_signal > 0.05:  # Bullish threshold
                signal_type = 'LONG'
                final_position_size = final_size
            elif ensemble_signal < -0.05:  # Bearish threshold  
                signal_type = 'SHORT'
                final_position_size = final_size
            else:
                signal_type = 'HOLD'
                final_position_size = 0.0
            
            # 5. Calculate execution parameters
            current_price = data['close'].iloc[-1]
            
            # Multiple target prices (25%, 50%, 75% of predicted move)
            if signal_type == 'LONG':
                target_prices = [
                    current_price * (1 + ml_prediction * 0.25),
                    current_price * (1 + ml_prediction * 0.50),
                    current_price * (1 + ml_prediction * 0.75)
                ]
                stop_loss_price = current_price * (1 - var_1d * 2)  # 2x VaR stop
            elif signal_type == 'SHORT':
                target_prices = [
                    current_price * (1 - abs(ml_prediction) * 0.25),
                    current_price * (1 - abs(ml_prediction) * 0.50),
                    current_price * (1 - abs(ml_prediction) * 0.75)
                ]
                stop_loss_price = current_price * (1 + var_1d * 2)  # 2x VaR stop
            else:
                target_prices = [current_price]
                stop_loss_price = current_price
            
            # 6. Create advanced signal first
            signal = AdvancedTradeSignal(
                symbol="BTCUSDT",
                primary_signal=signal_type,
                signal_strength=abs(ensemble_signal),
                strategy_components=active_signals,
                
                base_position_size=abs(ensemble_signal),
                volatility_adjusted_size=vol_adjusted_size,
                risk_adjusted_size=risk_adjusted_size,
                final_position_size=final_position_size,
                
                entry_price=current_price,
                target_prices=target_prices,
                stop_loss_price=stop_loss_price,
                trailing_stop=(regime.volatility_regime in ['low', 'medium']),
                
                max_position_risk=var_1d,
                portfolio_heat=self._calculate_portfolio_heat(),
                correlation_adjustment=1.0,  # Will implement if multi-asset
                
                optimal_execution_time='immediate',  # Default, will be updated below
                market_impact_cost=0.0,  # Default, will be updated below
                funding_rate_impact=0.0,  # Will implement if futures
                
                short_term_signal=self._get_short_term_signal(data),
                medium_term_signal=signal_type,
                long_term_signal=self._get_long_term_signal(data, regime),
                
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=24),
                regime=regime
            )
            
            # 7. Market microstructure analysis (after signal is created)
            execution_params = self.execution_engine.calculate_optimal_execution(
                signal, data
            )
            
            # Update execution parameters
            signal.optimal_execution_time = execution_params['optimal_timing']
            signal.market_impact_cost = execution_params['market_impact_bps'] / 10000
            
            logging.info(f"Generated advanced signal: {signal_type}")
            logging.info(f"Final position size: {final_position_size:.3f}")
            logging.info(f"Expected market impact: {execution_params['market_impact_bps']:.1f} bps")
            
            return signal
            
        except Exception as e:
            logging.error(f"Error generating advanced signal: {e}")
            return None
    
    def _calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (total risk exposure)"""
        # For single asset, this is simplified
        return min(sum(abs(pos) for pos in self.portfolio_state['positions'].values()), 1.0)
    
    def _get_short_term_signal(self, data: pd.DataFrame) -> str:
        """Get short-term (intraday) signal"""
        # Simplified RSI-based short-term signal
        close = data['close']
        rsi = self._calculate_rsi(close, 14)
        
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        if current_rsi > 70:
            return 'SHORT'
        elif current_rsi < 30:
            return 'LONG'
        else:
            return 'HOLD'
    
    def _get_long_term_signal(self, data: pd.DataFrame, regime: MarketRegime) -> str:
        """Get long-term signal based on regime and trends"""
        if regime.trend_strength > 0.3:
            return 'LONG'
        elif regime.trend_strength < -0.3:
            return 'SHORT'
        else:
            return 'HOLD'
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def train_enhanced_model(self, data: pd.DataFrame) -> bool:
        """Train ML model with enhanced features"""
        try:
            logging.info("Training enhanced ML model...")
            
            if len(data) < 500:  # Need more data for advanced model
                logging.error(f"Insufficient data for enhanced training: {len(data)} records")
                return False
            
            # Initialize predictor if needed
            if self.predictor is None:
                self.predictor = ImprovedBitcoinPredictor(
                    sequence_length=60, 
                    prediction_horizon=30
                )
            
            # Train with cross-validation and early stopping
            X_val, y_val, regime_seq = self.predictor.train_ensemble(
                data, 
                validation_split=0.2, 
                epochs=150,  # More epochs for better training
                batch_size=64   # Larger batch size
            )
            
            if X_val is None:
                logging.error("Enhanced model training failed")
                return False
            
            # Enhanced evaluation
            evaluation = self.predictor.evaluate_ensemble(X_val, y_val, regime_seq)
            
            # Store performance metrics
            self.performance_metrics['ml_model'] = evaluation
            
            logging.info(f"Enhanced model training completed:")
            logging.info(f"  MAE: {evaluation.get('mae', 0):.4f}")
            logging.info(f"  Direction Accuracy: {evaluation.get('direction_accuracy', 0):.3f}")
            logging.info(f"  Sharpe Ratio: {evaluation.get('sharpe_ratio', 0):.3f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error in enhanced model training: {e}")
            return False
    
    def run_comprehensive_backtest(self, data: pd.DataFrame, 
                                 initial_capital: float = 100000) -> Dict:
        """Run comprehensive institutional-grade backtest"""
        
        try:
            logging.info("Running comprehensive backtest...")
            
            # Ensure model is trained
            if not self.train_enhanced_model(data):
                return {'error': 'Model training failed'}
            
            # Prepare data
            df_proc = self.predictor.engineer_30day_target(data)
            features, _ = self.predictor.prepare_features(df_proc)
            targets = df_proc['target_return_30d'].values
            
            X, y, _ = self.predictor.create_sequences(features, targets)
            
            # Split data: 60% train, 20% validation, 20% test
            train_split = int(0.6 * len(X))
            val_split = int(0.8 * len(X))
            
            X_test = X[val_split:]
            y_test = y[val_split:]
            test_data = df_proc.iloc[val_split + 60:]  # Adjust for sequence length
            
            if len(X_test) < 50:
                return {'error': 'Insufficient test data'}
            
            # Initialize portfolio
            portfolio = {
                'capital': initial_capital,
                'positions': {},
                'trade_history': [],
                'daily_returns': [],
                'equity_curve': [initial_capital],
                'drawdowns': [],
                'risk_metrics': []
            }
            
            # Run backtest
            ensemble_predictions, _, _ = self.predictor.predict_ensemble(X_test)
            
            for i in range(len(ensemble_predictions)):
                try:
                    # Get current data slice for this prediction
                    current_idx = val_split + 60 + i
                    if current_idx >= len(df_proc):
                        break
                        
                    current_data = df_proc.iloc[max(0, current_idx-60):current_idx+1]
                    
                    if len(current_data) < 10:  # Need minimum data
                        continue
                    
                    # Generate signal
                    signal = self.generate_advanced_signal(current_data)
                    
                    if signal is None or signal.primary_signal == 'HOLD':
                        portfolio['daily_returns'].append(0)
                        portfolio['equity_curve'].append(portfolio['capital'])
                        continue
                    
                    # Execute trade
                    actual_return = y_test[i]
                    
                    trade_result = self._execute_backtest_trade(
                        signal, portfolio['capital'], actual_return
                    )
                    
                    # Update portfolio
                    portfolio['capital'] = trade_result['new_capital']
                    portfolio['daily_returns'].append(trade_result['return'])
                    portfolio['equity_curve'].append(portfolio['capital'])
                    portfolio['trade_history'].append(trade_result['trade_details'])
                    
                    # Calculate rolling risk metrics
                    if len(portfolio['daily_returns']) >= 20:
                        recent_returns = np.array(portfolio['daily_returns'][-20:])
                        var_1d, cvar_1d = self.risk_manager.calculate_var_cvar(recent_returns)
                        
                        portfolio['risk_metrics'].append({
                            'var_1d': var_1d,
                            'cvar_1d': cvar_1d,
                            'volatility': np.std(recent_returns) * np.sqrt(365)
                        })
                
                except Exception as e:
                    logging.warning(f"Error in backtest iteration {i}: {e}")
                    continue
            
            # Calculate comprehensive performance metrics
            performance = self._calculate_comprehensive_performance(portfolio, initial_capital)
            
            logging.info("Comprehensive backtest completed:")
            logging.info(f"  Total Return: {performance['total_return']:.2%}")
            logging.info(f"  Annualized Return: {performance['annualized_return']:.2%}")
            logging.info(f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
            logging.info(f"  Calmar Ratio: {performance['calmar_ratio']:.3f}")
            logging.info(f"  Max Drawdown: {performance['max_drawdown']:.2%}")
            logging.info(f"  Win Rate: {performance['win_rate']:.2%}")
            
            return performance
            
        except Exception as e:
            logging.error(f"Error in comprehensive backtest: {e}")
            return {'error': str(e)}
    
    def _execute_backtest_trade(self, signal: AdvancedTradeSignal, 
                              capital: float, actual_return: float) -> Dict:
        """Execute trade in backtest with advanced risk management"""
        
        # Position sizing with multiple constraints
        position_value = capital * signal.final_position_size
        
        # Apply transaction costs
        entry_cost = position_value * 0.001  # 0.1% entry cost
        position_value_net = position_value - entry_cost
        
        # Calculate P&L based on actual return
        if signal.primary_signal == 'LONG':
            pnl = position_value_net * actual_return
        else:  # SHORT
            pnl = -position_value_net * actual_return
        
        # Apply stop loss and take profit logic
        max_loss = position_value_net * 0.05  # 5% max loss
        if pnl < -max_loss:
            pnl = -max_loss  # Stop loss triggered
            
        # Exit cost
        exit_cost = position_value_net * 0.001
        
        # Final P&L
        net_pnl = pnl - exit_cost
        new_capital = capital + net_pnl
        
        return {
            'new_capital': new_capital,
            'return': net_pnl / capital,
            'trade_details': {
                'signal_type': signal.primary_signal,
                'position_size': signal.final_position_size,
                'entry_price': signal.entry_price,
                'actual_return': actual_return,
                'pnl': net_pnl,
                'transaction_costs': entry_cost + exit_cost
            }
        }
    
    def _calculate_comprehensive_performance(self, portfolio: Dict, 
                                           initial_capital: float) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        returns = np.array(portfolio['daily_returns'])
        equity_curve = np.array(portfolio['equity_curve'])
        
        # Basic metrics
        total_return = (portfolio['capital'] - initial_capital) / initial_capital
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (365 / max(n_periods, 1)) - 1
        
        # Risk metrics
        returns_nonzero = returns[returns != 0]
        if len(returns_nonzero) > 0:
            volatility = np.std(returns_nonzero) * np.sqrt(365)
            sharpe_ratio = np.mean(returns_nonzero) / np.std(returns_nonzero) * np.sqrt(365) if np.std(returns_nonzero) > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Drawdown metrics
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Trade statistics
        trades = portfolio['trade_history']
        if trades:
            winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
            win_rate = winning_trades / len(trades)
            avg_win = np.mean([trade['pnl'] for trade in trades if trade['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([trade['pnl'] for trade in trades if trade['pnl'] < 0]) if len(trades) - winning_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * (len(trades) - winning_trades))) if avg_loss < 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Advanced metrics
        if len(returns) >= 20:
            var_95, cvar_95 = self.risk_manager.calculate_var_cvar(returns, 0.05)
            skewness = stats.skew(returns[returns != 0]) if len(returns[returns != 0]) > 3 else 0
            kurtosis = stats.kurtosis(returns[returns != 0]) if len(returns[returns != 0]) > 3 else 0
        else:
            var_95 = 0
            cvar_95 = 0
            skewness = 0
            kurtosis = 0
        
        return {
            'initial_capital': initial_capital,
            'final_capital': portfolio['capital'],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_trade_return': np.mean(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0,
            'trade_frequency': len(trades) / max(n_periods, 1),
            'equity_curve': equity_curve.tolist(),
            'drawdown_curve': drawdowns.tolist()
        }

# Example usage and enhanced configuration
if __name__ == "__main__":
    
    # Create enhanced configuration
    enhanced_config = {
        'trading': {
            'initial_capital': 100000,
            'max_leverage': 1.0,
            'position_limits': {
                'max_single_position': 0.25,  # 25% max position
                'max_daily_trades': 5,
                'min_trade_interval_hours': 6
            }
        },
        'risk_management': {
            'max_portfolio_var': 0.025,    # 2.5% daily VaR
            'volatility_target': 0.20,     # 20% annual volatility target
            'max_drawdown': 0.15,          # 15% max drawdown
            'position_sizing_method': 'kelly_refined',
            'stop_loss_method': 'dynamic_atr'
        }
    }
    
    # Save enhanced config
    with open('institutional_config.json', 'w') as f:
        json.dump(enhanced_config, f, indent=2)
    
    # Initialize system
    institutional_system = InstitutionalTradingSystem()
    
    # Load data and run backtest
    logging.info("Loading data for institutional backtest...")
    
    # This would typically load your actual data
    # For demo purposes, assume you have data loaded
    # data = load_all_data()  # Your data loading function

    btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data()
    # Assuming you have your df with engineered features
    df_news = add_vader_sentiment(df_news)
    df_newsdaily_sentiment = aggregate_daily_sentiment(df_news)

    # 3. Feature engineering
    df = engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment)

    results = institutional_system.run_comprehensive_backtest(df)

    logging.info("Institutional trading system initialized successfully!")
    logging.info("Key features enabled:")
    logging.info("  • Advanced regime detection")
    logging.info("  • Multi-strategy ensemble")
    logging.info("  • Institutional risk management")
    logging.info("  • Market microstructure awareness")
    logging.info("  • Dynamic position sizing")
    logging.info("  • Comprehensive performance analytics")