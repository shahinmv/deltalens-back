from model_analyze import RegimeAwareBitcoinPredictor, add_window_analysis_to_predictor, ImprovedBitcoinPredictor
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
from feature_engineering import engineer_features  
from data_loader import load_all_data
from sentiment import add_vader_sentiment, aggregate_daily_sentiment



btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data()
# Assuming you have your df with engineered features
df_news = add_vader_sentiment(df_news)
df_newsdaily_sentiment = aggregate_daily_sentiment(df_news)

# 3. Feature engineering
df = engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment)

# Run Training Window Analysis on Your Feature-Engineered Data

# First, add the analysis capability to your predictor class

add_window_analysis_to_predictor()

# Create predictor instance
predictor = ImprovedBitcoinPredictor(
    sequence_length=60, 
    prediction_horizon=30,
    ridge_alpha=3.0  # Higher regularization for better generalization
)

# Check your data
print("Dataset Overview:")
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Total days: {len(df)}")
print(f"Approximate months: {len(df) // 30}")
print(f"Available features: {list(df.columns)}")

# Define window lengths to test (in months)
# Covering key periods in Bitcoin's evolution
window_lengths = [
    6,   # 6 months - very recent, high adaptability
    12,  # 1 year - recent trends
    18,  # 1.5 years - medium-term patterns
    24,  # 2 years - captures full cycles
    36,  # 3 years - longer-term stability
    48,  # 4 years - includes full halving cycle
    60,  # 5 years - pre-institutional vs institutional
    len(df) // 30  # All available data
]

print(f"\nTesting window lengths: {window_lengths} months")

# Run the comprehensive analysis
print("\n" + "="*60)
print("STARTING TRAINING WINDOW LENGTH ANALYSIS")
print("="*60)
print("This will test different training window lengths using walk-forward validation...")
print("Each window will be trained and tested on multiple time periods.")
print("Expected runtime: 5-15 minutes depending on data size and hardware.")
print("="*60 + "\n")

try:
    # Run analysis with more test periods for robust results
    results = predictor.analyze_training_window(
        df=df,
        window_lengths=window_lengths,
        test_periods=8  # Test on 8 different time periods for robust validation
    )
    
    if results is None:
        print("❌ Analysis failed - insufficient data or other issues")
    else:
        print("\n" + "🎯" * 20)
        print("ANALYSIS COMPLETE - RESULTS SUMMARY")
        print("🎯" * 20)
        
        recommendation = results['recommendation']
        
        print(f"\n📊 OPTIMAL TRAINING WINDOW")
        print(f"   Recommended: {recommendation['optimal_window_months']} months ({recommendation['optimal_window_years']:.1f} years)")
        print(f"   Composite Score: {recommendation['composite_score']:.3f}")
        
        print(f"\n💡 KEY INSIGHTS")
        for reason in recommendation['reasoning']:
            print(f"   {reason}")
        
        print(f"\n🔄 ALTERNATIVES")
        if recommendation['alternatives']:
            for alt in recommendation['alternatives'][:2]:  # Show top 2 alternatives
                print(f"   {alt['window_months']} months ({alt['window_years']:.1f} years): {alt['note']}")
        else:
            print("   No close alternatives found")
            
        print(f"\n⚙️  IMPLEMENTATION RECOMMENDATIONS")
        for note in recommendation['implementation_notes']:
            print(f"   {note}")
        
        # Answer your original question directly
        print(f"\n" + "="*60)
        print("🎯 ANSWER TO YOUR ORIGINAL QUESTION:")
        print("="*60)
        
        optimal_months = recommendation['optimal_window_months']
        total_months = len(df) // 30
        
        if optimal_months >= total_months * 0.8:
            answer = "✅ USE ALL AVAILABLE DATA"
            explanation = f"The analysis shows using {optimal_months} months ({optimal_months/12:.1f} years) of training data is optimal, which represents {optimal_months/total_months:.1%} of your available data. This suggests that older Bitcoin data still contains valuable patterns for prediction."
        elif optimal_months <= 24:
            answer = "✅ USE RECENT DATA ONLY"
            explanation = f"The analysis shows using only {optimal_months} months ({optimal_months/12:.1f} years) of recent data is optimal. This suggests significant concept drift - older Bitcoin patterns are less relevant for current predictions."
        else:
            answer = "✅ USE BALANCED HISTORICAL WINDOW"
            explanation = f"The analysis shows using {optimal_months} months ({optimal_months/12:.1f} years) of data is optimal. This represents a sweet spot between capturing long-term patterns while avoiding outdated market regimes."
        
        print(f"\n{answer}")
        print(f"\n{explanation}")
        
        # Concept drift analysis
        detailed_results = results['detailed_results']
        df_results = pd.DataFrame(detailed_results)
        
        # Check if longer windows consistently perform better
        mae_correlation = results['analysis']['correlations'].get('avg_mae', {}).get('correlation', 0)
        
        print(f"\n📈 CONCEPT DRIFT ANALYSIS:")
        if mae_correlation < -0.3:
            print("   📊 STABLE PATTERNS: Longer training windows consistently improve performance")
            print("   💡 Interpretation: Bitcoin's underlying patterns have remained relatively stable")
            print("   🎯 Strategy: Use longer training windows, retrain less frequently")
        elif mae_correlation > 0.3:
            print("   🔄 HIGH CONCEPT DRIFT: Shorter windows perform better")
            print("   💡 Interpretation: Bitcoin's market structure has evolved significantly")
            print("   🎯 Strategy: Use shorter windows, retrain more frequently")
        else:
            print("   ⚖️  MODERATE EVOLUTION: Mixed evidence of concept drift")
            print("   💡 Interpretation: Some patterns persist, others have changed")
            print("   🎯 Strategy: Use medium-length windows with regular retraining")
        
        # Performance summary
        best_result = df_results[df_results['window_months'] == optimal_months].iloc[0]
        print(f"\n📊 EXPECTED PERFORMANCE WITH OPTIMAL WINDOW:")
        print(f"   Mean Absolute Error: {best_result['avg_mae']:.6f}")
        print(f"   Sharpe Ratio: {best_result['avg_sharpe_ratio']:.3f}")
        print(f"   Direction Accuracy: {best_result['avg_direction_accuracy']:.2%}")
        print(f"   Model Stability Score: {best_result['stability_score']:.3f}")
        
        # Save results for further analysis
        results['df_results'] = df_results
        
        print(f"\n✅ Analysis complete! Results saved in 'results' variable.")
        print(f"📊 Visualizations have been displayed above.")
        
except Exception as e:
    print(f"❌ Error during analysis: {e}")
    print("\nTroubleshooting suggestions:")
    print("1. Check that your df has sufficient data (>500 rows)")
    print("2. Ensure df has datetime index")
    print("3. Verify required columns exist (close, volume, etc.)")
    print("4. Check for NaN values in critical columns")
    
    # Basic data validation
    print(f"\nData validation:")
    print(f"DataFrame shape: {df.shape}")
    print(f"Index type: {type(df.index)}")
    print(f"Required columns present: {all(col in df.columns for col in ['close', 'volume'])}")
    print(f"NaN percentage: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2%}")

# Optional: Quick test with a simple window comparison
print(f"\n" + "="*60)
print("QUICK COMPARISON: RECENT vs ALL DATA")
print("="*60)

try:
    # Test 2 years vs all data quickly
    recent_months = 24
    all_months = len(df) // 30
    
    print(f"Comparing {recent_months} months vs {all_months} months of training data...")
    
    quick_results = predictor.analyze_training_window(
        df=df,
        window_lengths=[recent_months, all_months],
        test_periods=3  # Fewer test periods for speed
    )
    
    if quick_results:
        quick_df = pd.DataFrame(quick_results['detailed_results'])
        
        print(f"\nQUICK RESULTS:")
        print(f"{'Metric':<20} {'Recent (24m)':<15} {'All Data':<15} {'Winner':<10}")
        print("-" * 65)
        
        recent = quick_df[quick_df['window_months'] == recent_months].iloc[0]
        all_data = quick_df[quick_df['window_months'] == all_months].iloc[0]
        
        mae_winner = "Recent" if recent['avg_mae'] < all_data['avg_mae'] else "All Data"
        sharpe_winner = "Recent" if recent['avg_sharpe_ratio'] > all_data['avg_sharpe_ratio'] else "All Data"
        stability_winner = "Recent" if recent['stability_score'] < all_data['stability_score'] else "All Data"
        
        print(f"{'MAE (lower better)':<20} {recent['avg_mae']:<15.6f} {all_data['avg_mae']:<15.6f} {mae_winner:<10}")
        print(f"{'Sharpe (higher better)':<20} {recent['avg_sharpe_ratio']:<15.3f} {all_data['avg_sharpe_ratio']:<15.3f} {sharpe_winner:<10}")
        print(f"{'Stability (lower better)':<20} {recent['stability_score']:<15.3f} {all_data['stability_score']:<15.3f} {stability_winner:<10}")
        
        # Quick recommendation
        recent_wins = sum([mae_winner == "Recent", sharpe_winner == "Recent", stability_winner == "Recent"])
        if recent_wins >= 2:
            print(f"\n🎯 QUICK VERDICT: Recent data (24 months) performs better")
        else:
            print(f"\n🎯 QUICK VERDICT: All available data performs better")
            
except Exception as e:
    print(f"Quick comparison failed: {e}")

print(f"\n" + "🎉" * 20)
print("TRAINING WINDOW ANALYSIS COMPLETE!")
print("🎉" * 20)