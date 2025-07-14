import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data
from sentiment import add_vader_sentiment, aggregate_daily_sentiment
from feature_engineering import engineer_features
from lstm_model import TunedBitcoinLSTMPredictor

if __name__ == "__main__":
    # 1. Load data
    btc_ohlcv, daily_oi, daily_funding_rate, df_news = load_all_data()

    # 2. Sentiment analysis
    df_news = add_vader_sentiment(df_news)
    df_newsdaily_sentiment = aggregate_daily_sentiment(df_news)

    # 3. Feature engineering
    df_daily = engineer_features(btc_ohlcv, daily_oi, daily_funding_rate, df_newsdaily_sentiment)

    # Save engineered features to CSV
    # df_daily.to_csv('engineered_features.csv', index=False)

    # 4. Modeling
    predictor = TunedBitcoinLSTMPredictor(df_daily)
    predictor.prepare_data()
    predictor.train_models()
    results = predictor.evaluate_models()
    predictor.plot_predictions(results)
    next_day = predictor.predict_next_day()
    print("Next day prediction:", next_day)