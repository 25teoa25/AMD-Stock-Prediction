"""
Predict next market day stock movement using current data.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import get_latest_trading_day_features
from news_fetcher import fetch_news_data
from sentiment_analyzer import aggregate_daily_sentiment
from predictor import predict
import config


def get_today_sentiment():
    """
    Get sentiment for news articles to use for next-day prediction.
    Due to NewsAPI's 24-hour delay, we use yesterday's news (most recent available).
    
    Returns:
        dict: Dictionary with avg_sentiment and sentiment_volume
    """
    # NewsAPI has 24-hour delay, so today's articles are available tomorrow
    # For predicting tomorrow, we use yesterday's news (most recent available)
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)
    
    # Fetch news from day_before to yesterday (yesterday is the most recent available)
    start_date = day_before.strftime('%Y-%m-%d')
    end_date = yesterday.strftime('%Y-%m-%d')
    
    print(f"Fetching news for {start_date} to {end_date}...")
    print(f"Note: NewsAPI has 24-hour delay. Using yesterday's news ({end_date}) for prediction.")
    
    try:
        news_df = fetch_news_data(start_date=start_date, end_date=end_date, cache=False)
        
        if news_df.empty:
            print("No news articles found. Using default sentiment (0.0).")
            return {'avg_sentiment': 0.0, 'sentiment_volume': 0.0}
        
        # Analyze sentiment
        sentiment_df = aggregate_daily_sentiment(news_df, cache=False)
        
        if sentiment_df.empty:
            print("No sentiment data available. Using default sentiment (0.0).")
            return {'avg_sentiment': 0.0, 'sentiment_volume': 0.0}
        
        # Get most recent sentiment (should be yesterday's)
        latest_sentiment = sentiment_df.iloc[-1]
        
        return {
            'avg_sentiment': float(latest_sentiment['avg_sentiment']),
            'sentiment_volume': float(latest_sentiment['sentiment_volume'])
        }
        
    except Exception as e:
        print(f"Error fetching/analyzing sentiment: {e}")
        print("Using default sentiment (0.0).")
        return {'avg_sentiment': 0.0, 'sentiment_volume': 0.0}


def predict_next_market_day():
    """
    Predict whether AMD stock will increase by more than 0.5% on the next trading day.
    
    Returns:
        dict: Prediction results with prediction, probability, and feature values
    """
    print("="*60)
    print("PREDICTING NEXT MARKET DAY")
    print("="*60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Step 1: Get latest stock features
        print("Step 1: Fetching latest stock data...")
        stock_features = get_latest_trading_day_features()
        
        print(f"Latest trading day: {stock_features['date'].strftime('%Y-%m-%d')}")
        print(f"Close price: ${stock_features['close_price']:.2f}")
        print(f"30-day return: {stock_features['30d_return']:.4f}")
        print(f"6-month return: {stock_features['6m_return']:.4f}")
        print(f"1-year return: {stock_features['1y_return']:.4f}")
        print(f"Volume (normalized): {stock_features['volume_normalized']:.4f}\n")
        
        # Step 2: Get sentiment (using yesterday's news due to 24-hour delay)
        print("Step 2: Fetching and analyzing news sentiment...")
        print("(Using yesterday's news due to NewsAPI 24-hour delay)")
        sentiment_features = get_today_sentiment()
        
        print(f"Average sentiment: {sentiment_features['avg_sentiment']:.4f}")
        print(f"Number of articles: {sentiment_features['sentiment_volume']:.0f}\n")
        
        # Step 3: Prepare features
        print("Step 3: Preparing features for prediction...")
        # Include all stock features from get_latest_trading_day_features
        stock_features_dict = {
            '30d_return': stock_features.get('30d_return', 0.0),
            '6m_return': stock_features.get('6m_return', 0.0),
            '1y_return': stock_features.get('1y_return', 0.0),
            'volume_normalized': stock_features.get('volume_normalized', 0.0),
            'rsi': stock_features.get('rsi', 0.5),
            'macd': stock_features.get('macd', 0.0),
            'macd_signal': stock_features.get('macd_signal', 0.0),
            'macd_histogram': stock_features.get('macd_histogram', 0.0),
            'bb_position': stock_features.get('bb_position', 0.5),
            'sma_5': stock_features.get('sma_5', 0.0),
            'sma_20': stock_features.get('sma_20', 0.0),
            'sma_50': stock_features.get('sma_50', 0.0),
            'atr': stock_features.get('atr', 0.0),
            'momentum_5': stock_features.get('momentum_5', 0.0),
            'momentum_10': stock_features.get('momentum_10', 0.0),
            'volatility_5d': stock_features.get('volatility_5d', 0.0),
            'volatility_20d': stock_features.get('volatility_20d', 0.0),
            'volatility_ratio': stock_features.get('volatility_ratio', 1.0),
            'price_position': stock_features.get('price_position', 0.5),
            '30d_return_lag1': stock_features.get('30d_return_lag1', 0.0),
            'volume_lag1': stock_features.get('volume_lag1', 0.0),
            'daily_return_lag1': stock_features.get('daily_return_lag1', 0.0),
            '30d_return_change': stock_features.get('30d_return_change', 0.0),
            'volume_change': stock_features.get('volume_change', 0.0)
        }
        
        # Add sentiment-derived features (calculate from base sentiment)
        sentiment_features_dict = {
            'avg_sentiment': sentiment_features.get('avg_sentiment', 0.0),
            'sentiment_volume': sentiment_features.get('sentiment_volume', 0.0),
            'sentiment_momentum': sentiment_features.get('avg_sentiment', 0.0),  # Simplified
            'sentiment_change': 0.0,  # Would need previous day's sentiment
            'sentiment_volatility': 0.0,  # Would need history
            'weighted_sentiment': sentiment_features.get('avg_sentiment', 0.0) * 
                                (1.0 if sentiment_features.get('sentiment_volume', 0) == 0 
                                 else np.log1p(sentiment_features.get('sentiment_volume', 0)))
        }
        
        # Time-based features
        latest_date = stock_features['date']
        time_features_dict = {
            'day_of_week': latest_date.dayofweek / 6.0,
            'month': latest_date.month / 12.0,
            'is_month_end': 1 if latest_date.day >= 25 else 0,
            'is_quarter_end': 1 if latest_date.month in [3, 6, 9, 12] else 0
        }
        
        # Step 4: Make prediction
        print("Step 4: Making prediction with XGBoost model...\n")
        prediction, probability = predict(
            'xgboost',
            stock_features_dict=stock_features_dict,
            sentiment_features_dict=sentiment_features_dict,
            time_features_dict=time_features_dict
        )
        
        # Step 5: Display results
        print("="*60)
        print("PREDICTION RESULT")
        print("="*60)
        
        if prediction == 1:
            result_text = "WILL INCREASE by more than 0.5%"
            result_emoji = "📈"
        else:
            result_text = "WILL NOT increase by more than 0.5%"
            result_emoji = "📉"
        
        print(f"\n{result_emoji} Prediction: {result_text}")
        print(f"Confidence: {probability:.2%}" if probability else "Confidence: N/A")
        print(f"\nNext trading day prediction based on data from: {stock_features['date'].strftime('%Y-%m-%d')}")
        
        return {
            'prediction': int(prediction),
            'probability': float(probability) if probability else None,
            'will_increase': bool(prediction == 1),
            'latest_trading_day': stock_features['date'].strftime('%Y-%m-%d'),
            'features': {
                'stock': stock_features_dict,
                'sentiment': sentiment_features_dict,
                'time': time_features_dict
            }
        }
        
    except FileNotFoundError as e:
        print(f"\nERROR: Model not found. Please train the model first:")
        print(f"  python main.py --mode train")
        return None
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = predict_next_market_day()
    
    if result:
        print("\n" + "="*60)
        print("Prediction completed successfully!")
        print("="*60)

