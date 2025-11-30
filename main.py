"""
Main orchestration script for AMD Stock Prediction pipeline.
"""

import argparse
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_fetcher import fetch_stock_data
from news_fetcher import fetch_news_data
from sentiment_analyzer import aggregate_daily_sentiment
from feature_engineer import create_features, get_feature_matrix_and_target
from model_trainer import train_and_evaluate_models
from predictor import predict_batch
import config


def fetch_data():
    """Fetch stock and news data."""
    print("="*60)
    print("STEP 1: Fetching Stock Data")
    print("="*60)
    stock_df = fetch_stock_data()
    
    print("\n" + "="*60)
    print("STEP 2: Fetching News Data")
    print("="*60)
    print("Note: NewsAPI free tier constraints:")
    print(f"  - Max {config.NEWSAPI_MAX_ARTICLES_PER_DAY} articles per day")
    print(f"  - {config.NEWSAPI_HISTORICAL_LIMIT_DAYS}-day historical limit")
    print(f"  - {config.NEWSAPI_DELAY_HOURS}-hour delay (today's articles available tomorrow)")
    print("  - News data may be limited for historical dates\n")
    news_df = fetch_news_data()
    
    return stock_df, news_df


def process_sentiment(news_df):
    """Analyze sentiment of news articles."""
    print("\n" + "="*60)
    print("STEP 3: Analyzing News Sentiment")
    print("="*60)
    sentiment_df = aggregate_daily_sentiment(news_df)
    return sentiment_df


def engineer_features(stock_df, sentiment_df):
    """Create features and target variable."""
    print("\n" + "="*60)
    print("STEP 4: Engineering Features")
    print("="*60)
    features_df = create_features(stock_df, sentiment_df)
    return features_df


def train_models(features_df):
    """Train and evaluate models."""
    print("\n" + "="*60)
    print("STEP 5: Training Models")
    print("="*60)
    results = train_and_evaluate_models(features_df)
    return results


def run_full_pipeline():
    """Run the complete pipeline from data fetching to model training."""
    print("\n" + "="*60)
    print("AMD STOCK PREDICTION - FULL PIPELINE")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    
    try:
        # Step 1 & 2: Fetch data
        stock_df, news_df = fetch_data()
        
        # Step 3: Analyze sentiment
        sentiment_df = process_sentiment(news_df)
        
        # Step 4: Engineer features
        features_df = engineer_features(stock_df, sentiment_df)
        
        # Step 5: Train models
        results = train_models(features_df)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"End time: {datetime.now()}")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_training_only():
    """Train models using cached data."""
    print("\n" + "="*60)
    print("AMD STOCK PREDICTION - TRAINING ONLY")
    print("="*60)
    
    try:
        # Load cached data
        from feature_engineer import create_features
        from data_fetcher import fetch_stock_data
        from news_fetcher import fetch_news_data
        from sentiment_analyzer import aggregate_daily_sentiment
        
        print("Loading cached data...")
        stock_df = fetch_stock_data()
        news_df = fetch_news_data()
        sentiment_df = aggregate_daily_sentiment(news_df)
        features_df = create_features(stock_df, sentiment_df)
        
        # Train models
        results = train_models(features_df)
        
        print("\nTraining completed successfully!")
        return results
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_prediction():
    """Make predictions on test data."""
    print("\n" + "="*60)
    print("AMD STOCK PREDICTION - PREDICTIONS")
    print("="*60)
    
    try:
        from feature_engineer import create_features
        from data_fetcher import fetch_stock_data
        from news_fetcher import fetch_news_data
        from sentiment_analyzer import aggregate_daily_sentiment
        from model_trainer import time_based_split
        
        # Load data
        print("Loading data...")
        stock_df = fetch_stock_data()
        news_df = fetch_news_data()
        sentiment_df = aggregate_daily_sentiment(news_df)
        features_df = create_features(stock_df, sentiment_df)
        
        # Split data
        train_df, test_df = time_based_split(features_df)
        
        # Make predictions
        print("\nMaking predictions with XGBoost...")
        xgb_pred, xgb_proba = predict_batch('xgboost', test_df)
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        results_df = test_df[['target']].copy()
        results_df['xgb_pred'] = xgb_pred
        results_df['xgb_proba'] = xgb_proba
        
        print("\nFirst 10 predictions:")
        print(results_df.head(10))
        
        print(f"\nXGBoost accuracy: {(xgb_pred == test_df['target'].values).mean():.4f}")
        
        return results_df
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_predict_next():
    """Predict next market day using current data."""
    from predict_next_day import predict_next_market_day
    
    result = predict_next_market_day()
    return result


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(description='AMD Stock Prediction Pipeline')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'fetch', 'train', 'predict', 'predict-next'],
        default='all',
        help='Pipeline mode: all (full pipeline), fetch (data only), train (training only), predict (test data), predict-next (next market day)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        run_full_pipeline()
    elif args.mode == 'fetch':
        fetch_data()
        print("\nData fetching completed. Run with --mode train to train models.")
    elif args.mode == 'train':
        run_training_only()
    elif args.mode == 'predict':
        run_prediction()
    elif args.mode == 'predict-next':
        run_predict_next()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

