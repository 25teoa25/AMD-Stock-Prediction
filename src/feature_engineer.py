"""
Combine stock and sentiment features, create target variable.
"""

import pandas as pd
import numpy as np
import os
import config


def create_features(stock_df, sentiment_df, cache=True):
    """
    Merge stock and sentiment features and create target variable.
    
    Args:
        stock_df: DataFrame with stock features (from data_fetcher)
        sentiment_df: DataFrame with sentiment features (from sentiment_analyzer)
        cache: Whether to use cached processed data if available
    
    Returns:
        DataFrame with features and target variable
    """
    cache_file = os.path.join(config.PROCESSED_DATA_DIR, "features_and_target.csv")
    
    # Check cache
    if cache and os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df
    
    print("Creating features and target variable...")
    
    # Start with stock features
    df = stock_df.copy()
    
    # Merge sentiment features by date
    # Convert sentiment index to date-only for matching
    if not sentiment_df.empty:
        sentiment_df_copy = sentiment_df.copy()
        sentiment_df_copy.index = sentiment_df_copy.index.date
        df['date_only'] = df.index.date
        
        # Merge on date
        df = df.merge(
            sentiment_df_copy[['avg_sentiment', 'sentiment_volume']],
            left_on='date_only',
            right_index=True,
            how='left'
        )
        
        # Drop temporary date column
        df.drop('date_only', axis=1, inplace=True)
        
        # Fill missing sentiment with 0 (no news days)
        df['avg_sentiment'] = df['avg_sentiment'].fillna(0.0)
        df['sentiment_volume'] = df['sentiment_volume'].fillna(0.0)
    else:
        # No sentiment data available
        df['avg_sentiment'] = 0.0
        df['sentiment_volume'] = 0.0
    
    # Create target variable: 1 if next_day_return > 0.5%, else 0
    df['target'] = (df['next_day_return'] > config.TARGET_THRESHOLD).astype(int)
    
    # Remove rows where target is NaN (last day has no next day return)
    df = df.dropna(subset=['target'])
    
    # Select feature columns (exclude Close and next_day_return from features)
    feature_cols = ['30d_return', '6m_return', '1y_return', 'volume_normalized',
                    'avg_sentiment', 'sentiment_volume']
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Keep target and next_day_return for analysis
    final_cols = feature_cols + ['target', 'next_day_return']
    df = df[final_cols]
    
    # Remove any remaining NaN values
    df = df.dropna()
    
    # Save to cache
    if cache:
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        df.to_csv(cache_file)
        print(f"Features cached to {cache_file}")
    
    print(f"Created features for {len(df)} days")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df


def get_feature_matrix_and_target(df):
    """
    Extract feature matrix X and target vector y from DataFrame.
    
    Args:
        df: DataFrame with features and target
    
    Returns:
        X: Feature matrix (numpy array or DataFrame)
        y: Target vector (numpy array or Series)
        feature_names: List of feature names
    """
    feature_cols = ['30d_return', '6m_return', '1y_return', 'volume_normalized',
                    'avg_sentiment', 'sentiment_volume']
    
    X = df[feature_cols].values
    y = df['target'].values
    feature_names = feature_cols
    
    return X, y, feature_names


if __name__ == "__main__":
    # Test the feature engineer
    from data_fetcher import fetch_stock_data
    from news_fetcher import fetch_news_data
    from sentiment_analyzer import aggregate_daily_sentiment
    
    print("Testing feature engineering...")
    stock_df = fetch_stock_data()
    news_df = fetch_news_data()
    sentiment_df = aggregate_daily_sentiment(news_df)
    
    features_df = create_features(stock_df, sentiment_df)
    print("\nFeatures DataFrame:")
    print(features_df.head())
    print("\nFeature statistics:")
    print(features_df.describe())
    
    X, y, feature_names = get_feature_matrix_and_target(features_df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Feature names: {feature_names}")

