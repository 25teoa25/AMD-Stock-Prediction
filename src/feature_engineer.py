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
        # Normalize timezone - ensure index is timezone-naive
        # Convert via date strings to completely remove timezone info
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                # Convert to date strings and back - this removes all timezone info
                df.index = pd.to_datetime([str(d.date()) for d in df.index])
            except:
                # Fallback: try tz_localize
                try:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                except:
                    pass
        return df
    
    print("Creating features and target variable...")
    
    # Start with stock features
    df = stock_df.copy()
    
    # Normalize timezone - ensure index is timezone-naive
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            # Convert to date strings and back - this removes all timezone info
            df.index = pd.to_datetime([str(d.date()) for d in df.index])
        except:
            # Fallback: try tz_localize
            try:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            except:
                pass
    
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
        
        # Sentiment-derived features
        df['sentiment_momentum'] = df['avg_sentiment'].rolling(5).mean()
        df['sentiment_change'] = df['avg_sentiment'] - df['avg_sentiment'].shift(1)
        df['sentiment_volatility'] = df['avg_sentiment'].rolling(10).std()
        df['weighted_sentiment'] = df['avg_sentiment'] * np.log1p(df['sentiment_volume'])
        
        # Fill NaN values for sentiment-derived features
        df['sentiment_momentum'] = df['sentiment_momentum'].fillna(0.0)
        df['sentiment_change'] = df['sentiment_change'].fillna(0.0)
        df['sentiment_volatility'] = df['sentiment_volatility'].fillna(0.0)
        df['weighted_sentiment'] = df['weighted_sentiment'].fillna(0.0)
    else:
        # No sentiment data available
        df['avg_sentiment'] = 0.0
        df['sentiment_volume'] = 0.0
        df['sentiment_momentum'] = 0.0
        df['sentiment_change'] = 0.0
        df['sentiment_volatility'] = 0.0
        df['weighted_sentiment'] = 0.0
    
    # Time-based features
    df['day_of_week'] = df.index.dayofweek / 6.0  # Normalize to 0-1
    df['month'] = df.index.month / 12.0  # Normalize to 0-1
    df['is_month_end'] = (df.index.day >= 25).astype(int)
    df['is_quarter_end'] = df.index.month.isin([3, 6, 9, 12]).astype(int)
    
    # Create target variable: 1 if next_day_return > 0.5%, else 0
    df['target'] = (df['next_day_return'] > config.TARGET_THRESHOLD).astype(int)
    
    # Remove rows where target is NaN (last day has no next day return)
    df = df.dropna(subset=['target'])
    
    # Define all feature columns (exclude Close and next_day_return from features)
    # Stock return features
    stock_return_features = ['30d_return', '6m_return', '1y_return', 'volume_normalized']
    
    # Technical indicators
    technical_features = ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                          'sma_5', 'sma_20', 'sma_50', 'atr']
    
    # Momentum and volatility features
    momentum_volatility_features = ['momentum_5', 'momentum_10',
                                   'volatility_5d', 'volatility_20d', 'volatility_ratio', 'price_position']
    
    # Lag features
    lag_features = ['30d_return_lag1', 'volume_lag1', 'daily_return_lag1',
                   '30d_return_change', 'volume_change']
    
    # Sentiment features
    sentiment_features = ['avg_sentiment', 'sentiment_volume', 'sentiment_momentum',
                         'sentiment_change', 'sentiment_volatility', 'weighted_sentiment']
    
    # Time-based features
    time_features = ['day_of_week', 'month', 'is_month_end', 'is_quarter_end']
    
    # Combine all features
    feature_cols = (stock_return_features + technical_features + momentum_volatility_features +
                   lag_features + sentiment_features + time_features)
    
    # Ensure all feature columns exist (fill missing with 0.0)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Keep target and next_day_return for analysis
    final_cols = feature_cols + ['target', 'next_day_return']
    df = df[final_cols]
    
    # Remove any remaining NaN values
    df = df.dropna()
    
    # Ensure index is timezone-naive before saving
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime([str(d.date()) for d in df.index])
        except:
            try:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            except:
                pass
    
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
    # Stock return features
    stock_return_features = ['30d_return', '6m_return', '1y_return', 'volume_normalized']
    
    # Technical indicators
    technical_features = ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                          'sma_5', 'sma_20', 'sma_50', 'atr']
    
    # Momentum and volatility features
    momentum_volatility_features = ['momentum_5', 'momentum_10',
                                   'volatility_5d', 'volatility_20d', 'volatility_ratio', 'price_position']
    
    # Lag features
    lag_features = ['30d_return_lag1', 'volume_lag1', 'daily_return_lag1',
                   '30d_return_change', 'volume_change']
    
    # Sentiment features
    sentiment_features = ['avg_sentiment', 'sentiment_volume', 'sentiment_momentum',
                         'sentiment_change', 'sentiment_volatility', 'weighted_sentiment']
    
    # Time-based features
    time_features = ['day_of_week', 'month', 'is_month_end', 'is_quarter_end']
    
    # Combine all features
    feature_cols = (stock_return_features + technical_features + momentum_volatility_features +
                   lag_features + sentiment_features + time_features)
    
    X = df[feature_cols].values
    y = df['target'].values
    feature_names = feature_cols
    
    return X, y, feature_names


def prepare_single_day_features(stock_features_dict, sentiment_features_dict=None, time_features_dict=None):
    """
    Prepare feature vector for a single day prediction.
    
    Args:
        stock_features_dict: Dictionary with stock features (all technical indicators, etc.)
        sentiment_features_dict: Dictionary with sentiment features (optional)
        time_features_dict: Dictionary with time-based features (optional)
    
    Returns:
        numpy array: Feature vector ready for prediction
    """
    # Stock return features
    stock_return_features = ['30d_return', '6m_return', '1y_return', 'volume_normalized']
    
    # Technical indicators
    technical_features = ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                          'sma_5', 'sma_20', 'sma_50', 'atr']
    
    # Momentum and volatility features
    momentum_volatility_features = ['momentum_5', 'momentum_10',
                                   'volatility_5d', 'volatility_20d', 'volatility_ratio', 'price_position']
    
    # Lag features
    lag_features = ['30d_return_lag1', 'volume_lag1', 'daily_return_lag1',
                   '30d_return_change', 'volume_change']
    
    # Sentiment features
    sentiment_features = ['avg_sentiment', 'sentiment_volume', 'sentiment_momentum',
                         'sentiment_change', 'sentiment_volatility', 'weighted_sentiment']
    
    # Time-based features
    time_features = ['day_of_week', 'month', 'is_month_end', 'is_quarter_end']
    
    # Combine all features
    feature_cols = (stock_return_features + technical_features + momentum_volatility_features +
                   lag_features + sentiment_features + time_features)
    
    features = []
    for col in feature_cols:
        # Try stock_features_dict first
        if col in stock_return_features + technical_features + momentum_volatility_features + lag_features:
            features.append(stock_features_dict.get(col, 0.0))
        # Then sentiment_features_dict
        elif col in sentiment_features:
            if sentiment_features_dict:
                features.append(sentiment_features_dict.get(col, 0.0))
            else:
                features.append(0.0)
        # Then time_features_dict
        elif col in time_features:
            if time_features_dict:
                features.append(time_features_dict.get(col, 0.0))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
    
    return np.array(features).reshape(1, -1)


def save_volume_normalization_params(stock_df, save_path=None):
    """
    Save volume normalization parameters from training data.
    
    Args:
        stock_df: DataFrame with Volume column
        save_path: Path to save normalization parameters (default: processed_data_dir)
    """
    if save_path is None:
        save_path = os.path.join(config.PROCESSED_DATA_DIR, "volume_normalization_params.json")
    
    volume_mean = stock_df['Volume'].mean()
    volume_std = stock_df['Volume'].std()
    
    norm_params = {
        'mean': float(volume_mean),
        'std': float(volume_std)
    }
    
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    import json
    with open(save_path, 'w') as f:
        json.dump(norm_params, f)
    
    print(f"Saved volume normalization parameters to {save_path}")


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

