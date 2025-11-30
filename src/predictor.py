"""
Make predictions using trained models.
"""

import pandas as pd
import numpy as np
import os
import pickle
import config


def load_model(model_type='xgboost'):
    """
    Load a trained model from disk.
    
    Args:
        model_type: 'xgboost' (only XGBoost is supported)
    
    Returns:
        Trained model object
    """
    model_path = os.path.join(config.MODEL_SAVE_DIR, f'{model_type}_model.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded {model_type} model from {model_path}")
    return model


def get_feature_columns():
    """
    Get the list of all feature columns in the correct order.
    
    Returns:
        list: List of feature column names
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
    
    return feature_cols


def prepare_features_for_prediction(stock_features_dict, sentiment_features_dict=None, time_features_dict=None):
    """
    Prepare feature vector for prediction from individual feature values.
    
    Args:
        stock_features_dict: Dictionary with all stock/technical features
        sentiment_features_dict: Dictionary with sentiment features (optional)
        time_features_dict: Dictionary with time-based features (optional)
    
    Returns:
        numpy array: Feature vector ready for prediction
    """
    feature_cols = get_feature_columns()
    
    features = []
    for col in feature_cols:
        # Try stock_features_dict first
        if col in ['30d_return', '6m_return', '1y_return', 'volume_normalized', 'rsi', 'macd', 
                   'macd_signal', 'macd_histogram', 'bb_position', 'sma_5', 'sma_20', 'sma_50', 
                   'atr', 'momentum_5', 'momentum_10', 'volatility_5d', 'volatility_20d', 
                   'volatility_ratio', 'price_position', '30d_return_lag1', 'volume_lag1', 
                   'daily_return_lag1', '30d_return_change', 'volume_change']:
            features.append(stock_features_dict.get(col, 0.0))
        # Then sentiment_features_dict
        elif col in ['avg_sentiment', 'sentiment_volume', 'sentiment_momentum',
                     'sentiment_change', 'sentiment_volatility', 'weighted_sentiment']:
            if sentiment_features_dict:
                features.append(sentiment_features_dict.get(col, 0.0))
            else:
                features.append(0.0)
        # Then time_features_dict
        elif col in ['day_of_week', 'month', 'is_month_end', 'is_quarter_end']:
            if time_features_dict:
                features.append(time_features_dict.get(col, 0.0))
            else:
                features.append(0.0)
        else:
            features.append(0.0)
    
    return np.array(features).reshape(1, -1)


def predict(model_type='xgboost', stock_features_dict=None, sentiment_features_dict=None, 
            time_features_dict=None, features_array=None):
    """
    Make a prediction using the trained model.
    
    Args:
        model_type: 'xgboost' (only XGBoost is supported)
        stock_features_dict: Dictionary with stock features (if features_array not provided)
        sentiment_features_dict: Dictionary with sentiment features (if features_array not provided)
        time_features_dict: Dictionary with time-based features (if features_array not provided)
        features_array: Pre-computed feature array (alternative to dicts)
    
    Returns:
        int: Prediction (0 or 1)
        float: Prediction probability (if available)
    """
    model = load_model(model_type)
    
    if features_array is not None:
        X = features_array
    else:
        X = prepare_features_for_prediction(stock_features_dict, sentiment_features_dict, time_features_dict)
    
    prediction = model.predict(X)[0]
    
    # Get prediction probability if available
    try:
        proba = model.predict_proba(X)[0]
        probability = proba[1]  # Probability of class 1
    except:
        probability = None
    
    return prediction, probability


def predict_batch(model_type='xgboost', features_df=None):
    """
    Make predictions for a batch of samples.
    
    Args:
        model_type: 'xgboost' (only XGBoost is supported)
        features_df: DataFrame with feature columns
    
    Returns:
        numpy array: Predictions
        numpy array: Prediction probabilities (if available)
    """
    model = load_model(model_type)
    
    feature_cols = get_feature_columns()
    
    # Ensure all feature columns exist
    missing_cols = [col for col in feature_cols if col not in features_df.columns]
    if missing_cols:
        print(f"Warning: Missing features in dataframe: {missing_cols}")
        print("Filling with 0.0")
        for col in missing_cols:
            features_df[col] = 0.0
    
    X = features_df[feature_cols].values
    
    predictions = model.predict(X)
    
    try:
        probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1
    except:
        probabilities = None
    
    return predictions, probabilities


if __name__ == "__main__":
    # Test the predictor
    from feature_engineer import create_features, get_feature_matrix_and_target
    from data_fetcher import fetch_stock_data
    from news_fetcher import fetch_news_data
    from sentiment_analyzer import aggregate_daily_sentiment
    
    print("Testing prediction...")
    
    # Load features
    stock_df = fetch_stock_data()
    news_df = fetch_news_data()
    sentiment_df = aggregate_daily_sentiment(news_df)
    features_df = create_features(stock_df, sentiment_df)
    
    # Make a sample prediction
    sample_features = features_df.iloc[-1]  # Last row
    
    stock_features = {
        '30d_return': sample_features['30d_return'],
        '6m_return': sample_features['6m_return'],
        '1y_return': sample_features['1y_return'],
        'volume_normalized': sample_features['volume_normalized']
    }
    
    sentiment_features = {
        'avg_sentiment': sample_features['avg_sentiment'],
        'sentiment_volume': sample_features['sentiment_volume']
    }
    
    pred, prob = predict('xgboost', stock_features, sentiment_features)
    print(f"\nPrediction: {pred} (probability: {prob:.4f})")
    print(f"Actual target: {sample_features['target']}")

