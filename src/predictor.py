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
        model_type: 'xgboost' or 'random_forest'
    
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


def prepare_features_for_prediction(stock_features_dict, sentiment_features_dict=None):
    """
    Prepare feature vector for prediction from individual feature values.
    
    Args:
        stock_features_dict: Dictionary with keys: 30d_return, 6m_return, 1y_return, volume_normalized
        sentiment_features_dict: Dictionary with keys: avg_sentiment, sentiment_volume (optional)
    
    Returns:
        numpy array: Feature vector ready for prediction
    """
    feature_cols = ['30d_return', '6m_return', '1y_return', 'volume_normalized',
                    'avg_sentiment', 'sentiment_volume']
    
    features = []
    features.append(stock_features_dict.get('30d_return', 0.0))
    features.append(stock_features_dict.get('6m_return', 0.0))
    features.append(stock_features_dict.get('1y_return', 0.0))
    features.append(stock_features_dict.get('volume_normalized', 0.0))
    
    if sentiment_features_dict:
        features.append(sentiment_features_dict.get('avg_sentiment', 0.0))
        features.append(sentiment_features_dict.get('sentiment_volume', 0.0))
    else:
        features.append(0.0)  # avg_sentiment
        features.append(0.0)  # sentiment_volume
    
    return np.array(features).reshape(1, -1)


def predict(model_type='xgboost', stock_features_dict=None, sentiment_features_dict=None, 
            features_array=None):
    """
    Make a prediction using the trained model.
    
    Args:
        model_type: 'xgboost' or 'random_forest'
        stock_features_dict: Dictionary with stock features (if features_array not provided)
        sentiment_features_dict: Dictionary with sentiment features (if features_array not provided)
        features_array: Pre-computed feature array (alternative to dicts)
    
    Returns:
        int: Prediction (0 or 1)
        float: Prediction probability (if available)
    """
    model = load_model(model_type)
    
    if features_array is not None:
        X = features_array
    else:
        X = prepare_features_for_prediction(stock_features_dict, sentiment_features_dict)
    
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
        model_type: 'xgboost' or 'random_forest'
        features_df: DataFrame with feature columns
    
    Returns:
        numpy array: Predictions
        numpy array: Prediction probabilities (if available)
    """
    model = load_model(model_type)
    
    feature_cols = ['30d_return', '6m_return', '1y_return', 'volume_normalized',
                    'avg_sentiment', 'sentiment_volume']
    
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

