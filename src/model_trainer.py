"""
Train XGBoost model for AMD stock prediction.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import config


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


def time_based_split(df, train_end_date=config.TRAIN_END_DATE):
    """
    Split data based on time (not random).
    
    Args:
        df: DataFrame with date index
        train_end_date: Last date for training set (inclusive)
    
    Returns:
        train_df, test_df: Training and testing DataFrames
    """
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
    
    train_end = pd.to_datetime(train_end_date)
    # Ensure train_end is also timezone-naive
    if hasattr(train_end, 'tz') and train_end.tz is not None:
        train_end = train_end.tz_localize(None)
    
    train_df = df[df.index <= train_end].copy()
    test_df = df[df.index > train_end].copy()
    
    print(f"Training set: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Test set: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")
    
    return train_df, test_df


def train_and_evaluate_models(features_df, save_models=True):
    """
    Train XGBoost model and evaluate it.
    
    Args:
        features_df: DataFrame with features and target
        save_models: Whether to save trained model to disk
    
    Returns:
        dict: Results dictionary with model performance metrics
    """
    # Time-based split
    train_df, test_df = time_based_split(features_df)
    
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Insufficient data for train/test split")
    
    # Extract features and target
    feature_cols = get_feature_columns()
    
    # Ensure all feature columns exist in the dataframe
    missing_cols = [col for col in feature_cols if col not in train_df.columns]
    if missing_cols:
        print(f"Warning: Missing features in dataframe: {missing_cols}")
        print("Filling with 0.0")
        for col in missing_cols:
            train_df[col] = 0.0
            test_df[col] = 0.0
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # Check class distribution
    train_class_dist = pd.Series(y_train).value_counts().to_dict()
    test_class_dist = pd.Series(y_test).value_counts().to_dict()
    print(f"\nTraining set target distribution: {train_class_dist}")
    print(f"Test set target distribution: {test_class_dist}")
    
    # Calculate class imbalance ratio
    if 0 in train_class_dist and 1 in train_class_dist:
        class_ratio = train_class_dist[0] / train_class_dist[1]
        print(f"Class imbalance ratio (0/1): {class_ratio:.2f}")
    else:
        class_ratio = 1.0
    
    results = {}
    
    # Baseline: Always predict 0
    baseline_pred = np.zeros_like(y_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_prec = precision_score(y_test, baseline_pred, zero_division=0)
    baseline_rec = recall_score(y_test, baseline_pred, zero_division=0)
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
    
    results['baseline'] = {
        'accuracy': baseline_acc,
        'precision': baseline_prec,
        'recall': baseline_rec,
        'f1_score': baseline_f1
    }
    
    print("\n" + "="*60)
    print("BASELINE (Always predict 0)")
    print("="*60)
    print(f"Accuracy:  {baseline_acc:.4f}")
    print(f"Precision: {baseline_prec:.4f}")
    print(f"Recall:    {baseline_rec:.4f}")
    print(f"F1-Score:  {baseline_f1:.4f}")
    
    # Train XGBoost with hyperparameter tuning
    print("\n" + "="*60)
    print("XGBOOST - HYPERPARAMETER TUNING")
    print("="*60)
    
    # Set scale_pos_weight for class imbalance if needed
    scale_pos_weight = 1.0
    if class_ratio > 1.5:
        scale_pos_weight = class_ratio
        print(f"Detected class imbalance. Using scale_pos_weight={scale_pos_weight:.2f}")
    elif class_ratio < 0.67:  # Inverse ratio if class 1 is more common
        scale_pos_weight = 1.0 / class_ratio
        print(f"Detected class imbalance. Using scale_pos_weight={scale_pos_weight:.2f}")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    # Base XGBoost model
    xgb_base = xgb.XGBClassifier(
        random_state=config.RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight
    )
    
    # Use TimeSeriesSplit for cross-validation (3 folds)
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("Performing grid search with TimeSeriesSplit (3 folds)...")
    print("This may take several minutes...")
    
    # Grid search with time series cross-validation
    grid_search = GridSearchCV(
        xgb_base,
        param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Use best model
    xgb_model = grid_search.best_estimator_
    xgb_pred = xgb_model.predict(X_test)
    
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
    xgb_rec = recall_score(y_test, xgb_pred, zero_division=0)
    xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
    
    results['xgboost'] = {
        'accuracy': xgb_acc,
        'precision': xgb_prec,
        'recall': xgb_rec,
        'f1_score': xgb_f1,
        'model': xgb_model
    }
    
    print(f"Accuracy:  {xgb_acc:.4f}")
    print(f"Precision: {xgb_prec:.4f}")
    print(f"Recall:    {xgb_rec:.4f}")
    print(f"F1-Score:  {xgb_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, xgb_pred))
    
    print("\nFeature Importance (XGBoost):")
    xgb_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(xgb_importance)
    
    # Save model
    if save_models:
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        
        import pickle
        
        xgb_path = os.path.join(config.MODEL_SAVE_DIR, 'xgboost_model.pkl')
        
        with open(xgb_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        print(f"\nXGBoost model saved to {xgb_path}")
        
        # Save volume normalization parameters for future predictions
        from feature_engineer import save_volume_normalization_params
        import yfinance as yf
        
        # Fetch original stock data with Volume column for normalization stats
        print("\nSaving volume normalization parameters...")
        ticker = yf.Ticker(config.STOCK_SYMBOL)
        df_with_volume = ticker.history(start=config.DATA_START_DATE, end=config.DATA_END_DATE)
        if not df_with_volume.empty and 'Volume' in df_with_volume.columns:
            save_volume_normalization_params(df_with_volume)
        else:
            print("Warning: Could not save volume normalization parameters.")
    
    # Summary comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison_df = pd.DataFrame({
        'Baseline': [results['baseline']['accuracy'], results['baseline']['precision'],
                     results['baseline']['recall'], results['baseline']['f1_score']],
        'XGBoost': [results['xgboost']['accuracy'], results['xgboost']['precision'],
                    results['xgboost']['recall'], results['xgboost']['f1_score']]
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
    print(comparison_df.round(4))
    
    return results


if __name__ == "__main__":
    # Test the model trainer
    from feature_engineer import create_features
    from data_fetcher import fetch_stock_data
    from news_fetcher import fetch_news_data
    from sentiment_analyzer import aggregate_daily_sentiment
    
    print("Testing model training...")
    stock_df = fetch_stock_data()
    news_df = fetch_news_data()
    sentiment_df = aggregate_daily_sentiment(news_df)
    features_df = create_features(stock_df, sentiment_df)
    
    results = train_and_evaluate_models(features_df)

