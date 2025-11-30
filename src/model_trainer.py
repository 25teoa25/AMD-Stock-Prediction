"""
Train XGBoost and Random Forest models for AMD stock prediction.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import config


def time_based_split(df, train_end_date=config.TRAIN_END_DATE):
    """
    Split data based on time (not random).
    
    Args:
        df: DataFrame with date index
        train_end_date: Last date for training set (inclusive)
    
    Returns:
        train_df, test_df: Training and testing DataFrames
    """
    train_end = pd.to_datetime(train_end_date)
    
    train_df = df[df.index <= train_end].copy()
    test_df = df[df.index > train_end].copy()
    
    print(f"Training set: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Test set: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")
    
    return train_df, test_df


def train_and_evaluate_models(features_df, save_models=True):
    """
    Train XGBoost and Random Forest models and evaluate them.
    
    Args:
        features_df: DataFrame with features and target
        save_models: Whether to save trained models to disk
    
    Returns:
        dict: Results dictionary with model performance metrics
    """
    # Time-based split
    train_df, test_df = time_based_split(features_df)
    
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Insufficient data for train/test split")
    
    # Extract features and target
    feature_cols = ['30d_return', '6m_return', '1y_return', 'volume_normalized',
                    'avg_sentiment', 'sentiment_volume']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    print(f"\nTraining set target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test set target distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
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
    
    # Train Random Forest
    print("\n" + "="*60)
    print("RANDOM FOREST")
    print("="*60)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
    
    results['random_forest'] = {
        'accuracy': rf_acc,
        'precision': rf_prec,
        'recall': rf_rec,
        'f1_score': rf_f1,
        'model': rf_model
    }
    
    print(f"Accuracy:  {rf_acc:.4f}")
    print(f"Precision: {rf_prec:.4f}")
    print(f"Recall:    {rf_rec:.4f}")
    print(f"F1-Score:  {rf_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, rf_pred))
    
    print("\nFeature Importance (Random Forest):")
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(rf_importance)
    
    # Train XGBoost
    print("\n" + "="*60)
    print("XGBOOST")
    print("="*60)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=config.RANDOM_STATE,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train, y_train)
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
    
    # Save models
    if save_models:
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        
        import pickle
        
        rf_path = os.path.join(config.MODEL_SAVE_DIR, 'random_forest_model.pkl')
        xgb_path = os.path.join(config.MODEL_SAVE_DIR, 'xgboost_model.pkl')
        
        with open(rf_path, 'wb') as f:
            pickle.dump(rf_model, f)
        print(f"\nRandom Forest model saved to {rf_path}")
        
        with open(xgb_path, 'wb') as f:
            pickle.dump(xgb_model, f)
        print(f"XGBoost model saved to {xgb_path}")
    
    # Summary comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison_df = pd.DataFrame({
        'Baseline': [results['baseline']['accuracy'], results['baseline']['precision'],
                     results['baseline']['recall'], results['baseline']['f1_score']],
        'Random Forest': [results['random_forest']['accuracy'], results['random_forest']['precision'],
                          results['random_forest']['recall'], results['random_forest']['f1_score']],
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

