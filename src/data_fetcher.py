"""
Fetch AMD stock data from yfinance and calculate price features.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime
import config


def fetch_stock_data(symbol=config.STOCK_SYMBOL, 
                     start_date=config.DATA_START_DATE, 
                     end_date=config.DATA_END_DATE,
                     cache=True):
    """
    Fetch stock data from yfinance and calculate features.
    
    Args:
        symbol: Stock symbol (default: AMD)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cache: Whether to use cached data if available
    
    Returns:
        DataFrame with date index and features: 
        - Close price
        - 30d_return
        - 6m_return
        - 1y_return
        - volume_normalized
        - next_day_return (for target calculation)
    """
    cache_file = os.path.join(config.RAW_DATA_DIR, f"{symbol}_stock_data.csv")
    
    # Check cache
    if cache and os.path.exists(cache_file):
        print(f"Loading cached stock data from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Ensure date range matches
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        if len(df) > 0:
            return df
    
    print(f"Fetching stock data for {symbol} from {start_date} to {end_date}...")
    
    # Fetch data from yfinance
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data retrieved for {symbol}")
    
    # Reset index to make Date a column, then set it as index with proper name
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date'}, inplace=True)
    df.set_index('date', inplace=True)
    
    # Calculate daily returns
    df['daily_return'] = df['Close'].pct_change()
    
    # Calculate next day return (for target variable)
    df['next_day_return'] = df['daily_return'].shift(-1)
    
    # Calculate rolling returns
    df['30d_return'] = df['Close'].pct_change(periods=config.RETURN_30D)
    df['6m_return'] = df['Close'].pct_change(periods=config.RETURN_6M)
    df['1y_return'] = df['Close'].pct_change(periods=config.RETURN_1Y)
    
    # Normalize volume (z-score normalization)
    volume_mean = df['Volume'].mean()
    volume_std = df['Volume'].std()
    if volume_std > 0:
        df['volume_normalized'] = (df['Volume'] - volume_mean) / volume_std
    else:
        df['volume_normalized'] = 0
    
    # Select relevant columns
    feature_cols = ['Close', '30d_return', '6m_return', '1y_return', 
                    'volume_normalized', 'next_day_return']
    df = df[feature_cols]
    
    # Remove rows with NaN (due to rolling calculations)
    df = df.dropna()
    
    # Save to cache
    if cache:
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        df.to_csv(cache_file)
        print(f"Stock data cached to {cache_file}")
    
    print(f"Fetched {len(df)} trading days of data")
    return df


if __name__ == "__main__":
    # Test the data fetcher
    df = fetch_stock_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData shape:", df.shape)
    print("\nColumn info:")
    print(df.info())

