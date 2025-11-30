"""
Fetch AMD stock data from yfinance and calculate price features.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime
import config


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Series of closing prices
        period: Period for RSI calculation (default: 14)
    
    Returns:
        Series: RSI values normalized to 0-1 range
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100  # Normalize to 0-1


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
    
    Returns:
        tuple: (macd, signal_line, histogram) - all normalized
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    # Normalize by price to make them comparable
    price_mean = prices.rolling(window=slow).mean()
    macd_norm = macd / price_mean
    signal_norm = signal_line / price_mean
    histogram_norm = histogram / price_mean
    
    return macd_norm, signal_norm, histogram_norm


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands position.
    
    Args:
        prices: Series of closing prices
        period: Period for moving average (default: 20)
        std_dev: Number of standard deviations (default: 2)
    
    Returns:
        Series: Normalized position within bands (0-1, where 0.5 is middle)
    """
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return bb_position


def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range (ATR) normalized by price.
    
    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of closing prices
        period: Period for ATR calculation (default: 14)
    
    Returns:
        Series: ATR normalized by closing price
    """
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    atr_norm = atr / close  # Normalize by price
    return atr_norm


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
        # Normalize timezone - convert to timezone-naive
        if isinstance(df.index, pd.DatetimeIndex):
            # Remove timezone info by converting to date and back
            try:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            except:
                # If that fails, convert via date
                df.index = pd.to_datetime([d.date() if hasattr(d, 'date') else d for d in df.index])
        # Try to filter by date range, but if it fails due to timezone issues, return all data
        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            # Ensure both are timezone-naive
            if hasattr(start_dt, 'tz') and start_dt.tz is not None:
                start_dt = start_dt.tz_localize(None)
            if hasattr(end_dt, 'tz') and end_dt.tz is not None:
                end_dt = end_dt.tz_localize(None)
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        except Exception as e:
            print(f"Note: Could not filter cached data by date range: {e}")
            print("Using all cached data.")
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
    
    # Ensure we have High, Low, Open columns (required for technical indicators)
    if 'High' not in df.columns or 'Low' not in df.columns or 'Open' not in df.columns:
        raise ValueError("Missing required columns: High, Low, or Open")
    
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
    
    # Technical Indicators
    # RSI
    df['rsi'] = calculate_rsi(df['Close'], period=14)
    
    # MACD
    macd, macd_signal, macd_hist = calculate_macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_hist
    
    # Bollinger Bands
    df['bb_position'] = calculate_bollinger_bands(df['Close'], period=20, std_dev=2)
    
    # Moving Averages (as price ratios)
    df['sma_5'] = df['Close'].rolling(5).mean() / df['Close'] - 1
    df['sma_20'] = df['Close'].rolling(20).mean() / df['Close'] - 1
    df['sma_50'] = df['Close'].rolling(50).mean() / df['Close'] - 1
    
    # ATR (Average True Range)
    df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'], period=14)
    
    # Momentum features
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)
    
    # Volatility features
    df['volatility_5d'] = df['daily_return'].rolling(5).std()
    df['volatility_20d'] = df['daily_return'].rolling(20).std()
    df['volatility_ratio'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-8)  # Avoid division by zero
    
    # Price position within recent range
    df['high_20'] = df['High'].rolling(20).max()
    df['low_20'] = df['Low'].rolling(20).min()
    price_range = df['high_20'] - df['low_20']
    df['price_position'] = (df['Close'] - df['low_20']) / (price_range + 1e-8)  # Avoid division by zero
    df.drop(['high_20', 'low_20'], axis=1, inplace=True)
    
    # Lag features (1-day lags)
    df['30d_return_lag1'] = df['30d_return'].shift(1)
    df['volume_lag1'] = df['volume_normalized'].shift(1)
    df['daily_return_lag1'] = df['daily_return'].shift(1)
    
    # Change in features
    df['30d_return_change'] = df['30d_return'] - df['30d_return'].shift(1)
    df['volume_change'] = df['volume_normalized'] - df['volume_normalized'].shift(1)
    
    # Select relevant columns (keep Close and next_day_return for target calculation)
    feature_cols = ['Close', '30d_return', '6m_return', '1y_return', 'volume_normalized',
                    'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                    'sma_5', 'sma_20', 'sma_50', 'atr',
                    'momentum_5', 'momentum_10',
                    'volatility_5d', 'volatility_20d', 'volatility_ratio', 'price_position',
                    '30d_return_lag1', 'volume_lag1', 'daily_return_lag1',
                    '30d_return_change', 'volume_change',
                    'next_day_return']
    df = df[feature_cols]
    
    # Remove rows with NaN (due to rolling calculations)
    df = df.dropna()
    
    # Normalize timezone - ensure index is timezone-naive before returning
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            # Convert to date strings and back - this removes all timezone info
            df.index = pd.to_datetime([str(d.date()) for d in df.index])
        except:
            try:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
            except:
                pass
    
    # Save to cache
    if cache:
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        df.to_csv(cache_file)
        print(f"Stock data cached to {cache_file}")
    
    print(f"Fetched {len(df)} trading days of data")
    return df


def fetch_latest_stock_data(symbol=config.STOCK_SYMBOL, 
                           start_date=None,
                           use_training_normalization=True,
                           cache=True):
    """
    Fetch latest stock data up to today and calculate features.
    Uses volume normalization from training data if available.
    
    Args:
        symbol: Stock symbol (default: AMD)
        start_date: Start date in 'YYYY-MM-DD' format (default: 1 year ago)
        use_training_normalization: If True, use normalization params from training data
        cache: Whether to use cached data if available
    
    Returns:
        DataFrame with date index and features, including the most recent trading day
    """
    from datetime import datetime, timedelta
    
    # Default to 1 year ago to ensure we have enough history for rolling calculations
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    
    # Fetch data up to today (no end_date limit)
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Fetching latest stock data for {symbol} from {start_date} to today...")
    
    # Fetch data from yfinance
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date)
    
    if df.empty:
        raise ValueError(f"No data retrieved for {symbol}")
    
    # Reset index to make Date a column, then set it as index with proper name
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date'}, inplace=True)
    df.set_index('date', inplace=True)
    
    # Ensure we have High, Low, Open columns (required for technical indicators)
    if 'High' not in df.columns or 'Low' not in df.columns or 'Open' not in df.columns:
        raise ValueError("Missing required columns: High, Low, or Open")
    
    # Calculate daily returns
    df['daily_return'] = df['Close'].pct_change()
    
    # Calculate next day return (for target variable)
    df['next_day_return'] = df['daily_return'].shift(-1)
    
    # Calculate rolling returns
    df['30d_return'] = df['Close'].pct_change(periods=config.RETURN_30D)
    df['6m_return'] = df['Close'].pct_change(periods=config.RETURN_6M)
    df['1y_return'] = df['Close'].pct_change(periods=config.RETURN_1Y)
    
    # Handle volume normalization
    if use_training_normalization:
        # Try to load normalization parameters from training data
        norm_params_file = os.path.join(config.PROCESSED_DATA_DIR, "volume_normalization_params.json")
        if os.path.exists(norm_params_file):
            import json
            with open(norm_params_file, 'r') as f:
                norm_params = json.load(f)
            volume_mean = norm_params['mean']
            volume_std = norm_params['std']
            print(f"Using training data normalization (mean={volume_mean:.2f}, std={volume_std:.2f})")
        else:
            # Use current data statistics (fallback)
            volume_mean = df['Volume'].mean()
            volume_std = df['Volume'].std()
            print(f"Using current data normalization (mean={volume_mean:.2f}, std={volume_std:.2f})")
    else:
        # Use current data statistics
        volume_mean = df['Volume'].mean()
        volume_std = df['Volume'].std()
    
    if volume_std > 0:
        df['volume_normalized'] = (df['Volume'] - volume_mean) / volume_std
    else:
        df['volume_normalized'] = 0
    
    # Technical Indicators
    # RSI
    df['rsi'] = calculate_rsi(df['Close'], period=14)
    
    # MACD
    macd, macd_signal, macd_hist = calculate_macd(df['Close'], fast=12, slow=26, signal=9)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_hist
    
    # Bollinger Bands
    df['bb_position'] = calculate_bollinger_bands(df['Close'], period=20, std_dev=2)
    
    # Moving Averages (as price ratios)
    df['sma_5'] = df['Close'].rolling(5).mean() / df['Close'] - 1
    df['sma_20'] = df['Close'].rolling(20).mean() / df['Close'] - 1
    df['sma_50'] = df['Close'].rolling(50).mean() / df['Close'] - 1
    
    # ATR (Average True Range)
    df['atr'] = calculate_atr(df['High'], df['Low'], df['Close'], period=14)
    
    # Momentum features
    df['momentum_5'] = df['Close'].pct_change(5)
    df['momentum_10'] = df['Close'].pct_change(10)
    
    # Volatility features
    df['volatility_5d'] = df['daily_return'].rolling(5).std()
    df['volatility_20d'] = df['daily_return'].rolling(20).std()
    df['volatility_ratio'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-8)  # Avoid division by zero
    
    # Price position within recent range
    df['high_20'] = df['High'].rolling(20).max()
    df['low_20'] = df['Low'].rolling(20).min()
    price_range = df['high_20'] - df['low_20']
    df['price_position'] = (df['Close'] - df['low_20']) / (price_range + 1e-8)  # Avoid division by zero
    df.drop(['high_20', 'low_20'], axis=1, inplace=True)
    
    # Lag features (1-day lags)
    df['30d_return_lag1'] = df['30d_return'].shift(1)
    df['volume_lag1'] = df['volume_normalized'].shift(1)
    df['daily_return_lag1'] = df['daily_return'].shift(1)
    
    # Change in features
    df['30d_return_change'] = df['30d_return'] - df['30d_return'].shift(1)
    df['volume_change'] = df['volume_normalized'] - df['volume_normalized'].shift(1)
    
    # Select relevant columns (keep Close and next_day_return for target calculation)
    feature_cols = ['Close', '30d_return', '6m_return', '1y_return', 'volume_normalized',
                    'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                    'sma_5', 'sma_20', 'sma_50', 'atr',
                    'momentum_5', 'momentum_10',
                    'volatility_5d', 'volatility_20d', 'volatility_ratio', 'price_position',
                    '30d_return_lag1', 'volume_lag1', 'daily_return_lag1',
                    '30d_return_change', 'volume_change',
                    'next_day_return']
    df = df[feature_cols]
    
    # Remove rows with NaN (due to rolling calculations)
    df = df.dropna()
    
    print(f"Fetched {len(df)} trading days of data (latest: {df.index[-1].strftime('%Y-%m-%d')})")
    return df


def get_latest_trading_day_features(symbol=config.STOCK_SYMBOL):
    """
    Get features for the most recent trading day.
    
    Args:
        symbol: Stock symbol (default: AMD)
    
    Returns:
        dict: Dictionary with feature values for the latest trading day
    """
    df = fetch_latest_stock_data(symbol, use_training_normalization=True, cache=False)
    
    if df.empty:
        raise ValueError("No stock data available")
    
    latest = df.iloc[-1]
    
    features = {
        'date': df.index[-1],
        '30d_return': latest['30d_return'],
        '6m_return': latest['6m_return'],
        '1y_return': latest['1y_return'],
        'volume_normalized': latest['volume_normalized'],
        'rsi': latest.get('rsi', 0.5),
        'macd': latest.get('macd', 0.0),
        'macd_signal': latest.get('macd_signal', 0.0),
        'macd_histogram': latest.get('macd_histogram', 0.0),
        'bb_position': latest.get('bb_position', 0.5),
        'sma_5': latest.get('sma_5', 0.0),
        'sma_20': latest.get('sma_20', 0.0),
        'sma_50': latest.get('sma_50', 0.0),
        'atr': latest.get('atr', 0.0),
        'momentum_5': latest.get('momentum_5', 0.0),
        'momentum_10': latest.get('momentum_10', 0.0),
        'volatility_5d': latest.get('volatility_5d', 0.0),
        'volatility_20d': latest.get('volatility_20d', 0.0),
        'volatility_ratio': latest.get('volatility_ratio', 1.0),
        'price_position': latest.get('price_position', 0.5),
        '30d_return_lag1': latest.get('30d_return_lag1', 0.0),
        'volume_lag1': latest.get('volume_lag1', 0.0),
        'daily_return_lag1': latest.get('daily_return_lag1', 0.0),
        '30d_return_change': latest.get('30d_return_change', 0.0),
        'volume_change': latest.get('volume_change', 0.0),
        'close_price': latest['Close']
    }
    
    return features


if __name__ == "__main__":
    # Test the data fetcher
    df = fetch_stock_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData shape:", df.shape)
    print("\nColumn info:")
    print(df.info())

