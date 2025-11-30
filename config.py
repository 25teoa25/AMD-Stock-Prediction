"""
Configuration constants for AMD Stock Prediction project
"""

# Date ranges
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE = "2023-06-30"
TEST_START_DATE = "2023-07-01"
TEST_END_DATE = "2023-12-31"
DATA_START_DATE = "2020-01-01"
DATA_END_DATE = "2023-12-31"

# Stock symbol
STOCK_SYMBOL = "AMD"

# Return calculation windows (trading days)
RETURN_30D = 30
RETURN_6M = 126  # Approximately 6 months
RETURN_1Y = 252  # Approximately 1 year

# Target threshold
TARGET_THRESHOLD = 0.005  # 0.5% return threshold

# News API settings
MAX_NEWS_ARTICLES_PER_DAY = 5
NEWS_QUERY = "AMD OR Advanced Micro Devices"

# Sentiment analysis prompt
SENTIMENT_PROMPT_TEMPLATE = (
    "Analyze this financial news about AMD for market impact. "
    "Output only a single sentiment score between -1 (very negative) and +1 (very positive), "
    "where 0 is neutral: {article_text}"
)

# Model settings
RANDOM_STATE = 42
MODEL_SAVE_DIR = "models"

# Data paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

