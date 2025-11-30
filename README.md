# AMD Stock Price Prediction

A machine learning project to predict whether AMD's stock price will increase by more than 0.5% the next day, using historical price data and news sentiment analysis.

## Features

- **Price Features**: 30-day, 6-month, and 1-year historical returns, plus normalized trading volume
- **News Sentiment**: Daily sentiment analysis of top 5 AMD news articles using OpenAI API
- **Model**: XGBoost Classifier
- **Time-series Integrity**: Strict chronological train/test split with no look-ahead bias

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`
   - Add your NewsAPI key: `NEWSAPI_KEY=your_key_here`

3. **Get API keys**:
   - OpenAI: https://platform.openai.com/api-keys
   - NewsAPI: https://newsapi.org/register

## Usage

### Run the complete pipeline:
```bash
python main.py --mode all
```

### Run individual steps:
```bash
# Fetch data only
python main.py --mode fetch

# Train model only
python main.py --mode train

# Make predictions on test data
python main.py --mode predict

# Predict next market day (uses current data)
python main.py --mode predict-next
```

### Predict Next Market Day:
```bash
# Option 1: Using main.py
python main.py --mode predict-next

# Option 2: Using dedicated script
python predict_next_day.py
```

## Project Structure

```
AMD Stock Prediction/
├── requirements.txt          # Python dependencies
├── .env                      # API keys (not in git)
├── .env.example             # API key template
├── config.py                # Configuration constants
├── main.py                  # Main orchestration script
├── data/
│   ├── raw/                # Raw fetched data
│   └── processed/          # Processed features
├── models/                 # Saved trained models
├── predict_next_day.py     # Next market day prediction script
└── src/
    ├── data_fetcher.py     # Stock data fetching
    ├── news_fetcher.py     # News data fetching
    ├── sentiment_analyzer.py # Sentiment analysis
    ├── feature_engineer.py # Feature engineering
    ├── model_trainer.py    # Model training
    └── predictor.py        # Prediction script
```

## Data Timeframe

- **Training**: 2020-01-01 to 2025-05-31
- **Testing**: 2025-06-01 to 2025-11-29
- **Data Range**: 2020-01-01 to 2025-11-29 (up to today)

## Model Evaluation

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Feature importance analysis
- Comparison against baseline (always predict 0)

## Predicting Next Market Day

After training the model, you can predict the next trading day's movement:

```bash
python main.py --mode predict-next
```

This will:
1. Fetch the latest stock data (up to today)
2. Get yesterday's news articles and analyze sentiment (due to NewsAPI 24-hour delay)
3. Calculate features for the most recent trading day
4. Make a prediction using the trained XGBoost model
5. Display the prediction and confidence level

**Notes**: 
- The model uses volume normalization parameters from the training data to ensure consistency
- Due to NewsAPI's 24-hour delay, predictions use yesterday's news (most recent available)

## NewsAPI Constraints (Free Tier)

The model respects NewsAPI free tier limitations:

- **100 articles per day limit**: Maximum 100 articles can be fetched per day
- **24-hour delay**: Articles are delayed by 24 hours (today's articles available tomorrow)
- **1-month historical limit**: Can only search articles up to 30 days old

### Impact on Model:

- **Historical training**: For dates older than 30 days, news data will be unavailable and sentiment defaults to 0.0
- **Recent data**: News is only fetched for the last 30 days
- **Predictions**: Uses yesterday's news (due to 24-hour delay) to predict next trading day
- **Rate limiting**: The system tracks daily API usage and stops when approaching the 100 articles/day limit

The model is designed to work gracefully with missing news data - stock features alone can still provide predictions.

## Notes

- The pipeline handles API rate limits and missing data gracefully
- Data is cached to avoid redundant API calls
- No look-ahead bias: features from day t, target from day t+1
- Strict time-based train/test split (no random shuffling)
- Volume normalization parameters are saved during training for consistent predictions
- News data is limited by NewsAPI constraints (see above)

