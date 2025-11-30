# AMD Stock Price Prediction

A machine learning project to predict whether AMD's stock price will increase by more than 0.5% the next day, using historical price data and news sentiment analysis.

## Features

- **Price Features**: 30-day, 6-month, and 1-year historical returns, plus normalized trading volume
- **News Sentiment**: Daily sentiment analysis of top 5 AMD news articles using OpenAI API
- **Models**: XGBoost Classifier (primary) and Random Forest (baseline)
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

# Make predictions
python main.py --mode predict
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
└── src/
    ├── data_fetcher.py     # Stock data fetching
    ├── news_fetcher.py     # News data fetching
    ├── sentiment_analyzer.py # Sentiment analysis
    ├── feature_engineer.py # Feature engineering
    ├── model_trainer.py    # Model training
    └── predictor.py        # Prediction script
```

## Data Timeframe

- **Training**: 2020-01-01 to 2023-06-30
- **Testing**: 2023-07-01 to 2023-12-31

## Model Evaluation

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Feature importance analysis
- Comparison against baseline (always predict 0)

## Notes

- The pipeline handles API rate limits and missing data gracefully
- Data is cached to avoid redundant API calls
- No look-ahead bias: features from day t, target from day t+1
- Strict time-based train/test split (no random shuffling)

