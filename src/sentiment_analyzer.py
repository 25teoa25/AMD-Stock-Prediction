"""
Analyze news article sentiment using OpenAI API.
"""

import pandas as pd
import numpy as np
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import time
import config

# Load environment variables
load_dotenv()


def extract_sentiment_score(response_text):
    """
    Extract numeric sentiment score from OpenAI response.
    
    Args:
        response_text: Text response from OpenAI
    
    Returns:
        float: Sentiment score between -1 and 1, or 0 if extraction fails
    """
    # Try to find a number between -1 and 1
    # Look for patterns like: -0.5, 0.75, 1.0, etc.
    pattern = r'-?\d+\.?\d*'
    numbers = re.findall(pattern, response_text)
    
    for num_str in numbers:
        try:
            num = float(num_str)
            # Clamp to [-1, 1] range
            if -1 <= num <= 1:
                return num
        except ValueError:
            continue
    
    # If no valid number found, try to infer from text
    response_lower = response_text.lower()
    if any(word in response_lower for word in ['positive', 'bullish', 'good', 'up']):
        return 0.5
    elif any(word in response_lower for word in ['negative', 'bearish', 'bad', 'down']):
        return -0.5
    else:
        return 0.0


def analyze_article_sentiment(article_text, client):
    """
    Analyze sentiment of a single article using OpenAI API.
    
    Args:
        article_text: Text content of the article
        client: OpenAI client instance
    
    Returns:
        float: Sentiment score between -1 and 1
    """
    if not article_text or len(article_text.strip()) == 0:
        return 0.0
    
    prompt = config.SENTIMENT_PROMPT_TEMPLATE.format(article_text=article_text[:2000])  # Limit text length
    
    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyzer. Output only a numeric score."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        sentiment = extract_sentiment_score(response_text)
        
        return sentiment
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0.0


def aggregate_daily_sentiment(news_df, cache=True):
    """
    Analyze sentiment for all articles and aggregate by day.
    
    Args:
        news_df: DataFrame with news articles (from news_fetcher)
        cache: Whether to use cached sentiment data if available
    
    Returns:
        DataFrame with date index and columns: avg_sentiment, sentiment_volume
    """
    cache_file = os.path.join(config.PROCESSED_DATA_DIR, "daily_sentiment.csv")
    
    # Check cache
    if cache and os.path.exists(cache_file):
        print(f"Loading cached sentiment data from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    
    client = OpenAI(api_key=api_key)
    
    if news_df.empty:
        print("Warning: No news articles to analyze. Returning empty sentiment DataFrame.")
        return pd.DataFrame(columns=['avg_sentiment', 'sentiment_volume'], index=pd.DatetimeIndex([]))
    
    print("Analyzing sentiment for news articles...")
    
    # Analyze sentiment for each article
    news_df = news_df.copy()
    news_df['sentiment'] = 0.0
    
    total_articles = len(news_df)
    for idx, row in news_df.iterrows():
        article_text = str(row.get('article_text', ''))
        sentiment = analyze_article_sentiment(article_text, client)
        news_df.at[idx, 'sentiment'] = sentiment
        
        # Progress indicator
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{total_articles} articles...")
        
        # Rate limiting: Add small delay to avoid hitting API limits
        time.sleep(0.1)
    
    # Aggregate by date
    daily_sentiment = news_df.groupby(news_df.index.date).agg({
        'sentiment': ['mean', 'count']
    })
    
    # Flatten column names
    daily_sentiment.columns = ['avg_sentiment', 'sentiment_volume']
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    
    # Save to cache
    if cache:
        os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
        daily_sentiment.to_csv(cache_file)
        print(f"Sentiment data cached to {cache_file}")
    
    print(f"Analyzed sentiment for {len(daily_sentiment)} days")
    return daily_sentiment


if __name__ == "__main__":
    # Test the sentiment analyzer
    from news_fetcher import fetch_news_data
    
    # Fetch a small sample of news
    news_df = fetch_news_data(start_date="2023-01-01", end_date="2023-01-05")
    if not news_df.empty:
        sentiment_df = aggregate_daily_sentiment(news_df)
        print("\nDaily sentiment:")
        print(sentiment_df.head())

