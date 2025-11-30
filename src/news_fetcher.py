"""
Fetch AMD news articles from NewsAPI.
"""

import pandas as pd
import os
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from dotenv import load_dotenv
import time
import config

# Load environment variables
load_dotenv()


def fetch_news_data(start_date=config.DATA_START_DATE, 
                   end_date=config.DATA_END_DATE,
                   cache=True):
    """
    Fetch top AMD news articles for each day from NewsAPI.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cache: Whether to use cached data if available
    
    Returns:
        DataFrame with columns: date, article_text, title, url, publishedAt
    """
    cache_file = os.path.join(config.RAW_DATA_DIR, "amd_news_data.csv")
    
    # Check cache
    if cache and os.path.exists(cache_file):
        print(f"Loading cached news data from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Ensure date range matches
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        if len(df) > 0:
            return df
    
    api_key = os.getenv('NEWSAPI_KEY')
    if not api_key:
        raise ValueError("NEWSAPI_KEY not found in environment variables. Please set it in .env file.")
    
    newsapi = NewsApiClient(api_key=api_key)
    
    print(f"Fetching news data from {start_date} to {end_date}...")
    
    # Convert dates to datetime
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_articles = []
    current_date = start
    
    # Fetch articles day by day to respect rate limits
    while current_date <= end:
        date_str = current_date.strftime('%Y-%m-%d')
        next_date = current_date + timedelta(days=1)
        next_date_str = next_date.strftime('%Y-%m-%d')
        
        try:
            # Search for articles about AMD
            articles = newsapi.get_everything(
                q=config.NEWS_QUERY,
                from_param=date_str,
                to=next_date_str,
                language='en',
                sort_by='relevancy',
                page_size=config.MAX_NEWS_ARTICLES_PER_DAY
            )
            
            if articles['status'] == 'ok' and articles['articles']:
                for article in articles['articles'][:config.MAX_NEWS_ARTICLES_PER_DAY]:
                    # Combine title and description for full text
                    title = article.get('title', '')
                    description = article.get('description', '')
                    content = article.get('content', '')
                    
                    # Use description if available, otherwise content, otherwise title
                    article_text = description or content or title
                    
                    all_articles.append({
                        'date': date_str,
                        'title': title,
                        'article_text': article_text,
                        'url': article.get('url', ''),
                        'publishedAt': article.get('publishedAt', '')
                    })
            
            # Rate limiting: NewsAPI free tier allows 100 requests per day
            # Add small delay to be safe
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error fetching news for {date_str}: {e}")
            # Continue with next date
        
        current_date = next_date
        
        # Progress indicator
        if (current_date - start).days % 30 == 0:
            print(f"Processed {current_date.strftime('%Y-%m-%d')}...")
    
    if not all_articles:
        print("Warning: No news articles found. Creating empty DataFrame.")
        df = pd.DataFrame(columns=['date', 'title', 'article_text', 'url', 'publishedAt'])
    else:
        df = pd.DataFrame(all_articles)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # Save to cache
    if cache:
        os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
        df.to_csv(cache_file)
        print(f"News data cached to {cache_file}")
    
    print(f"Fetched {len(df)} news articles")
    return df


if __name__ == "__main__":
    # Test the news fetcher
    df = fetch_news_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData shape:", df.shape)
    if len(df) > 0:
        print("\nSample article text:")
        print(df.iloc[0]['article_text'][:200] if len(df.iloc[0]['article_text']) > 200 else df.iloc[0]['article_text'])

