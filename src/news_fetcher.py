"""
Fetch AMD news articles from NewsAPI.
Respects NewsAPI free tier constraints:
- 100 articles per day limit
- 24-hour delay (today's articles available tomorrow)
- 1-month historical limit (can only search up to 30 days old)
"""

import pandas as pd
import os
import json
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from dotenv import load_dotenv
import time
import config

# Load environment variables
load_dotenv()


def get_api_call_tracker():
    """
    Get or create API call tracker file.
    
    Returns:
        dict: Dictionary with today's date and article count
    """
    tracker_file = os.path.join(config.RAW_DATA_DIR, "newsapi_tracker.json")
    today = datetime.now().strftime('%Y-%m-%d')
    
    if os.path.exists(tracker_file):
        try:
            with open(tracker_file, 'r') as f:
                tracker = json.load(f)
            # Reset if it's a new day
            if tracker.get('date') != today:
                tracker = {'date': today, 'articles_fetched': 0}
        except:
            tracker = {'date': today, 'articles_fetched': 0}
    else:
        tracker = {'date': today, 'articles_fetched': 0}
    
    return tracker, tracker_file


def update_api_call_tracker(tracker, tracker_file, articles_count):
    """
    Update API call tracker with new article count.
    
    Args:
        tracker: Current tracker dictionary
        tracker_file: Path to tracker file
        articles_count: Number of articles fetched in this call
    """
    tracker['articles_fetched'] += articles_count
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    with open(tracker_file, 'w') as f:
        json.dump(tracker, f)


def validate_date_range(start_date, end_date):
    """
    Validate date range against NewsAPI constraints.
    
    Args:
        start_date: Start date string
        end_date: End date string
    
    Returns:
        tuple: (is_valid, error_message, adjusted_start, adjusted_end)
    """
    today = datetime.now()
    max_historical_date = today - timedelta(days=config.NEWSAPI_HISTORICAL_LIMIT_DAYS)
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Check if dates are too old
    if start < max_historical_date:
        print(f"Warning: Start date {start_date} is older than {config.NEWSAPI_HISTORICAL_LIMIT_DAYS} days.")
        print(f"NewsAPI only allows searching up to {max_historical_date.strftime('%Y-%m-%d')}.")
        print(f"Adjusting start date to {max_historical_date.strftime('%Y-%m-%d')}.")
        start = max_historical_date
    
    if end > today:
        print(f"Warning: End date {end_date} is in the future. Adjusting to today.")
        end = today
    
    # Apply 24-hour delay: can't fetch today's articles, only up to yesterday
    yesterday = today - timedelta(days=1)
    if end > yesterday:
        print(f"Note: NewsAPI has 24-hour delay. Adjusting end date to {yesterday.strftime('%Y-%m-%d')}.")
        end = yesterday
    
    adjusted_start = start.strftime('%Y-%m-%d')
    adjusted_end = end.strftime('%Y-%m-%d')
    
    if start > end:
        return False, "Adjusted start date is after end date. No news data available for this range.", None, None
    
    return True, None, adjusted_start, adjusted_end


def fetch_news_data(start_date=config.DATA_START_DATE, 
                   end_date=config.DATA_END_DATE,
                   cache=True):
    """
    Fetch top AMD news articles for each day from NewsAPI.
    Respects NewsAPI constraints: 100 articles/day limit, 30-day historical limit, 24-hour delay.
    
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
    
    # Validate and adjust date range
    is_valid, error_msg, adjusted_start, adjusted_end = validate_date_range(start_date, end_date)
    if not is_valid:
        print(f"Warning: {error_msg}")
        print("Returning empty DataFrame. News data will be set to 0 sentiment.")
        return pd.DataFrame(columns=['date', 'title', 'article_text', 'url', 'publishedAt'])
    
    if adjusted_start != start_date or adjusted_end != end_date:
        print(f"Using adjusted date range: {adjusted_start} to {adjusted_end}")
        start_date = adjusted_start
        end_date = adjusted_end
    
    api_key = os.getenv('NEWSAPI_KEY')
    if not api_key:
        raise ValueError("NEWSAPI_KEY not found in environment variables. Please set it in .env file.")
    
    newsapi = NewsApiClient(api_key=api_key)
    
    print(f"Fetching news data from {start_date} to {end_date}...")
    print(f"Note: NewsAPI free tier allows max {config.NEWSAPI_MAX_ARTICLES_PER_DAY} articles per day.")
    
    # Get API call tracker
    tracker, tracker_file = get_api_call_tracker()
    articles_remaining = config.NEWSAPI_MAX_ARTICLES_PER_DAY - tracker['articles_fetched']
    
    if articles_remaining <= 0:
        print(f"Warning: Daily limit of {config.NEWSAPI_MAX_ARTICLES_PER_DAY} articles reached.")
        print("Returning empty DataFrame. News data will be set to 0 sentiment.")
        return pd.DataFrame(columns=['date', 'title', 'article_text', 'url', 'publishedAt'])
    
    print(f"Articles remaining today: {articles_remaining}")
    
    # Convert dates to datetime
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_articles = []
    current_date = start
    total_fetched = 0
    
    # Fetch articles day by day to respect rate limits
    while current_date <= end and total_fetched < articles_remaining:
        date_str = current_date.strftime('%Y-%m-%d')
        next_date = current_date + timedelta(days=1)
        next_date_str = next_date.strftime('%Y-%m-%d')
        
        # Check if we've reached the daily limit
        if total_fetched >= articles_remaining:
            print(f"\nReached daily limit of {config.NEWSAPI_MAX_ARTICLES_PER_DAY} articles.")
            break
        
        # Calculate how many articles we can fetch for this day
        articles_to_fetch = min(config.MAX_NEWS_ARTICLES_PER_DAY, articles_remaining - total_fetched)
        
        try:
            # Search for articles about AMD
            articles = newsapi.get_everything(
                q=config.NEWS_QUERY,
                from_param=date_str,
                to=next_date_str,
                language='en',
                sort_by='relevancy',
                page_size=articles_to_fetch
            )
            
            if articles['status'] == 'ok' and articles['articles']:
                day_articles = 0
                for article in articles['articles'][:articles_to_fetch]:
                    if total_fetched >= articles_remaining:
                        break
                    
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
                    day_articles += 1
                    total_fetched += 1
                
                if day_articles > 0:
                    print(f"Fetched {day_articles} articles for {date_str} (Total: {total_fetched}/{articles_remaining})")
            
            # Rate limiting: Add small delay to be safe
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error fetching news for {date_str}: {e}")
            # Continue with next date
        
        current_date = next_date
    
    # Update tracker
    if total_fetched > 0:
        update_api_call_tracker(tracker, tracker_file, total_fetched)
        print(f"\nTotal articles fetched in this session: {total_fetched}")
        print(f"Daily total: {tracker['articles_fetched']}/{config.NEWSAPI_MAX_ARTICLES_PER_DAY}")
    
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
    
    print(f"Fetched {len(df)} news articles total")
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

