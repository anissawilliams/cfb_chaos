import pandas as pd
import requests
import certifi
from tqdm import tqdm


# Load your datasets
games = pd.read_csv('../chaos_data.csv')
sentiment = pd.read_csv('../sentiment_data.csv')

# Step 1: Extract year from start_date where it exists
games['year'] = pd.to_datetime(games['start_date'], errors='coerce').dt.year


# Step 2: Fill missing years from CFBD API for games without start_date
def get_year_from_api(game_id):
    """Fetch just the year from CFBD API for a specific game"""
    url = f"https://api.collegefootballdata.com/games?id={game_id}"
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    try:
        response = requests.get(url, headers=headers, verify=certifi.where(), timeout=10)
        data = response.json()
        if data and isinstance(data, list):
            game = data[0]
            # Extract year from the season field or startDate
            if 'season' in game:
                return game['season']
            elif 'startDate' in game:
                return pd.to_datetime(game['startDate']).year
    except Exception as e:
        print(f"Error fetching game {game_id}: {e}")
    return None


# Fill missing years
missing_year_mask = games['year'].isna()
print(f"Found {missing_year_mask.sum()} games missing year data")

if missing_year_mask.sum() > 0:
    print("Fetching missing years from CFBD API...")
    for idx in tqdm(games[missing_year_mask].index):
        game_id = games.loc[idx, 'game_id']
        year = get_year_from_api(game_id)
        if year:
            games.loc[idx, 'year'] = year

    # Save updated games with years
    games.to_csv('chaos_data_with_years.csv', index=False)

# Step 3: Now merge with sentiment data
# Prepare sentiment data
sentiment_clean = sentiment[['Year', 'Week_Index', 'Conference', 'Sentiment_conf', 'Mentions_conf']].copy()
sentiment_clean.columns = ['year', 'week', 'conference', 'sentiment', 'mentions']

# Convert year to int for proper matching
games['year'] = games['year'].astype('Int64')
sentiment_clean['year'] = sentiment_clean['year'].astype('Int64')

print(f"\nGames year range: {games['year'].min()} to {games['year'].max()}")
print(f"Sentiment year range: {sentiment_clean['year'].min()} to {sentiment_clean['year'].max()}")

# Step 4: Merge HOME conference sentiment
games_merged = games.merge(
    sentiment_clean,
    left_on=['year', 'week', 'home_conference'],
    right_on=['year', 'week', 'conference'],
    how='left'
).drop('conference', axis=1)

games_merged.rename(columns={
    'sentiment': 'home_sentiment',
    'mentions': 'home_mentions'
}, inplace=True)

# Step 5: Merge AWAY conference sentiment
games_merged = games_merged.merge(
    sentiment_clean,
    left_on=['year', 'week', 'away_conference'],
    right_on=['year', 'week', 'conference'],
    how='left',
    suffixes=('', '_away')
).drop('conference', axis=1)

games_merged.rename(columns={
    'sentiment': 'away_sentiment',
    'mentions': 'away_mentions'
}, inplace=True)

# Step 6: Create average sentiment
games_merged['avg_sentiment'] = (
                                        games_merged['home_sentiment'] + games_merged['away_sentiment']
                                ) / 2

# Step 7: Save final merged dataset
games_merged.to_csv('games_with_sentiment.csv', index=False)

# Quick validation
print("\nMerge Summary:")
print(f"Total games: {len(games_merged)}")
print(f"Games with home sentiment: {games_merged['home_sentiment'].notna().sum()}")
print(f"Games with away sentiment: {games_merged['away_sentiment'].notna().sum()}")
print(
    f"Games with both sentiments: {(games_merged['home_sentiment'].notna() & games_merged['away_sentiment'].notna()).sum()}")

# Show sample
print("\nSample merged data:")
print(games_merged[['year', 'week', 'home_team', 'away_team', 'home_conference',
                    'away_conference', 'home_sentiment', 'away_sentiment',
                    'avg_sentiment', 'excitement_index']].head(10))