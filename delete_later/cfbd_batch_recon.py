import pandas as pd
import requests
import certifi
from tqdm import tqdm

CFBD_API_KEY = "mvtFD9Rg26SQFMDNkQL30s6nmfpDp8Vg7jgQRo0ec0frsxF/qMkbQ6vSpAWI/XlU"

# ------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------
games = pd.read_csv('../chaos_data.csv')
sentiment = pd.read_csv('../sentiment_data.csv')

# ------------------------------------------------------
# 2. Extract year from start_date (where available)
# ------------------------------------------------------
games['year'] = pd.to_datetime(games['start_date'], errors='coerce').dt.year

# Identify missing years
missing_games = games[games['year'].isna()]
missing_ids = missing_games['game_id'].tolist()
print(f"Games missing year: {len(missing_ids)}")

# ------------------------------------------------------
# 3. Batch Fetch Seasons (REGULAR + POSTSEASON)
# ------------------------------------------------------
if len(missing_ids) > 0:
    print("\nBuilding game_id → year lookup table (regular + postseason)...")

    # Determine reasonable year range from sentiment dataset
    min_year = int(sentiment['Year'].min())
    max_year = int(sentiment['Year'].max())
    year_range = list(range(min_year, max_year + 1))

    print(f"Fetching seasons {min_year}–{max_year}...")

    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    season_lookup = {}  # game_id → season

    for yr in tqdm(year_range):
        for stype in ["regular", "postseason"]:
            url = f"https://api.collegefootballdata.com/games?year={yr}&seasonType={stype}"

            try:
                r = requests.get(
                    url,
                    headers=headers,
                    verify=certifi.where(),
                    timeout=30
                )
                r.raise_for_status()
                data = r.json()

                for g in data:
                    season_lookup[g["id"]] = g["season"]

            except Exception as e:
                print(f"Failed for year {yr} seasonType={stype}: {e}")

    # ------------------------------------------------------
    # 4. Fill Missing Year Values
    # ------------------------------------------------------
    print("\nFilling missing years with lookup table...")
    filled = 0

    for idx in missing_games.index:
        gid = games.loc[idx, 'game_id']
        if gid in season_lookup:
            games.loc[idx, 'year'] = season_lookup[gid]
            filled += 1

    print(f"Filled {filled} of {len(missing_ids)} missing years.")

# Save intermediate (optional)
games.to_csv("chaos_data_with_years.csv", index=False)

# ------------------------------------------------------
# 5. Normalize Conferences
# ------------------------------------------------------
conference_map = {
    'ACC': 'ACC',
    'Big Ten': 'Big10',
    'Big 12': 'Big12',
    'Pac-12': 'Pac12',
    'SEC': 'SEC',
    # The rest will remain unmapped (won’t match sentiment)
}

games['home_conf_norm'] = games['home_conference'].map(conference_map)
games['away_conf_norm'] = games['away_conference'].map(conference_map)

# ------------------------------------------------------
# 6. Prepare Sentiment Data
# ------------------------------------------------------
sentiment_clean = sentiment[['Year', 'Week_Index', 'Conference',
                             'Sentiment_conf', 'Mentions_conf']].copy()

sentiment_clean.columns = ['year', 'week', 'conference', 'sentiment', 'mentions']

# Ensure types match
games['year'] = games['year'].astype('Int64')
sentiment_clean['year'] = sentiment_clean['year'].astype('Int64')

print(f"\nGames year range: {games['year'].min()}–{games['year'].max()}")
print(f"Sentiment year range: {sentiment_clean['year'].min()}–{sentiment_clean['year'].max()}")

# ------------------------------------------------------
# 7. Merge HOME sentiment
# ------------------------------------------------------
games_merged = games.merge(
    sentiment_clean,
    left_on=['year', 'week', 'home_conf_norm'],
    right_on=['year', 'week', 'conference'],
    how='left'
).drop('conference', axis=1).rename(columns={
    'sentiment': 'home_sentiment',
    'mentions': 'home_mentions'
})

# ------------------------------------------------------
# 8. Merge AWAY sentiment
# ------------------------------------------------------
games_merged = games_merged.merge(
    sentiment_clean,
    left_on=['year', 'week', 'away_conf_norm'],
    right_on=['year', 'week', 'conference'],
    how='left'
).drop('conference', axis=1).rename(columns={
    'sentiment': 'away_sentiment',
    'mentions': 'away_mentions'
})

# ------------------------------------------------------
# 9. Average Sentiment
# ------------------------------------------------------
games_merged['avg_sentiment'] = (
    games_merged['home_sentiment'] + games_merged['away_sentiment']
) / 2

# ------------------------------------------------------
# 10. Save Final Dataset
# ------------------------------------------------------
games_merged.to_csv('games_with_sentiment.csv', index=False)

# ------------------------------------------------------
# 11. Validation Output
# ------------------------------------------------------
print("\nMerge Summary:")
print(f"Total games: {len(games_merged)}")
print(f"Games w/ home sentiment: {games_merged['home_sentiment'].notna().sum()}")
print(f"Games w/ away sentiment: {games_merged['away_sentiment'].notna().sum()}")
print(f"Games w/ both: {(games_merged['home_sentiment'].notna() & games_merged['away_sentiment'].notna()).sum()}")

print("\nSample merged data:")
print(games_merged[['year', 'week', 'home_team', 'away_team',
                    'home_conference', 'away_conference',
                    'home_sentiment', 'away_sentiment',
                    'avg_sentiment', 'excitement_index']].head(10))
