import pandas as pd

games = pd.read_csv('../chaos_data.csv')
sentiment = pd.read_csv('../sentiment_data.csv')
#missing_ids = games[games['year'].isna()]['game_id'].tolist()
#print("Total missing:", len(missing_ids))
#print("First 25 missing IDs:", missing_ids[:25])

# print(sorted(games['home_conference'].unique()))
# print(sorted(games['away_conference'].unique()))
# print(sorted(sentiment['Conference'].unique()))

cfbd_confs = set(games['home_conference'].dropna().astype(str).unique()) | \
             set(games['away_conference'].dropna().astype(str).unique())

sentiment_confs = set(sentiment['Conference'].dropna().astype(str).unique())

print("CFBD conferences:", sorted(cfbd_confs))
print("Sentiment df conferences:", sorted(sentiment_confs))

