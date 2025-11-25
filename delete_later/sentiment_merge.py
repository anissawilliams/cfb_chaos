import pandas as pd

# Load datasets
#sentiment_df = pd.read_excel("pone.0325840.s002.xlsx", sheet_name="Time Series")
#sentiment_df.to_csv("sentiment_data.csv", index=False)
sentiment_df = pd.read_csv("../sentiment_data.csv")
games_df = pd.read_csv("../chaos_data.csv")

print(sentiment_df.shape)
print(games_df.shape)


conf_map = {
    "Big Ten": "Big10",
    "Big 12": "Big12",
    "Pac-12": "Pac12",
    "SEC": "SEC",
    "ACC": "ACC"
}
games_df['home_conference'] = games_df['home_conference'].map(conf_map)
games_df['away_conference'] = games_df['away_conference'].map(conf_map)

games_df['week'] = pd.to_numeric(games_df['week'], errors='coerce').fillna(0).astype(int)
sentiment_df['Week_Index'] = pd.to_numeric(sentiment_df['Week_Index'], errors='coerce').astype(int)


# Step 0: Define chaos_score
games_df['chaos_score'] = (
    games_df['lead_change_count'] * 0.4 +
    games_df['explosive_play_delta'] * 0.3 +
    games_df['win_prob_volatility'] * 100 * 0.3
)

#games_df['year'] = games_df['start_date'].dt.year



# Keep only the columns you care about
sentiment_df = sentiment_df[["Year", "Week_Index", "Conference", "Mentions_conf", "Sentiment_conf"]]



# --- Chaos dataset prep ---
games_df['start_date'] = pd.to_datetime(games_df['start_date'], errors='coerce')
games_df['year'] = games_df['start_date'].dt.year

games_df['chaos_score'] = (
    games_df['lead_change_count'] * 0.4 +
    games_df['explosive_play_delta'] * 0.3 +
    games_df['excitement_index'] * 0.3
)

chaos_df = (
    games_df
    .groupby(["year", "week", "home_conference", "away_conference"], as_index=False)
    .agg(avg_chaos_score=("chaos_score", "mean"))
    .rename(columns={"year":"Year", "week": "Week_Index", "home_conference":"Conference", "away_conference":"Conference_Opponent"})
)

# --- Merge sentiment + chaos ---
final_df = sentiment_df.merge(
    chaos_df,
    on=["Year", "Week_Index", "Conference"],
    how="left"
)

# Optional: trim to weeks <= 13
final_df = final_df[final_df["Week_Index"] <= 13]
final_df.to_csv("final_df.csv", index=False)


