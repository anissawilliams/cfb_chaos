import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np

# Title
st.title("College Football Chaos Dashboard")

# Load data
df = pd.read_csv("chaos_data.csv")

st.sidebar.header("Chaos Score Tuner")

lead_weight = st.sidebar.slider("Lead Change Weight", 0.0, 1.0, 0.4)
explosive_weight = st.sidebar.slider("Explosive Play Weight", 0.0, 1.0, 0.3)
volatility_weight = st.sidebar.slider("Volatility Weight", 0.0, 1.0, 0.3)

# Normalize weights if needed
total = lead_weight + explosive_weight + volatility_weight
lead_weight /= total
explosive_weight /= total
volatility_weight /= total

# Recalculate chaos score
df["chaos_score"] = (
    lead_weight * df["lead_change_count"] +
    explosive_weight * df["explosive_play_delta"] +
    volatility_weight * df["win_prob_volatility"]
)

st.subheader("Chaos Score by Game")

# Create a list of unique teams from both home and away columns
teams = sorted(set(df["home"]).union(set(df["away"])))

# Sidebar dropdown
selected_team = st.sidebar.selectbox("Select a Team", ["All Teams"] + teams)

# Filter the DataFrame
if selected_team != "All Teams":
    df_filtered = df[(df["home"] == selected_team) | (df["away"] == selected_team)]
else:
    df_filtered = df

fig = px.scatter(
    df_filtered,
    x="game_id",
    y="chaos_score",
    hover_data=["home", "away", "lead_change_count", "explosive_play_delta", "win_prob_volatility"],
    color="chaos_score",
    color_continuous_scale="Turbo",
    title=f"Chaos Scores for {selected_team}" if selected_team != "All Teams" else "Chaos Scores for All Games"
)


st.plotly_chart(fig)

# Combine home and away games for each team
teams_all = []

for _, row in df.iterrows():
    teams_all.append({"team": row["home"], "chaos_score": row["chaos_score"]})
    teams_all.append({"team": row["away"], "chaos_score": row["chaos_score"]})

# Create DataFrame and group by team
team_df = pd.DataFrame(teams_all)
leaderboard = team_df.groupby("team").mean().sort_values("chaos_score", ascending=False).reset_index()

# Show top 10
st.subheader("🏆 Chaos Leaderboard")
st.dataframe(leaderboard.head(10))


fig = px.bar(
    leaderboard.head(10),
    x="chaos_score",
    y="team",
    orientation="h",
    color="chaos_score",
    color_continuous_scale="Inferno",
    title="Top 10 Most Chaotic Teams"
)

st.plotly_chart(fig)

import random

st.subheader("🎲 Random Game Spotlight")

if st.button("Surprise Me with Chaos"):
    random_game = df_filtered.sample(1).iloc[0]

    st.markdown(f"### {random_game['home']} vs. {random_game['away']}")
    st.markdown(f"**Chaos Score:** {random_game['chaos_score']:.2f}")
    st.markdown(f"- Lead Changes: {random_game['lead_change_count']}")
    st.markdown(f"- Explosive Play Delta: {random_game['explosive_play_delta']}")
    st.markdown(f"- Win Probability Volatility: {random_game['win_prob_volatility']:.3f}")
    st.markdown(f"**Game ID:** {random_game['game_id']}")



# Train model
features = df[["lead_change_count", "explosive_play_delta", "win_prob_volatility"]]
target = df["chaos_score"]
model = LinearRegression().fit(features, target)

# Sidebar inputs
st.sidebar.header("🔮 Chaos Predictor")
lead_input = st.sidebar.number_input("Lead Changes", min_value=0, max_value=10, value=2)
explosive_input = st.sidebar.number_input("Explosive Play Delta", min_value=0, max_value=20, value=5)
volatility_input = st.sidebar.slider("Win Prob Volatility", 0.0, 0.5, 0.15)

# Predict
input_array = np.array([[lead_input, explosive_input, volatility_input]])
predicted_chaos = model.predict(input_array)[0]

st.subheader("Predicted Chaos Score")
st.metric(label="Chaos Score", value=f"{predicted_chaos:.2f}")

# Fit model
X = df[["lead_change_count", "explosive_play_delta", "win_prob_volatility"]]
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
df["archetype"] = kmeans.labels_

# Visualize
fig = px.scatter(
    df,
    x="explosive_play_delta",
    y="win_prob_volatility",
    color="archetype",
    hover_data=["home", "away", "lead_change_count"],
    title="Game Archetypes"
)
st.plotly_chart(fig)

st.subheader("Weekly Chaos Tracker")
print(df.columns)
# Group by week and calculate average chaos score
# Group by week and calculate average chaos score
weekly_chaos = df.groupby("week")["chaos_score"].mean().reset_index()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(weekly_chaos["week"], weekly_chaos["chaos_score"], marker='o', color='darkorange')
ax.set_title("Chaos Score by Week")
ax.set_xlabel("Week")
ax.set_ylabel("Average Chaos Score")
ax.grid(True)

# Display in Streamlit
st.pyplot(fig)

