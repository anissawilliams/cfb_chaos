import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np
from datetime import datetime
import seaborn as sns

# Page config
st.set_page_config(page_title="College Football Chaos Dashboard", page_icon="🏈", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid;
        background: #fff5f5;
    }
    .rivalry-card {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        color: white;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏈 College Football Chaos Dashboard")
st.caption("Real-time chaos analysis with sentiment insights")


# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("chaos_data.csv")

    # Calculate initial chaos_score if it doesn't exist
    if 'chaos_score' not in df.columns:
        df["chaos_score"] = (
                0.4 * df["lead_change_count"] +
                0.3 * df["explosive_play_delta"] +
                0.3 * df["win_prob_volatility"]
        )

    # Filter to Power 5 conferences
    power5_conferences = ['SEC', 'Big Ten', 'ACC', 'Big 12', 'Pac-12']
    df_new = df[df['home_conference'].isin(power5_conferences)].copy()

    return df_new

# Cache the ML model
@st.cache_resource
def train_chaos_model(features, target):
    return LinearRegression().fit(features, target)


# Sidebar Controls
st.sidebar.header("⚙️ Chaos Score Tuner")

lead_weight = st.sidebar.slider("Lead Change Weight", 0.0, 1.0, 0.4,
                                help="Weight for lead changes in chaos calculation")
explosive_weight = st.sidebar.slider("Explosive Play Weight", 0.0, 1.0, 0.3, help="Weight for explosive plays")
volatility_weight = st.sidebar.slider("Volatility Weight", 0.0, 1.0, 0.3, help="Weight for win probability volatility")

# Normalize weights
total = lead_weight + explosive_weight + volatility_weight
lead_weight /= total
explosive_weight /= total
volatility_weight /= total
df = load_data()
# Recalculate chaos score with normalized weights
df["chaos_score"] = (
        lead_weight * df["lead_change_count"] +
        explosive_weight * df["explosive_play_delta"] +
        volatility_weight * df["win_prob_volatility"]
)
# Add ELO difference column
df["elo_diff"] = df["home_pregame_elo"] - df["away_pregame_elo"]

# Add chaos level categories
df["chaos_level"] = pd.cut(
    df["chaos_score"],
    bins=[0, df["chaos_score"].quantile(0.33), df["chaos_score"].quantile(0.67), df["chaos_score"].max()],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# Calculate historical comparisons
season_avg = df["chaos_score"].mean()
season_std = df["chaos_score"].std()

# Detect upsets (ranked team loses OR high chaos game with ranked team)
# Check if rank columns exist before using them
has_rankings = 'home_rank' in df.columns and 'away_rank' in df.columns

if has_rankings:
    df['is_upset'] = (
            ((df['home_rank'].notna()) | (df['away_rank'].notna())) &
            (df['chaos_score'] > df['chaos_score'].quantile(0.7))
    )
else:
    # If no rankings, just use top chaos games as potential upsets
    df['is_upset'] = df['chaos_score'] > df['chaos_score'].quantile(0.75)

# ==========================
# TOP METRICS ROW
# ==========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_chaos = df["chaos_score"].mean()
    st.metric("📊 Average Chaos", f"{avg_chaos:.2f}",
              delta=f"{(avg_chaos - df['chaos_score'].median()):.2f} vs median")

with col2:
    max_chaos_game = df.loc[df["chaos_score"].idxmax()]
    st.metric("🔥 Peak Chaos", f"{max_chaos_game['chaos_score']:.2f}",
              delta=f"{max_chaos_game['home'][:10]} vs {max_chaos_game['away'][:10]}")

with col3:
    upset_count = df['is_upset'].sum()
    upset_pct = upset_count / len(df) * 100 if len(df) > 0 else 0
    st.metric("😱 Upset Games", f"{upset_pct:.1f}%",
              delta=f"{upset_count} potential upsets")

with col4:

        most_volatile_week = df.groupby("week")["chaos_score"].std().idxmax()
        st.metric("🌪️ Most Volatile Week", f"Week {most_volatile_week}",
                  delta=f"σ = {df[df['week'] == most_volatile_week]['chaos_score'].std():.2f}")

st.markdown("---")

# ==========================
# TEAM SELECTION & FILTERING
# ==========================
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🎯 Team Analysis")
    teams = sorted(set(df["home"]).union(set(df["away"])))
    selected_team = st.selectbox("Select a Team", ["All Teams"] + teams, key="team_selector")


# Filter DataFrame
if selected_team != "All Teams":
    df_filtered = df[(df["home"] == selected_team) | (df["away"] == selected_team)].copy()
else:
    df_filtered = df.copy()

# Handle empty filtered dataframe
if len(df_filtered) == 0:
    st.warning(f"No games found for {selected_team}")
    st.stop()

# ==========================
# MAIN VISUALIZATION TABS
# ==========================
tab_names = ["📊 Overview", "🏟️ Conference Analysis", "📈 Team Deep Dive",  "🔮 Predictor", " 🏉 ELO"]

tabs = st.tabs(tab_names)

with tabs[0]:  # Overview
    st.subheader("Interactive Chaos Timeline")

    color_by = "chaos_level"

    color_map = {
        "Low": "#43e97b", "Medium": "#4facfe", "High": "#f5576c",
        "Positive": "#43e97b", "Neutral": "#4facfe", "Negative": "#f5576c"
    }

    hover_data_cols = ["home", "away", "week", "lead_change_count", "explosive_play_delta", "win_prob_volatility"]



    fig_scatter = px.scatter(
        df_filtered,
        x="home",
        y="chaos_score",
        hover_data=hover_data_cols,
        color=color_by,
        color_discrete_map=color_map,
        title=f"Chaos Scores for {selected_team}" if selected_team != "All Teams" else "Chaos Scores for All Games",
        size="explosive_play_delta",
        size_max=15,
        labels={'home': 'Home Team', 'chaos_score': 'Chaos Score'}
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Game selector for detailed view
    st.subheader("🎮 Game Details")

    game_options = df_filtered.apply(
        lambda x: f"Game {x['game_id']}: {x['home']} vs {x['away']} (Week {x['week']})",
        axis=1
    ).tolist()

    if game_options:
        selected_game_str = st.selectbox(
            "Select a game for detailed analysis:",
            options=["Choose a game..."] + game_options
        )

        if selected_game_str != "Choose a game...":
            game_id = int(selected_game_str.split(":")[0].replace("Game ", ""))
            selected_game = df_filtered[df_filtered['game_id'] == game_id].iloc[0]


            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Chaos Score", f"{selected_game['chaos_score']:.2f}")
                pct_vs_avg = ((selected_game['chaos_score'] - season_avg) / season_avg) * 100
                comparison = f"{'🔥' if pct_vs_avg > 0 else '😌'} {abs(pct_vs_avg):.1f}% {'more' if pct_vs_avg > 0 else 'less'} chaotic than average"
                st.caption(comparison)

            with col2:
                st.metric("Lead Changes", int(selected_game['lead_change_count']))
                st.metric("Explosive Plays", int(selected_game['explosive_play_delta']))

            with col3:
                z_score = (selected_game['chaos_score'] - season_avg) / season_std
                st.metric("Z-Score", f"{z_score:.2f}")
                percentile = (df['chaos_score'] < selected_game['chaos_score']).mean() * 100
                st.metric("Percentile", f"{percentile:.1f}%")

            # Similar games
            st.write("**🎯 Similar Games:**")
            df_temp = df.copy()
            df_temp['similarity'] = np.abs(df_temp['chaos_score'] - selected_game['chaos_score'])
            similar = df_temp[df_temp['game_id'] != selected_game['game_id']].nsmallest(5, 'similarity')

            for _, sim in similar.iterrows():
                st.caption(f"• {sim['home']} vs {sim['away']} — Chaos: {sim['chaos_score']:.2f} (Week {sim['week']})")

with tabs[1]:  # Conference Analysis
    st.subheader("🏟️ Conference Chaos Rankings")

    conf_stats = df.groupby('home_conference').agg({
        'chaos_score': ['mean', 'std', 'max'],
        'game_id': 'count'
    }).round(2)
    conf_stats.columns = ['Avg Chaos', 'Volatility', 'Peak Chaos', 'Games']
    conf_stats = conf_stats.sort_values('Avg Chaos', ascending=False).reset_index()
    conf_stats.columns = ['Conference', 'Avg Chaos', 'Volatility', 'Peak Chaos', 'Games']

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(
            conf_stats,
            column_config={
                "Conference": st.column_config.TextColumn("Conference"),
                "Avg Chaos": st.column_config.NumberColumn("Avg Chaos", format="%.2f"),
                "Volatility": st.column_config.NumberColumn("Volatility", format="%.2f"),
                "Peak Chaos": st.column_config.NumberColumn("Peak", format="%.2f"),
                "Games": st.column_config.NumberColumn("Games")
            },
            hide_index=True,
            use_container_width=True
        )

    with col2:
        fig_conf = px.bar(
            conf_stats,
            x='Avg Chaos',
            y='Conference',
            orientation='h',
            color='Avg Chaos',
            color_continuous_scale='Reds',
            title="Conference Chaos Comparison"
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    # Upset and Rivalry Sections
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("😱 Upset Tracker")

        upset_games = df[df['is_upset'] == True].sort_values('chaos_score', ascending=False).head(10)

        if not upset_games.empty:
            for _, game in upset_games.iterrows():
                rank_text = ""
                if has_rankings and pd.notna(game.get('home_rank')):
                    rank_text += f"#{int(game['home_rank'])} "
                rank_text += game['home'] + " vs "
                if has_rankings and pd.notna(game.get('away_rank')):
                    rank_text += f"#{int(game['away_rank'])} "
                rank_text += game['away']


                st.markdown(f"""
                <div class='alert-box' style='border-left-color: #f5576c;'>
                    <strong>🚨 {rank_text}</strong><br>
                    Chaos: {game['chaos_score']:.2f} | Week {int(game['week'])}</div>
                """, unsafe_allow_html=True)
        else:
            st.info("No major upsets detected in current filter")

    with col2:
        st.subheader("🔥 Top Chaos Games")

        rivalry_games = df.nlargest(5, 'chaos_score')[['home', 'away', 'chaos_score', 'week']]


        st.markdown(f"""
            <div class='rivalry-card'>
                <strong>{game['home']} vs {game['away']}</strong><br>
                <small>Week {int(game['week'])} • Chaos: {game['chaos_score']:.2f}</small>
            </div>
            """, unsafe_allow_html=True)

with tabs[2]:  # Team Deep Dive
    st.subheader("📈 Team Deep Dive")

    if selected_team != "All Teams":
        team_games = df[(df['home'] == selected_team) | (df['away'] == selected_team)].sort_values('week')

        if len(team_games) > 0:
            # Momentum tracker
            team_games = team_games.copy()
            team_games['rolling_chaos'] = team_games['chaos_score'].rolling(window=3, min_periods=1).mean()

            fig_momentum = go.Figure()
            fig_momentum.add_trace(go.Scatter(
                x=team_games['week'],
                y=team_games['chaos_score'],
                mode='markers',
                name='Game Chaos',
                marker=dict(size=12, color='#f5576c')
            ))
            fig_momentum.add_trace(go.Scatter(
                x=team_games['week'],
                y=team_games['rolling_chaos'],
                mode='lines',
                name='3-Game Trend',
                line=dict(color='#4facfe', width=3)
            ))


            # Trend analysis
            recent_trend = team_games['chaos_score'].tail(3).mean()
            early_trend = team_games['chaos_score'].head(3).mean()
            trend_direction = "📈 Increasing" if recent_trend > early_trend else "📉 Decreasing"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Trend", trend_direction)
            with col2:
                st.metric("Recent Avg", f"{recent_trend:.2f}")
            with col3:
                st.metric("Season Avg", f"{team_games['chaos_score'].mean():.2f}")

            # Leaderboard and archetypes
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏆 Chaos Leaderboard")

                teams_all = []
                for _, row in df.iterrows():
                    teams_all.append({"team": row["home"], "chaos_score": row["chaos_score"]})
                    teams_all.append({"team": row["away"], "chaos_score": row["chaos_score"]})

                team_df = pd.DataFrame(teams_all)
                leaderboard = team_df.groupby("team").agg({
                    "chaos_score": ["mean", "std", "count"]
                }).reset_index()
                leaderboard.columns = ["team", "avg_chaos", "chaos_std", "game_count"]
                leaderboard = leaderboard.sort_values("avg_chaos", ascending=False)

                top_10 = leaderboard.head(10).copy()
                top_10["avg_chaos"] = top_10["avg_chaos"].round(2)
                top_10["chaos_std"] = top_10["chaos_std"].round(2)

                st.dataframe(
                    top_10,
                    column_config={
                        "team": st.column_config.TextColumn("Team", width="medium"),
                        "avg_chaos": st.column_config.NumberColumn("Avg Chaos", format="%.2f"),
                        "chaos_std": st.column_config.NumberColumn("Volatility", format="%.2f"),
                        "game_count": st.column_config.NumberColumn("Games", format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )

            with col2:
                st.subheader("🎭 Game Archetypes")

                X = df[["lead_change_count", "explosive_play_delta", "win_prob_volatility"]]
                kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
                df_copy = df.copy()
                df_copy["archetype"] = kmeans.labels_

                archetype_names = {
                    0: "Grind-it-out",
                    1: "Explosive Fireworks",
                    2: "Back-and-Forth Thriller"
                }
                df_copy["archetype_name"] = df_copy["archetype"].map(archetype_names)

                archetype_counts = df_copy["archetype_name"].value_counts()
                fig_pie = px.pie(
                    values=archetype_counts.values,
                    names=archetype_counts.index,
                    title="Game Type Distribution",
                    color_discrete_sequence=["#43e97b", "#f5576c", "#4facfe"],
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("👈 Select a specific team from the dropdown above to see detailed analysis")

with tabs[3]:  # Predictor
    st.subheader("🔮 Advanced Chaos Predictor")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        with st.form("predictor_form"):
            st.write("**Input Game Metrics:**")
            lead_input = st.number_input("Lead Changes", min_value=0, max_value=15, value=3)
            explosive_input = st.number_input("Explosive Play Delta", min_value=0, max_value=30, value=8)
            volatility_input = st.slider("Win Prob Volatility", 0.0, 0.6, 0.2)

            predict_button = st.form_submit_button("⚡ Predict Chaos", type="primary")

    with col2:
        if predict_button:
            features = df[["lead_change_count", "explosive_play_delta", "win_prob_volatility"]]
            target = df["chaos_score"]
            model = train_chaos_model(features, target)

            input_df = pd.DataFrame(
                [[lead_input, explosive_input, volatility_input]],
                columns=["lead_change_count", "explosive_play_delta", "win_prob_volatility"]
            )
            predicted_chaos = model.predict(input_df)[0]

            if predicted_chaos < df["chaos_score"].quantile(0.33):
                level = "Low 😴"
                color = "#43e97b"
            elif predicted_chaos < df["chaos_score"].quantile(0.67):
                level = "Medium 🔥"
                color = "#4facfe"
            else:
                level = "High 🌋"
                color = "#f5576c"

            st.markdown(f"""
            <div style='background: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
                <h2 style='color: white; margin: 0;'>{predicted_chaos:.2f}</h2>
                <p style='color: white; margin: 5px 0 0 0;'>Predicted Level: {level}</p>
            </div>
            """, unsafe_allow_html=True)

            pct_vs_avg = ((predicted_chaos - season_avg) / season_avg) * 100
            st.info(f"📊 {abs(pct_vs_avg):.1f}% {'above' if pct_vs_avg > 0 else 'below'} season average")

    with col3:
        if predict_button:
            st.write("**🎯 Similar Historical Games:**")

            df_temp = df.copy()
            df_temp["similarity"] = np.abs(df_temp["chaos_score"] - predicted_chaos)
            similar_games = df_temp.nsmallest(5, "similarity")[["home", "away", "chaos_score", "week"]]

            for _, game in similar_games.iterrows():
                st.caption(f"• {game['home']} vs {game['away']} — {game['chaos_score']:.2f} (Wk {int(game['week'])})")
## ELO

elo_range = st.sidebar.slider("Elo Difference Range", -300, 300, (-100, 100))


# filtered_df = df_filtered[
#     (df_filtered['elo_diff'].between(elo_range[0], elo_range[1])) &
#     (df_filtered['conference'] == df_filtered['home_conference']) & df_filtered['away_conference']
# ]

# filtered_df['elo_expected_winner'] = filtered_df.apply(
#     lambda row: 'home' if row['elo_diff'] > 0 else 'away', axis=1
# )
#filtered_df['elo_surprise'] = filtered_df['winner'] != filtered_df['elo_expected_winner']


with tabs[4]:
    st.header("Chaos vs ELO Differential")
    elo_data_cols = ["home", "away", "week", "lead_change_count", "explosive_play_delta", "win_prob_volatility", "elo_score"]
    fig_scatter_elo = px.scatter(
        df_filtered,
        x="elo_diff",
        y="chaos_score",
        hover_data=hover_data_cols,
        color=color_by,
        color_discrete_map=color_map,
        title=f"ELO Differential v. Chaos Scores with {selected_team}" if selected_team != "All Teams" else "ELO Differential v. Chaos Scores for All Games",
        size="explosive_play_delta",
        size_max=15,
        labels={'elo_diff': 'ELO Difference', 'chaos_score': 'Chaos Score'}
    )
    fig_scatter_elo.update_layout(height=500)
    st.plotly_chart(fig_scatter_elo, use_container_width=True)

    import plotly.figure_factory as ff

    components = ["win_prob_volatility", "explosive_play_delta", "lead_change_count", "elo_diff"]
    corr = df[components].corr()

    st.header("Correlation Heatmap: Chaos Components vs ELO")

    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=components,
        y=components,
        annotation_text=corr.round(2).values,
        colorscale="RdBu",
        showscale=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Time Series ELO v Chaos
    weekly = df.groupby("week").agg({
        "chaos_score": "mean",
        "elo_diff": "mean"
    }).reset_index()

    color_by = df_filtered["win_prob_volatility"]

    fig_scatter_elo_2 = px.scatter(
        df_filtered,
        x="elo_diff",
        y="win_prob_volatility",
        hover_data=hover_data_cols,
        color=color_by,
        color_discrete_map=color_map,
        title=f"ELO Differential v. Win Probability Volatility {selected_team}" if selected_team != "All Teams" else "ELO Differential v. Win Probability Volatility for All Games",
        size="explosive_play_delta",
        size_max=15,
        labels={'elo_diff': 'ELO Difference', 'win_prob_volatility': 'Win Probability Volatility'}
    )
    fig_scatter_elo_2.update_layout(height=500)
    st.plotly_chart(fig_scatter_elo_2, use_container_width=True)

    weekly = df.groupby("week")[["chaos_score", "elo_diff"]].mean().reset_index()
    st.write(df[["win_prob_volatility", "explosive_play_delta", "lead_change_count", "chaos_score"]].head())

    st.header("Weekly Chaos vs ELO Gap (Interactive)")

    fig_chaos_elo_ts = px.line(
        weekly,
        x="week",
        y=["chaos_score", "elo_diff"],
        markers=True,
        title="Weekly Chaos vs ELO Gap"
    )
    st.plotly_chart(fig_chaos_elo_ts, use_container_width=True)


# ==========================
# DOWNLOAD SECTION
# ==========================
st.subheader("📥 Export Data & Reports")

col1, col2, col3, col4 = st.columns(4)

with col1:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Full Dataset",
        data=csv,
        file_name=f"chaos_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="🎯 Filtered Data",
        data=csv_filtered,
        file_name=f"chaos_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col3:
    teams_all = []
    for _, row in df.iterrows():
        teams_all.append({"team": row["home"], "chaos_score": row["chaos_score"]})
        teams_all.append({"team": row["away"], "chaos_score": row["chaos_score"]})
    team_df = pd.DataFrame(teams_all)
    leaderboard = team_df.groupby("team").agg({"chaos_score": ["mean", "std", "count"]}).reset_index()
    leaderboard.columns = ["team", "avg_chaos", "chaos_std", "game_count"]
    #
    leaderboard_csv = leaderboard.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="🏆 Leaderboard",
        data=leaderboard_csv,
        file_name=f"chaos_leaderboard_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col4:
    upset_csv = df[df['is_upset'] == True].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="😱 Upset Report",
        data=upset_csv,
        file_name=f"upsets_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
footer_text = "🏈 College Football Chaos Dashboard v3.0 | Refactored with Sentiment Analysis"
# if has_sentiment:
#     footer_text += " ✅"
st.caption(footer_text)
