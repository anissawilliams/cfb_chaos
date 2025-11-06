import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np
from datetime import datetime

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
    df_init = pd.read_csv("chaos_data.csv")

    # Calculate initial chaos_score if it doesn't exist
    if 'chaos_score' not in df_init.columns:
        df_init["chaos_score"] = (
                0.4 * df_init["lead_change_count"] +
                0.3 * df_init["explosive_play_delta"] +
                0.3 * df_init["win_prob_volatility"]
        )

    # Filter to Power 5 conferences
    power5_conferences = ['SEC', 'Big Ten', 'ACC', 'Big 12', 'Pac-12']
    df = df_init[df_init['home_conference'].isin(power5_conferences)].copy()

    return df


@st.cache_data
def load_sentiment_data():
    """Load sentiment analysis data from separate CSV"""
    try:
        sentiment_df = pd.read_csv("sentiment_data.csv")
        # Expected columns: game_id, sentiment_score, sentiment_label, comment_count, avg_excitement
        return sentiment_df
    except FileNotFoundError:
        st.warning("⚠️ sentiment_data.csv not found. Sentiment features disabled.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading sentiment data: {e}")
        return None


# Cache the ML model
@st.cache_resource
def train_chaos_model(features, target):
    return LinearRegression().fit(features, target)


# Load data
try:
    df = load_data()
    sentiment_df = load_sentiment_data()

    # Merge sentiment data if available
    if sentiment_df is not None:
        df = df.merge(sentiment_df, on='game_id', how='left')
        has_sentiment = True
    else:
        has_sentiment = False

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Initialize session state
if 'favorite_teams' not in st.session_state:
    st.session_state.favorite_teams = set()

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

# Recalculate chaos score with normalized weights
df["chaos_score"] = (
        lead_weight * df["lead_change_count"] +
        explosive_weight * df["explosive_play_delta"] +
        volatility_weight * df["win_prob_volatility"]
)

# Sentiment integration option
if has_sentiment:
    st.sidebar.markdown("---")
    st.sidebar.header("💬 Sentiment Options")
    show_sentiment = st.sidebar.checkbox("Include Sentiment Analysis", value=True)

    if show_sentiment and 'sentiment_score' in df.columns:
        sentiment_weight = st.sidebar.slider("Sentiment Impact", 0.0, 0.3, 0.1,
                                             help="How much fan sentiment affects chaos perception")
        # Add sentiment-adjusted chaos score
        df["chaos_score_adjusted"] = df["chaos_score"] * (1 + sentiment_weight * df["sentiment_score"].fillna(0))
        use_adjusted = st.sidebar.checkbox("Use Sentiment-Adjusted Chaos", value=False)

        if use_adjusted:
            df["chaos_score"] = df["chaos_score_adjusted"]
else:
    show_sentiment = False

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
df['is_upset'] = (
        ((df['home_rank'].notna()) | (df['away_rank'].notna())) &
        (df['chaos_score'] > df['chaos_score'].quantile(0.7))
)

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
    if has_sentiment and show_sentiment and 'sentiment_score' in df.columns:
        avg_sentiment = df['sentiment_score'].mean()
        sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😢"
        st.metric(f"{sentiment_emoji} Avg Sentiment", f"{avg_sentiment:.2f}",
                  delta=f"{df['comment_count'].sum() if 'comment_count' in df.columns else 0} comments")
    else:
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

with col2:
    st.subheader("⭐ Favorites")
    if teams:
        fav_team = st.selectbox("Add to favorites:", [""] + teams, key="fav_selector")
        if fav_team and st.button("➕ Add"):
            st.session_state.favorite_teams.add(fav_team)
            st.success(f"Added {fav_team}!")

    if st.session_state.favorite_teams:
        st.caption("📌 " + ", ".join(list(st.session_state.favorite_teams)[:5]))

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
tab_names = ["📊 Overview", "🏟️ Conference Analysis", "📈 Team Deep Dive", "🔮 Predictor"]
if has_sentiment and show_sentiment:
    tab_names.append("💬 Sentiment Analysis")

tabs = st.tabs(tab_names)

with tabs[0]:  # Overview
    st.subheader("Interactive Chaos Timeline")

    # Add sentiment color option
    color_by = "chaos_level"
    if has_sentiment and show_sentiment and 'sentiment_label' in df_filtered.columns:
        color_option = st.radio("Color by:", ["Chaos Level", "Sentiment"], horizontal=True)
        if color_option == "Sentiment":
            color_by = "sentiment_label"

    color_map = {
        "Low": "#43e97b", "Medium": "#4facfe", "High": "#f5576c",
        "Positive": "#43e97b", "Neutral": "#4facfe", "Negative": "#f5576c"
    }

    hover_data_cols = ["home", "away", "week", "lead_change_count", "explosive_play_delta", "win_prob_volatility"]
    if has_sentiment and show_sentiment and 'sentiment_score' in df_filtered.columns:
        hover_data_cols.extend(["sentiment_score", "comment_count"])

    fig_scatter = px.scatter(
        df_filtered,
        x="game_id",
        y="chaos_score",
        hover_data=hover_data_cols,
        color=color_by,
        color_discrete_map=color_map,
        title=f"Chaos Scores for {selected_team}" if selected_team != "All Teams" else "Chaos Scores for All Games",
        size="explosive_play_delta",
        size_max=15
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

            # Sentiment card if available
            if has_sentiment and show_sentiment and 'sentiment_score' in selected_game.index:
                sentiment_val = selected_game['sentiment_score']
                if pd.notna(sentiment_val):
                    sentiment_class = "sentiment-positive" if sentiment_val > 0.1 else "sentiment-negative" if sentiment_val < -0.1 else "sentiment-neutral"
                    sentiment_label = selected_game.get('sentiment_label', 'Unknown')
                    st.markdown(f"""
                    <div class='{sentiment_class}'>
                        <strong>💬 Fan Sentiment: {sentiment_label}</strong><br>
                        Score: {sentiment_val:.2f} | Comments: {int(selected_game.get('comment_count', 0))}
                    </div>
                    """, unsafe_allow_html=True)

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
                if pd.notna(game['home_rank']):
                    rank_text += f"#{int(game['home_rank'])} "
                rank_text += game['home'] + " vs "
                if pd.notna(game['away_rank']):
                    rank_text += f"#{int(game['away_rank'])} "
                rank_text += game['away']

                sentiment_badge = ""
                if has_sentiment and 'sentiment_label' in game.index and pd.notna(game['sentiment_label']):
                    sentiment_badge = f" | 💬 {game['sentiment_label']}"

                st.markdown(f"""
                <div class='alert-box' style='border-left-color: #f5576c;'>
                    <strong>🚨 {rank_text}</strong><br>
                    Chaos: {game['chaos_score']:.2f} | Week {int(game['week'])}{sentiment_badge}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No major upsets detected in current filter")

    with col2:
        st.subheader("🔥 Top Chaos Games")

        rivalry_games = df.nlargest(5, 'chaos_score')[['home', 'away', 'chaos_score', 'week']]

        for idx, game in rivalry_games.iterrows():
            sentiment_info = ""
            if has_sentiment and 'sentiment_score' in df.columns:
                game_full = df[df['game_id'] == df.iloc[idx]['game_id']].iloc[0]
                if pd.notna(game_full.get('sentiment_score')):
                    sentiment_info = f" | Sentiment: {game_full['sentiment_score']:.2f}"

            st.markdown(f"""
            <div class='rivalry-card'>
                <strong>{game['home']} vs {game['away']}</strong><br>
                <small>Week {int(game['week'])} • Chaos: {game['chaos_score']:.2f}{sentiment_info}</small>
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

            # Add sentiment trend if available
            if has_sentiment and show_sentiment and 'sentiment_score' in team_games.columns:
                team_games['rolling_sentiment'] = team_games['sentiment_score'].rolling(window=3, min_periods=1).mean()
                fig_momentum.add_trace(go.Scatter(
                    x=team_games['week'],
                    y=team_games['rolling_sentiment'],
                    mode='lines',
                    name='Sentiment Trend',
                    line=dict(color='#43e97b', width=2, dash='dash'),
                    yaxis='y2'
                ))

            fig_momentum.update_layout(
                title=f"{selected_team} Chaos Momentum",
                xaxis_title="Week",
                yaxis_title="Chaos Score",
                yaxis2=dict(title="Sentiment", overlaying='y',
                            side='right') if has_sentiment and show_sentiment else None,
                hovermode='x unified'
            )
            st.plotly_chart(fig_momentum, use_container_width=True)

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

# Sentiment Analysis Tab
if has_sentiment and show_sentiment:
    with tabs[4]:
        st.subheader("💬 Fan Sentiment Analysis")

        if 'sentiment_score' in df.columns and 'sentiment_label' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                # Sentiment distribution
                sentiment_counts = df['sentiment_label'].value_counts()
                fig_sentiment_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Overall Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={"Positive": "#43e97b", "Neutral": "#4facfe", "Negative": "#f5576c"},
                    hole=0.4
                )
                st.plotly_chart(fig_sentiment_pie, use_container_width=True)

                # Average sentiment by conference
                conf_sentiment = df.groupby('home_conference')['sentiment_score'].mean().sort_values(
                    ascending=False).reset_index()
                conf_sentiment.columns = ['Conference', 'Avg Sentiment']

                st.write("**Conference Fan Sentiment**")
                st.dataframe(
                    conf_sentiment,
                    column_config={
                        "Conference": st.column_config.TextColumn("Conference"),
                        "Avg Sentiment": st.column_config.NumberColumn("Avg Sentiment", format="%.2f")
                    },
                    hide_index=True,
                    use_container_width=True
                )

            with col2:
                # Chaos vs Sentiment correlation
                fig_correlation = px.scatter(
                    df,
                    x='chaos_score',
                    y='sentiment_score',
                    color='sentiment_label',
                    size='comment_count' if 'comment_count' in df.columns else None,
                    hover_data=['home', 'away', 'week'],
                    title="Chaos vs Fan Sentiment",
                    labels={'chaos_score': 'Chaos Score', 'sentiment_score': 'Sentiment Score'},
                    color_discrete_map={"Positive": "#43e97b", "Neutral": "#4facfe", "Negative": "#f5576c"}
                )
                fig_correlation.update_layout(height=400)
                st.plotly_chart(fig_correlation, use_container_width=True)

                # Correlation stats
                if df['sentiment_score'].notna().sum() > 0:
                    correlation = df[['chaos_score', 'sentiment_score']].corr().iloc[0, 1]
                    st.metric("Chaos-Sentiment Correlation", f"{correlation:.3f}")

                    if correlation > 0.3:
                        st.info("🔥 Strong positive correlation: More chaotic games = More excited fans!")
                    elif correlation < -0.3:
                        st.info("😴 Negative correlation: Chaotic games = Mixed fan reactions")
                    else:
                        st.info("🤷 Weak correlation: Sentiment varies independently of chaos")

            # Most discussed games
            st.subheader("🗣️ Most Discussed Games")
            if 'comment_count' in df.columns:
                most_discussed = df.nlargest(10, 'comment_count')[
                    ['home', 'away', 'week', 'chaos_score', 'sentiment_score', 'sentiment_label', 'comment_count']]

                for idx, game in most_discussed.iterrows():
                    sentiment_class = "sentiment-positive" if game[
                                                                  'sentiment_label'] == "Positive" else "sentiment-negative" if \
                    game['sentiment_label'] == "Negative" else "sentiment-neutral"

                    st.markdown(f"""
                    <div class='{sentiment_class}'>
                        <strong>{game['home']} vs {game['away']}</strong> (Week {int(game['week'])})<br>
                        <small>💬 {int(game['comment_count'])} comments | Chaos: {game['chaos_score']:.2f} | Sentiment: {game['sentiment_score']:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)

            # Sentiment over time
            st.subheader("📊 Sentiment Trends Over Season")
            weekly_sentiment = df.groupby('week').agg({
                'sentiment_score': 'mean',
                'chaos_score': 'mean',
                'comment_count': 'sum' if 'comment_count' in df.columns else 'count'
            }).reset_index()

            fig_trends = go.Figure()
            fig_trends.add_trace(go.Scatter(
                x=weekly_sentiment['week'],
                y=weekly_sentiment['sentiment_score'],
                mode='lines+markers',
                name='Avg Sentiment',
                line=dict(color='#43e97b', width=3),
                yaxis='y'
            ))
            fig_trends.add_trace(go.Scatter(
                x=weekly_sentiment['week'],
                y=weekly_sentiment['chaos_score'],
                mode='lines+markers',
                name='Avg Chaos',
                line=dict(color='#f5576c', width=3),
                yaxis='y2'
            ))

            fig_trends.update_layout(
                title="Weekly Sentiment vs Chaos Trends",
                xaxis_title="Week",
                yaxis=dict(title="Sentiment Score", side='left'),
                yaxis2=dict(title="Chaos Score", overlaying='y', side='right'),
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_trends, use_container_width=True)

        else:
            st.info("Sentiment data loaded but missing required columns (sentiment_score, sentiment_label)")

st.markdown("---")

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
if has_sentiment:
    footer_text += " ✅"
st.caption(footer_text)
if st.session_state.favorite_teams:
    st.caption(f"⭐ Favorite teams: {', '.join(st.session_state.favorite_teams)}")