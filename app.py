import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np
import random
from datetime import datetime

# Page config
st.set_page_config(page_title="College Football Chaos Dashboard", page_icon="🔥", layout="wide")

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
    .chaos-high { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .chaos-medium { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .chaos-low { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    </style>
""", unsafe_allow_html=True)

# Title with animated emoji
st.title("🔥 College Football Chaos Dashboard 🏈")
st.caption("Real-time chaos analysis with sentiment analysis")


# Load data
@st.cache_data
def load_data():
    df_init = pd.read_csv("chaos_data.csv")
    # Calculate chaos_score FIRST if it doesn't exist
    if 'chaos_score' not in df_init.columns:
        # Use default weights for initial load
        df_init["chaos_score"] = (
                0.4 * df_init["lead_change_count"] +
                0.3 * df_init["explosive_play_delta"] +
                0.3 * df_init["win_prob_volatility"]
        )
    power5_conferences = ['SEC', 'Big Ten', 'ACC', 'Big 12', 'Pac-12']

    print(df_init.columns)
    print(df_init.head())

    # Filter to only Power 5 conferences
    df = df_init[df_init['home_conference'].isin(power5_conferences)]

    st.write("Columns in Data:", df.columns.tolist())
    st.write("Sample Power 5 Data:", df.head())
    return df

df = load_data()
# Initialize session state
if 'viewed_games' not in st.session_state:
    st.session_state.viewed_games = []
if 'favorite_teams' not in st.session_state:
    st.session_state.favorite_teams = set()
if 'chaos_alerts' not in st.session_state:
    st.session_state.chaos_alerts = []
if 'selected_game_video' not in st.session_state:
    st.session_state.selected_game_video = None

# Sidebar Controls
st.sidebar.header("⚙️ Chaos Score Tuner")

lead_weight = st.sidebar.slider("Lead Change Weight", 0.0, 1.0, 0.4)
explosive_weight = st.sidebar.slider("Explosive Play Weight", 0.0, 1.0, 0.3)
volatility_weight = st.sidebar.slider("Volatility Weight", 0.0, 1.0, 0.3)

# Normalize weights
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

# Add chaos level categories
df["chaos_level"] = pd.cut(df["chaos_score"], 
                            bins=[0, df["chaos_score"].quantile(0.33), 
                                  df["chaos_score"].quantile(0.67), df["chaos_score"].max()],
                            labels=["Low", "Medium", "High"])

# Calculate historical comparisons
season_avg = df["chaos_score"].mean()
season_std = df["chaos_score"].std()

# Detect upsets (ranked team loses OR game exceeds spread by significant margin)
df['is_upset'] = ((df['home_rank'].notna()) | (df['away_rank'].notna())) & (df['chaos_score'] > df['chaos_score'].quantile(0.7))


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
    upset_pct = upset_count / len(df) * 100
    st.metric("😱 Upset Games", f"{upset_pct:.1f}%",
              delta=f"{upset_count} potential upsets")

with col4:
    most_volatile_week = df.groupby("week")["chaos_score"].std().idxmax()
    st.metric("🌪️ Most Volatile Week", f"Week {most_volatile_week}",
              delta=f"σ = {df[df['week']==most_volatile_week]['chaos_score'].std():.2f}")

st.markdown("---")


# ==========================
# TEAM SELECTION & FILTERING
# ==========================
col1, col2, col3 = st.columns([2, 1, 1])

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
        st.caption("📌 " + ", ".join(list(st.session_state.favorite_teams)[:3]))


# Filter DataFrame
if selected_team != "All Teams":
    df_filtered = df[(df["home"] == selected_team) | (df["away"] == selected_team)]
else:
    df_filtered = df


# ==========================
# MAIN SCATTER PLOT WITH VIDEO SELECTION
# ==========================
st.subheader("📊 Interactive Chaos Timeline")

fig_scatter = px.scatter(
    df_filtered,
    x="game_id",
    y="chaos_score",
    hover_data=["home", "away", "week", "lead_change_count", "explosive_play_delta", "win_prob_volatility"],
    color="chaos_level",
    color_discrete_map={"Low": "#43e97b", "Medium": "#4facfe", "High": "#f5576c"},
    title=f"Chaos Scores for {selected_team}" if selected_team != "All Teams" else "Chaos Scores for All Games",
    size="explosive_play_delta",
    size_max=15,
    custom_data=['game_id', 'home', 'away', 'video_url']
)
fig_scatter.update_layout(height=500, clickmode='event+select')
st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_chart")

# Video player section
selected_points = st.session_state.get('selected_game_video')

col1, col2 = st.columns([2, 1])

with col1:
    # /*game_selector = st.selectbox(
    #     "Or select game directly:",
    #     options=[""] + df_filtered[df_filtered['video_url'].notna()]['game_id'].astype(str).tolist(),
    #     format_func=lambda x: f"Game {x}" if x else "Choose a game..."
    # )
    game_selector = st.selectbox("Select a Game", ["All Games"], key="game_selector")
    
    if game_selector:
        selected_game = df_filtered[df_filtered['game_id'] == int(game_selector)].iloc[0]
            
        # Historical comparison
        pct_vs_avg = ((selected_game['chaos_score'] - season_avg) / season_avg) * 100
        comparison_text = f"{'🔥 ' if pct_vs_avg > 0 else '😌 '}{abs(pct_vs_avg):.1f}% {'more' if pct_vs_avg > 0 else 'less'} chaotic than season average"
            
        st.info(f"**Historical Context:** {comparison_text}")

            
        # Game stats
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Chaos Score", f"{selected_game['chaos_score']:.2f}")
        with col_b:
            st.metric("Lead Changes", int(selected_game['lead_change_count']))
        with col_c:
            st.metric("Explosive Plays", int(selected_game['explosive_play_delta']))

with col2:
    if game_selector:
        st.subheader("📈 Game Context")
        
        # Z-score calculation
        z_score = (selected_game['chaos_score'] - season_avg) / season_std
        
        st.write(f"**Z-Score:** {z_score:.2f}")
        st.write(f"**Percentile:** {(df['chaos_score'] < selected_game['chaos_score']).mean() * 100:.1f}%")
        
        # Similar games
        df_temp = df.copy()
        df_temp['similarity'] = np.abs(df_temp['chaos_score'] - selected_game['chaos_score'])
        similar = df_temp[df_temp['game_id'] != selected_game['game_id']].nsmallest(3, 'similarity')
        
        st.write("**🎯 Similar Games:**")
        for _, sim in similar.iterrows():
            st.caption(f"• {sim['home']} vs {sim['away']} ({sim['chaos_score']:.2f})")

st.markdown("---")

# ==========================
# CONFERENCE CHAOS ANALYSIS
# ==========================
st.subheader("🏟️ Conference Chaos Rankings")

conf_stats = df.groupby('home_conference').agg({
    'chaos_score': ['mean', 'std', 'max'],
    'game_id': 'count'
}).round(2)
conf_stats.columns = ['Avg Chaos', 'Volatility', 'Peak Chaos', 'Games']
conf_stats = conf_stats.sort_values('Avg Chaos', ascending=False).reset_index()

col1, col2 = st.columns([1, 1])

with col1:
    st.dataframe(
        conf_stats,
        column_config={
            "conference": st.column_config.TextColumn("Conference"),
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
        y='conference',
        orientation='h',
        color='Avg Chaos',
        color_continuous_scale='Reds',
        title="Conference Chaos Comparison"
    )
    st.plotly_chart(fig_conf, use_container_width=True)

st.markdown("---")

# ==========================
# RIVALRY & UPSET TRACKER
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("😱 Upset Predictor & Tracker")
    
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
            if pd.notna(game['week']):
                upset_games['week'] = df['week'].astype(int)
            
            st.markdown(f"""
            <div class='alert-box' style='border-left-color: #f5576c; background: #fff5f5;'>
                <strong>🚨 {rank_text}</strong><br>
                Chaos: {game['chaos_score']:.2f} | Week {game['week']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No major upsets detected in current filter")

with col2:
    st.subheader("🔥 Rivalry Game Analysis")
    
    # Mock rivalry detection - in real app, use a rivalry database
    st.write("**Top Rivalry Chaos Scores:**")
    
    rivalry_games = df.nlargest(5, 'chaos_score')[['home', 'away', 'chaos_score', 'week']]
    
    for idx, game in rivalry_games.iterrows():
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%); 
                    padding: 10px; border-radius: 5px; margin: 5px 0; color: white;'>
            <strong>{game['home']} vs {game['away']}</strong><br>
            <small>Week {game['week']} • Chaos: {game['chaos_score']:.2f}</small>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ==========================
# MOMENTUM TRACKER
# ==========================
st.subheader("📈 Team Momentum Tracker")

if selected_team != "All Teams":
    team_games = df[(df['home'] == selected_team) | (df['away'] == selected_team)].sort_values('week')
    
    if len(team_games) > 0:
        # Calculate rolling average
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
        
        fig_momentum.update_layout(
            title=f"{selected_team} Chaos Momentum",
            xaxis_title="Week",
            yaxis_title="Chaos Score",
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
else:
    st.info("Select a specific team to see momentum analysis")

st.markdown("---")

# ==========================
# TEAM LEADERBOARD
# ==========================
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
    df["archetype"] = kmeans.labels_
    
    archetype_names = {
        0: "Grind-it-out",
        1: "Explosive Fireworks", 
        2: "Back-and-Forth Thriller"
    }
    df["archetype_name"] = df["archetype"].map(archetype_names)
    
    archetype_counts = df["archetype_name"].value_counts()
    fig_pie = px.pie(
        values=archetype_counts.values,
        names=archetype_counts.index,
        title="Game Type Distribution",
        color_discrete_sequence=["#43e97b", "#f5576c", "#4facfe"],
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ==========================
# CHAOS PREDICTOR
# ==========================
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
        model = LinearRegression().fit(features, target)
        
        # Use DataFrame with column names to match training data
        input_df = pd.DataFrame([[lead_input, explosive_input, volatility_input]], 
                                columns=["lead_change_count", "explosive_play_delta", "win_prob_volatility"])
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
        
        # Historical comparison
        pct_vs_avg = ((predicted_chaos - season_avg) / season_avg) * 100
        st.info(f"📊 {abs(pct_vs_avg):.1f}% {'above' if pct_vs_avg > 0 else 'below'} season average")

with col3:
    if predict_button:
        st.write("**🎯 Similar Historical Games:**")
        
        df["similarity"] = np.abs(df["chaos_score"] - predicted_chaos)
        similar_games = df.nsmallest(5, "similarity")[["home", "away", "chaos_score", "week", "video_url"]]
        
        for _, game in similar_games.iterrows():
            has_video = "🎬" if pd.notna(game['video_url']) else ""
            st.caption(f"{has_video} {game['home']} vs {game['away']} - {game['chaos_score']:.2f} (Wk {game['week']})")

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
st.caption("🏈 College Football Chaos Dashboard v2.0 | Now with video integration, upset tracking, and conference analysis")
st.caption(f"Session Stats: {len(st.session_state.viewed_games)} games viewed | {len(st.session_state.favorite_teams)} favorite teams")
