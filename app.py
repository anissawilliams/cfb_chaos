import streamlit as st
from components.data import load_data
from components.sidebar import chaos_score_tuner
from components.metrics import render_top_metrics
from components.team_filter import team_filter
from components.overview_tab import render_overview
from components.conference_tab import render_conference_tab
from components.predictor_tab import render_predictor_tab
from components.elo_tab import render_elo_tab
from components.team_tab import render_team_tab
from components.downloads import render_downloads
from components.utils import (
    calculate_chaos_score,
    add_upset_column,
    add_chaos_level,
    get_color_map,
    get_hover_columns,
    get_conference_colors
)


# Page setup
st.set_page_config(page_title="College Football Chaos Dashboard", page_icon="🏈", layout="wide")
st.title("🏈 College Football Chaos Dashboard")
st.caption("Real-time chaos analysis with ELO insights")

# Load data
df = load_data()

# Sidebar chaos tuner
lead_w, expl_w, vol_w = chaos_score_tuner()

# Apply chaos score + elo diff
df = calculate_chaos_score(df, lead_w, expl_w, vol_w)
df["elo_diff"] = df["home_pregame_elo"] - df["away_pregame_elo"]

# Add upset + chaos level
df = add_upset_column(df)
df = add_chaos_level(df)

# Season stats
season_avg = df["chaos_score"].mean()
season_std = df["chaos_score"].std()

# Top metrics row
render_top_metrics(df)

# Team filter (creates df_filtered + selected_team)
df_filtered, selected_team = team_filter(df)

# Apply chaos_level + upset to filtered df too
df_filtered = add_chaos_level(df_filtered)
df_filtered = add_upset_column(df_filtered)

# Shared configs
color_map = get_color_map()
hover_data_cols = get_hover_columns()

# Tabs
tabs = st.tabs(["📊 Overview", "🏟️ Conference Analysis", "📈 Team Deep Dive", "🔮 Predictor", "🏉 ELO"])

with tabs[0]:
    render_overview(df_filtered, selected_team, season_avg, season_std, color_map, hover_data_cols)

with tabs[1]:
    render_conference_tab(df, has_rankings=('home_rank' in df.columns and 'away_rank' in df.columns))

with tabs[2]:
    render_team_tab(df_filtered, selected_team, df, hover_data_cols)

with tabs[3]:
    render_predictor_tab(df, season_avg)

with tabs[4]:
    render_elo_tab(df, df_filtered, selected_team, "chaos_level", color_map, hover_data_cols)
    render_downloads(df, df_filtered)
