# team_tab.py
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.cluster import KMeans
from components.utils import momentum_chart

def render_team_tab(df, selected_team, color_map, hover_data_cols):
    st.subheader("📈 Team Deep Dive")

    if selected_team != "All Teams":
        team_games = df[(df['home'] == selected_team) | (df['away'] == selected_team)].sort_values('week')

        if len(team_games) > 0:
            # Momentum chart (reusable from utils)
            fig_momentum = momentum_chart(team_games, selected_team)
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

            # Leaderboard
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏆 Chaos Leaderboard")
                # leaderboard logic inline for now

            with col2:
                st.subheader("🎭 Game Archetypes")
                # archetype clustering logic inline
    else:
        st.info("👈 Select a specific team from the dropdown above to see detailed analysis")
