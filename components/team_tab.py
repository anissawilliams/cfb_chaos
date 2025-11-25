# team_tab.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans

def render_team_tab(df, selected_team, color_map, hover_data_cols):
    # Intro narrative
    st.subheader("📈 Team Deep Dive")
    st.caption(
        "This tab moves from **micro → meso → macro** views:\n"
        "- First, a deep dive into the selected team’s chaos journey.\n"
        "- Then, a league-wide leaderboard of chaos standings.\n"
        "- Finally, the broader archetypes of game styles across the season."
    )

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

            # Leaderboard and archetypes side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏆 Chaos Leaderboard")
                st.caption(
                    "This is the **league-wide chaos standings**. "
                    "It shows how all teams rank nationally in average chaos, "
                    "not just opponents of your selected team."
                )

                # Build leaderboard
                teams_all = []
                for _, row in df.iterrows():
                    teams_all.append({"team": row["home"], "chaos_score": row["chaos_score"]})
                    teams_all.append({"team": row["away"], "chaos_score": row["chaos_score"]})

                team_df = pd.DataFrame(teams_all)
                leaderboard = team_df.groupby("team").agg(
                    avg_chaos=("chaos_score", "mean"),
                    game_count=("chaos_score", "count")
                ).reset_index().sort_values("avg_chaos", ascending=False)

                leaderboard["Rank"] = range(1, len(leaderboard) + 1)
                leaderboard = leaderboard.rename(columns={
                    "team": "Team",
                    "avg_chaos": "Average Chaos",
                    "game_count": "Games Played"
                })

                # Show top 10 as card deck (2 rows of 5)
                top_10 = leaderboard.head(10)
                for i in range(0, len(top_10), 5):
                    cols_cards = st.columns(5)
                    for j, (_, row) in enumerate(top_10.iloc[i:i+5].iterrows()):
                        with cols_cards[j]:
                            highlight = (row["Team"] == selected_team)
                            label = f"#{row['Rank']} {row['Team']}"
                            value = f"{row['Average Chaos']:.2f}"
                            delta = f"Games {row['Games Played']}"
                            if highlight:
                                st.success(f"{label}\nChaos {value} | {delta}")
                            else:
                                st.metric(label=label, value=value, delta=delta)

                # Explicit caption for selected team
                if selected_team in leaderboard["Team"].values:
                    selected_row = leaderboard[leaderboard["Team"] == selected_team].iloc[0]
                    st.caption(
                        f"Your team **{selected_team}** ranks #{selected_row['Rank']} nationally "
                        f"with an average chaos of {selected_row['Average Chaos']:.2f} "
                        f"across {selected_row['Games Played']} games."
                    )

            with col2:
                st.subheader("🎭 Game Archetypes")
                st.caption(
                    "Beyond individual teams, these clusters reveal the *style* of games being played — "
                    "whether they’re grind-it-outs, explosive shootouts, or back-and-forth thrillers."
                )

                required_cols = ["lead_change_count", "explosive_play_delta", "win_prob_volatility"]
                if all(col in df.columns for col in required_cols):
                    X = df[required_cols].dropna()
                    if len(X) > 0:
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
                        st.info("No valid rows for archetype clustering.")
                else:
                    st.info("Missing columns for archetype clustering.")

    else:
        st.info("👈 Select a specific team from the dropdown above to see detailed analysis")
