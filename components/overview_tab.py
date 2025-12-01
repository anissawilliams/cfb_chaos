import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

def render_overview(df_filtered, selected_team, season_avg, season_std, color_map, hover_data_cols):
    st.subheader("Chaos Visualizations")

    # Create tabs for two perspectives
    tab1, tab2 = st.tabs(["📅 Chaos Timeline", "🏈 Chaos by Team"])

    # --- Timeline View ---
    with tab1:
        df_filtered["matchup"] = df_filtered["home"] + " vs " + df_filtered["away"]
        fig_timeline = px.scatter(
            df_filtered,
            x="week",
            y="chaos_score",
            hover_name="matchup",
            hover_data=["chaos_score", "explosive_play_delta"],
            color="chaos_level",
            color_discrete_map=color_map,
            size="explosive_play_delta",
            size_max=15,
            labels={
                "week": "Week",
                "chaos_score": "Chaos Score",
                "explosive_play_delta": "Explosive Play Δ",
                "matchup": "Matchup"
            },
            title="Chaos Timeline (All Games)"
        )
        fig_timeline.update_layout(template="plotly_white")
        st.plotly_chart(fig_timeline, use_container_width=True)
        st.caption("Each point = one game. Hover shows matchup, chaos score, and explosive play delta.")

    # --- Team View ---
    with tab2:
        scatter_df = []
        for _, row in df_filtered.iterrows():
            scatter_df.append({"team": row["home"], "chaos_score": row["chaos_score"],
                               "explosive_play_delta": row["explosive_play_delta"], "chaos_level": row["chaos_level"]})
            scatter_df.append({"team": row["away"], "chaos_score": row["chaos_score"],
                               "explosive_play_delta": row["explosive_play_delta"], "chaos_level": row["chaos_level"]})
        scatter_df = pd.DataFrame(scatter_df)

        fig_team = px.scatter(
            scatter_df,
            x="team",
            y="chaos_score",
            hover_name="team",
            hover_data=["chaos_score", "explosive_play_delta"],
            color="chaos_level",
            color_discrete_map=color_map,
            size="explosive_play_delta",
            size_max=15,
            labels={"chaos_score": "Chaos Score", "team": "Team", "explosive_play_delta": "Explosive Play Δ"},
            title="Chaos Scores by Team"
        )
        fig_team.update_layout(xaxis_tickangle=-45, template="plotly_white")
        st.plotly_chart(fig_team, use_container_width=True)
        st.caption("Each point = one team-game chaos score. Hover shows team, chaos score, and explosive play delta.")



    # Game details
    st.subheader("🎮 Game Details")
    game_options = df_filtered.apply(
        lambda x: f"Game {x['game_id']}: {x['home']} vs {x['away']} (Week {x['week']})", axis=1
    ).tolist()

    if game_options:
        selected_game_str = st.selectbox("Select a game:", ["Choose a game..."] + game_options)
        if selected_game_str != "Choose a game...":
            game_id = int(selected_game_str.split(":")[0].replace("Game ", ""))
            selected_game = df_filtered[df_filtered['game_id'] == game_id].iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chaos Score", f"{selected_game['chaos_score']:.2f}")
                pct_vs_avg = ((selected_game['chaos_score'] - season_avg) / season_avg) * 100
                st.caption(f"{'🔥' if pct_vs_avg > 0 else '😌'} {abs(pct_vs_avg):.1f}% {'more' if pct_vs_avg > 0 else 'less'} chaotic than average")

            with col2:
                st.metric("Lead Changes", int(selected_game['lead_change_count']))
                st.metric("Explosive Plays", int(selected_game['explosive_play_delta']))

            with col3:
                z_score = (selected_game['chaos_score'] - season_avg) / season_std
                st.metric("Z-Score", f"{z_score:.2f}")
                percentile = (df_filtered['chaos_score'] < selected_game['chaos_score']).mean() * 100
                st.metric("Percentile", f"{percentile:.1f}%")

            # Similar games
            st.write("**🎯 Similar Games:**")
            df_temp = df_filtered.copy()
            df_temp['similarity'] = np.abs(df_temp['chaos_score'] - selected_game['chaos_score'])
            similar = df_temp[df_temp['game_id'] != selected_game['game_id']].nsmallest(5, 'similarity')
            for _, sim in similar.iterrows():
                st.caption(f"• {sim['home']} vs {sim['away']} — Chaos: {sim['chaos_score']:.2f} (Week {sim['week']})")
