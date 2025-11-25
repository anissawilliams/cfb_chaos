import streamlit as st
import plotly.express as px
import numpy as np

def render_overview(df_filtered, selected_team, season_avg, season_std, color_map, hover_data_cols):
    st.subheader("Interactive Chaos Timeline")

    fig = px.scatter(
        df_filtered, x="home", y="chaos_score",
        hover_data=hover_data_cols, color="chaos_level",
        color_discrete_map=color_map,
        title=f"Chaos Scores for {selected_team}" if selected_team != "All Teams" else "Chaos Scores for All Games",
        size="explosive_play_delta", size_max=15
    )
    st.plotly_chart(fig, use_container_width=True)

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
