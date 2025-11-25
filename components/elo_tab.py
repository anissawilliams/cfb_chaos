# elo_tab.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np



def render_elo_tab(df, df_filtered, selected_team, color_by, color_map, hover_data_cols):
    st.subheader("📊 ELO Ratings vs Chaos")
    st.caption(
        "ELO ratings measure expected strength. Chaos scores measure unpredictability. "
        "Together, they reveal when order breaks down."
    )

    # --- Scatter: Chaos vs ELO Differential ---
    fig_scatter_elo = px.scatter(
        df_filtered,
        x="elo_diff",
        y="chaos_score",
        hover_data=hover_data_cols,
        color=color_by,
        color_discrete_map=color_map,
        title=f"ELO Differential v. Chaos Scores with {selected_team}"
        if selected_team != "All Teams" else "ELO Differential v. Chaos Scores for All Games",
        size="explosive_play_delta",
        size_max=15,
        range_x=[-750, 750],
        labels={'elo_diff': 'ELO Difference', 'chaos_score': 'Chaos Score'}
    )
    st.plotly_chart(fig_scatter_elo, use_container_width=True)

    # --- Correlation Heatmap ---
    components = ["win_prob_volatility", "explosive_play_delta", "lead_change_count", "elo_diff"]
    corr = df[components].corr()

    st.subheader("🔗 Correlation Heatmap")
    st.caption("How chaos components interact with ELO differences.")
    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=components,
        y=components,
        annotation_text=corr.round(2).values,
        colorscale="RdBu",
        showscale=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Scatter: ELO diff vs volatility ---
    fig_scatter_elo_2 = px.scatter(
        df_filtered,
        x="elo_diff",
        y="win_prob_volatility",
        hover_data=hover_data_cols,
        color=color_by,
        color_discrete_map=color_map,
        title=f"ELO Differential v. Win Probability Volatility ({selected_team})"
        if selected_team != "All Teams" else "ELO Differential v. Win Probability Volatility (All Games)",
        size="explosive_play_delta",
        size_max=15,
        range_x=[-750, 750],
        labels={'elo_diff': 'ELO Difference', 'win_prob_volatility': 'Win Probability Volatility'}
    )
    st.plotly_chart(fig_scatter_elo_2, use_container_width=True)

    # --- Weekly averages with dual-axis overlay ---
    df["week"] = df["week"].astype(int)
    weekly = df.groupby("week")[["chaos_score", "elo_diff"]].mean().reset_index()

    st.subheader("📅 Weekly Chaos vs ELO Gap")
    st.caption("League-wide averages per week, showing when chaos spikes against expected strength.")

    fig_chaos_elo_ts = go.Figure()
    fig_chaos_elo_ts.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["chaos_score"],
        mode="lines+markers", name="Chaos Score",
        line=dict(color="#f5576c", width=3), marker=dict(size=8)
    ))
    fig_chaos_elo_ts.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["elo_diff"],
        mode="lines+markers", name="ELO Difference",
        line=dict(color="#4facfe", width=3, dash="dash"),
        marker=dict(size=8), yaxis="y2"
    ))
    chaos_threshold = weekly["chaos_score"].quantile(0.75)
    fig_chaos_elo_ts.add_hrect(
        y0=chaos_threshold, y1=weekly["chaos_score"].max(),
        fillcolor="rgba(245,87,108,0.1)", line_width=0,
        annotation_text="High Chaos Weeks", annotation_position="top left"
    )
    fig_chaos_elo_ts.update_layout(
        title="Weekly Chaos vs ELO Gap",
        xaxis=dict(title="Week"),
        yaxis=dict(title="Chaos Score", color="#f5576c"),
        yaxis2=dict(title="ELO Difference", overlaying="y", side="right", color="#4facfe"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_chaos_elo_ts, use_container_width=True)

    # # --- Upset Spotlight ---
    # st.subheader("⚡ Chaos Upsets")
    # st.caption("Games where the lower-ELO team won — proof that chaos can override expected strength.")
    # if {"elo_home","elo_away","winner"}.issubset(df.columns):
    #     upsets = df[((df["home_pregame_elo"] > df["away_pregame_elo"]) & (df["winner"] == df["away"])) |
    #                 ((df["elo_away"] > df["elo_home"]) & (df["winner"] == df["home"]))]
    #     if not upsets.empty:
    #         for _, game in upsets.iterrows():
    #             st.caption(f"• {game['home']} vs {game['away']} — Chaos {game['chaos_score']:.2f}")
    #     else:
    #         st.info("No upsets detected in dataset.")

    # --- Team ELO Trajectories ---
    st.subheader("📈 Team ELO Trajectories")
    st.caption("Select a team to see how their ELO rating evolved, with chaos markers overlayed.")
    team_choice = st.selectbox("Choose a team", sorted(df["home"].unique()))
    team_games = df[(df["home"] == team_choice) | (df["away"] == team_choice)].copy()

    # Use pregame ELO if available
    elo_home_col = "home_pregame_elo" if "home_pregame_elo" in df.columns else "elo_home"
    elo_away_col = "away_pregame_elo" if "away_pregame_elo" in df.columns else "elo_away"

    team_games["elo"] = team_games.apply(
        lambda row: row[elo_home_col] if row["home"] == team_choice else row[elo_away_col],
        axis=1
    )
    team_games = team_games.sort_values("week")

    st.write("Chaos preview:", team_games[["week", "chaos_score"]].describe())

    team_games = team_games.sort_values("week")

    # Select the correct ELO for the chosen team
    team_games["elo"] = team_games.apply(
        lambda row: row[elo_home_col] if row["home"] == team_choice else row[elo_away_col],
        axis=1
    )

    fig_team = go.Figure()
    fig_team.add_trace(go.Scatter(
        x=team_games["week"],
        y=team_games["elo"],
        mode="lines+markers",
        name="ELO Rating",
        line=dict(color="#4facfe", width=3)
    ))

    # Overlay chaos on a secondary axis for clarity
    fig_team.add_trace(go.Scatter(
        x=team_games["week"],
        y=team_games["chaos_score"],
        mode="lines+markers",
        name="Chaos Score",
        line=dict(color="#f5576c", width=2, dash="dot"),
        yaxis="y2"
    ))

    fig_team.update_layout(
        title=f"{team_choice} ELO vs Chaos",
        xaxis_title="Week",
        yaxis=dict(title="ELO Rating", color="#4facfe"),
        yaxis2=dict(title="Chaos Score", overlaying="y", side="right", color="#f5576c"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_team, use_container_width=True)
    st.caption(
        "Tracking weekly ELO ratings and chaos scores for the selected team — a view into consistency vs volatility.")

    # --- League Scatter: Chaos vs ELO ---
    st.subheader("🌐 League-wide Chaos vs ELO")
    st.caption("Average chaos vs average ELO rating per team.")
    team_summary = df.groupby("home").agg(
        avg_elo=("home_pregame_elo", "mean"),
        avg_chaos=("chaos_score", "mean")
    ).reset_index()

    fig_scatter_league = px.scatter(
        team_summary,
        x="avg_elo",
        y="avg_chaos",
        text="home",
        title="Chaos vs ELO (Team Averages)",
        labels={"avg_elo": "Average ELO", "avg_chaos": "Average Chaos"},
        color_discrete_sequence=["#4facfe"]
    )
    fig_scatter_league.update_traces(textposition="top center")
    fig_scatter_league.update_layout(hovermode="closest")
    st.plotly_chart(fig_scatter_league, use_container_width=True)
    st.caption(
        "Each point represents a team’s average chaos score and ELO rating — revealing which programs defy expectations.")

    # --- Distribution View ---
    st.subheader("📊 Distribution of ELO Differences")

    fig_hist = px.histogram(
        df,
        x="elo_diff",
        nbins=30,
        title="Distribution of ELO Differences",
        labels={"elo_diff": "ELO Difference"},
        color_discrete_sequence=["#4facfe"]
    )
    fig_hist.update_layout(
        bargap=0.1,
        xaxis_title="ELO Difference",
        yaxis_title="Count",
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("Histogram of ELO differences across all games, showing where mismatches produced volatility.")

