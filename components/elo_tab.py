# elo_tab.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

def _continuous_color(fig, label, scale="RdBu", reverse=False):
    # Ensure continuous color bar has a clear label and consistent scale direction
    fig.update_layout(coloraxis_colorbar=dict(title=label))
    fig.update_layout(coloraxis=dict(colorscale=scale, reversescale=reverse))
    return fig

def _add_zero_lines(fig, x0=True, y0=True):
    if x0:
        fig.add_vline(x=0, line=dict(color="rgba(0,0,0,0.2)", width=1))
    if y0:
        fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.2)", width=1))
    return fig

def render_elo_tab(df, df_filtered, selected_team, color_by, color_map, hover_data_cols):
    st.subheader("📊 ELO Ratings vs Chaos")
    st.caption(
        "This section explores how ELO ratings interact with chaos across college football. "
        "From team trajectories to league-wide scatterplots, histograms, and heatmaps, "
        "the visuals highlight where expected strength aligns with outcomes — and where volatility "
        "and surprise take over."
    )

    # --- Scatter: Chaos vs ELO Differential ---
    # If color_by is chaos_score, use continuous; otherwise, use categorical
    use_continuous = (color_by == "chaos_score")
    if use_continuous:
        fig_scatter_elo = px.scatter(
            df_filtered,
            x="elo_diff",
            y="chaos_score",
            hover_data=hover_data_cols,
            color="chaos_score",  # continuous
            color_continuous_scale="RdBu",
            labels={"elo_diff": "ELO Differential", "chaos_score": "Chaos Score"},
            title=(f"ELO Differential vs Chaos Scores ({selected_team})"
                   if selected_team != "All Teams" else "ELO Differential vs Chaos Scores (All Games)"),
            size="explosive_play_delta",
            size_max=15,
            range_x=[-750, 750]
        )
        _continuous_color(fig_scatter_elo, label="Chaos Score", scale="RdBu", reverse=True)
    else:
        fig_scatter_elo = px.scatter(
            df_filtered,
            x="elo_diff",
            y="chaos_score",
            hover_data=hover_data_cols,
            color=color_by,  # categorical (e.g., conference)
            color_discrete_map=color_map,
            labels={"elo_diff": "ELO Differential", "chaos_score": "Chaos Score", color_by: color_by.replace("_", " ").title()},
            title=(f"ELO Differential vs Chaos Scores colored by {color_by.replace('_',' ').title()} ({selected_team})"
                   if selected_team != "All Teams" else f"ELO Differential vs Chaos Scores colored by {color_by.replace('_',' ').title()}"),
            size="explosive_play_delta",
            size_max=15,
            range_x=[-750, 750]
        )

    _add_zero_lines(fig_scatter_elo, x0=True, y0=False)
    fig_scatter_elo.update_layout(hovermode="closest")
    st.plotly_chart(fig_scatter_elo, use_container_width=True)
    st.caption("Game-level view of how ELO mismatches correlate with chaos — bigger gaps often mean bigger surprises.")

    # --- Correlation Heatmap ---
    components = ["win_prob_volatility", "explosive_play_delta", "lead_change_count", "elo_diff"]
    corr = df[components].corr()

    st.subheader("🌡️ Chaos component correlations")
    fig_corr = ff.create_annotated_heatmap(
        z=corr.values,
        x=components,
        y=components,
        annotation_text=corr.round(2).values,
        colorscale="RdBu",
        showscale=True
    )
    fig_corr.layout.xaxis.title = "Components"
    fig_corr.layout.yaxis.title = "Components"
    # Center zero to emphasize positive vs negative relationships
    fig_corr.update_layout(coloraxis=dict(colorscale="RdBu", cmin=-1, cmax=1))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Correlation across chaos components and ELO differences.")

    # --- Scatter: ELO diff vs volatility (continuous color) ---
    st.subheader("🔴 ELO Differential vs Win Probability Volatility")
    fig_scatter_elo_2 = px.scatter(
        df_filtered,
        x="elo_diff",
        y="win_prob_volatility",
        hover_data=hover_data_cols,
        color="win_prob_volatility",  # continuous
        color_continuous_scale="Plasma",
        labels={"elo_diff": "ELO Differential", "win_prob_volatility": "Win Prob Volatility"},
        title=("ELO Differential vs Win Probability Volatility"
               if selected_team == "All Teams" else f"ELO Differential vs Win Probability Volatility ({selected_team})"),
        size="explosive_play_delta",
        size_max=15,
        range_x=[-750, 750]
    )
    _continuous_color(fig_scatter_elo_2, label="Win Prob Volatility", scale="Plasma")
    _add_zero_lines(fig_scatter_elo_2, x0=True, y0=False)
    fig_scatter_elo_2.update_layout(hovermode="closest")
    st.plotly_chart(fig_scatter_elo_2, use_container_width=True)
    st.caption("Larger ELO gaps often coincide with more volatile win probability trajectories.")

    # --- Weekly averages with dual-axis overlay ---
    df["week"] = df["week"].astype(int)
    weekly = df.groupby("week")[["chaos_score", "elo_diff"]].mean().reset_index()

    st.subheader("📅 Weekly Chaos vs ELO Gap")
    fig_chaos_elo_ts = go.Figure()
    fig_chaos_elo_ts.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["chaos_score"],
        mode="lines+markers", name="Chaos Score",
        line=dict(color="#f5576c", width=3), marker=dict(size=8)
    ))
    fig_chaos_elo_ts.add_trace(go.Scatter(
        x=weekly["week"], y=weekly["elo_diff"],
        mode="lines+markers", name="ELO Differential (avg)",
        line=dict(color="#4facfe", width=3, dash="dash"),
        marker=dict(size=8), yaxis="y2"
    ))
    chaos_q75 = weekly["chaos_score"].quantile(0.75)
    fig_chaos_elo_ts.add_hrect(
        y0=chaos_q75, y1=weekly["chaos_score"].max(),
        fillcolor="rgba(245,87,108,0.12)", line_width=0,
        annotation_text="High Chaos Weeks (≥ 75th percentile)", annotation_position="top left"
    )
    fig_chaos_elo_ts.update_layout(
        title="Weekly Chaos vs ELO Gap",
        xaxis=dict(title="Week"),
        yaxis=dict(title="Chaos Score", color="#f5576c"),
        yaxis2=dict(title="ELO Differential", overlaying="y", side="right", color="#4facfe"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_chaos_elo_ts, use_container_width=True)
    st.caption("Comparison of Elo rating differentials and Chaos Factor trends across the season. Chaos spikes in late-season games despite stable Elo predictions, revealing volatility not captured by traditional strength metrics.")

    # --- Team ELO Trajectories with chaos overlay (dual-axis) ---
    st.subheader("📈 Team ELO Trajectories")
    st.caption("Select a team to see ELO evolution with chaos overlay.")
    team_choice = st.selectbox("Choose a team", sorted(pd.unique(pd.concat([df["home"], df["away"]]))))
    team_games = df[(df["home"] == team_choice) | (df["away"] == team_choice)].copy()

    elo_home_col = "home_pregame_elo" if "home_pregame_elo" in df.columns else "elo_home"
    elo_away_col = "away_pregame_elo" if "away_pregame_elo" in df.columns else "elo_away"

    team_games["elo"] = team_games.apply(
        lambda row: row[elo_home_col] if row["home"] == team_choice else row[elo_away_col],
        axis=1
    )
    team_games = team_games.sort_values("week")

    fig_team = go.Figure()
    fig_team.add_trace(go.Scatter(
        x=team_games["week"],
        y=team_games["elo"],
        mode="lines+markers",
        name="ELO Rating",
        line=dict(color="#4facfe", width=3)
    ))
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
    # Reference band for high chaos weeks (team-specific)
    chaos_q75_team = team_games["chaos_score"].quantile(0.75)
    fig_team.add_hrect(
        y0=chaos_q75_team, y1=team_games["chaos_score"].max(),
        fillcolor="rgba(245,87,108,0.12)", line_width=0,
        annotation_text="High Chaos Weeks", annotation_position="top left", yref="y2"
    )
    st.plotly_chart(fig_team, use_container_width=True)
    st.caption("Weekly ELO ratings and chaos scores — a view into consistency vs volatility for the selected team.")

    # --- League Scatter: Chaos vs ELO (Team Averages) ---
    st.subheader("🌐 League-wide Chaos vs ELO")
    st.caption("Average chaos vs average ELO rating per team.")
    # Use the 'team' identity across home/away by normalizing team names from both sides
    power5 = ["ACC", "Big Ten", "Big 12", "Pac-12", "SEC"]

    # Get all unique teams that belong to Power 5
    teams_power5 = pd.unique(
        pd.concat([df.loc[df["home_conference"].isin(power5), "home"],
                   df.loc[df["away_conference"].isin(power5), "away"]])
    )

    team_rows = []
    for team in teams_power5:
        t_games = df[(df["home"] == team) | (df["away"] == team)]
        # Use pregame elo for the team perspective
        elo_vals = np.where(t_games["home"] == team, t_games.get("home_pregame_elo", t_games.get("elo_home")), t_games.get("away_pregame_elo", t_games.get("elo_away")))
        team_rows.append({"team": team, "avg_elo": pd.Series(elo_vals).mean(), "avg_chaos": t_games["chaos_score"].mean()})
    team_summary = pd.DataFrame(team_rows).dropna()

    fig_scatter_league = px.scatter(
        team_summary,
        x="avg_elo",
        y="avg_chaos",
        text="team",  # words as markers
        labels={"avg_elo": "Average ELO", "avg_chaos": "Average Chaos"},
        title="Chaos vs ELO (Team Averages)",
        color="avg_chaos",  # still color by chaos
        color_continuous_scale="RdYlBu",
    )

    # --- decluttering tweaks ---
    fig_scatter_league.update_traces(
        textposition="top center",
        textfont_size=10,  # smaller font
        marker=dict(size=8)  # smaller marker anchor
    )

    # remove the color bar to free space
    fig_scatter_league.update_layout(coloraxis_colorbar=None)

    # improve spacing and readability
    fig_scatter_league.update_layout(
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig_scatter_league, use_container_width=True)
    st.caption("Each point is a team’s average chaos and ELO rating.")

    # --- Distribution View ---
    # --- Distribution View ---
    st.subheader("📊 Distribution of Elo Differentials")

    # Calculate mean and median
    mean_val = df["elo_diff"].mean()
    median_val = df["elo_diff"].median()

    # Create histogram
    fig_hist = px.histogram(
        df,
        x="elo_diff",
        nbins=40,
        title="Distribution of Elo Differentials (2018–2023)",
        labels={"elo_diff": "Elo Differential"},
        color_discrete_sequence=["#4facfe"],
        opacity=0.75
    )

    # Add mean and median lines
    fig_hist.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean = {mean_val:.2f}",
        annotation_position="top right"
    )
    fig_hist.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median = {median_val:.2f}",
        annotation_position="bottom right"
    )

    # Update layout
    fig_hist.update_layout(
        bargap=0.1,
        xaxis_title="Elo Differential",
        yaxis_title="Game Count",
        hovermode="x unified",
        template="simple_white",
        xaxis_range=[-400, 400]
    )

    _add_zero_lines(fig_hist, x0=True, y0=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Caption - updated to be more poignant with the smaller range
    st.caption(
        "Distribution of pregame Elo rating differentials across Power Five matchups, 2018–2023. "
        "Most contests fall within ±150 Elo points, but a long tail of lopsided games underscores "
        "the need for volatility measures such as the Chaos Factor."
    )