import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff

def render_elo_tab(df, df_filtered, selected_team, color_by, color_map, hover_data_cols):
    st.header("Chaos vs ELO Differential")

    # Scatter: Chaos vs ELO diff
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
    st.plotly_chart(fig_scatter_elo, use_container_width=True)

    # Correlation heatmap
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

    # Scatter: ELO diff vs volatility
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
    st.plotly_chart(fig_scatter_elo_2, use_container_width=True)

    # Weekly averages
    df["week"] = df["week"].astype(int)
    weekly = df.groupby("week")[["chaos_score", "elo_diff"]].mean().reset_index()
    st.header("Weekly Chaos vs ELO Gap (Interactive)")
    fig_chaos_elo_ts = px.line(
        weekly,
        x="week",
        y=["chaos_score", "elo_diff"],
        markers=True,
        title="Weekly Chaos vs ELO Gap"
    )
    st.plotly_chart(fig_chaos_elo_ts, use_container_width=True)
