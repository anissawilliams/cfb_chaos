import streamlit as st
import plotly.express as px

def render_conference_tab(df, has_rankings):
    st.subheader("🏟️ Conference Chaos Trends")

    # Weekly conference averages
    conf_weekly = (
        df.groupby(["home_conference", "week"])["chaos_score"]
        .mean()
        .reset_index()
    )

    # Trend lines
    fig_conf_trend = px.line(
        conf_weekly,
        x="week",
        y="chaos_score",
        color="home_conference",
        markers=True,
        title="Weekly Chaos Trends by Conference"
    )
    st.plotly_chart(fig_conf_trend, use_container_width=True)

    # Overall averages
    conf_avg = df.groupby("home_conference")["chaos_score"].mean().reset_index()
    fig_conf_avg = px.bar(
        conf_avg.sort_values("chaos_score", ascending=True),
        x="chaos_score",
        y="home_conference",
        color="home_conference",
        orientation="h",
        title="Average Chaos Score by Conference"
    )
    st.plotly_chart(fig_conf_avg, use_container_width=True)

    # Percentile leaderboard (conference ranking each week)
    st.subheader("📊 Weekly Conference Percentile Leaderboard")
    conf_weekly_ranked = conf_weekly.copy()
    conf_weekly_ranked["percentile"] = conf_weekly_ranked.groupby("week")["chaos_score"].rank(pct=True) * 100

    fig_conf_percentile = px.line(
        conf_weekly_ranked,
        x="week",
        y="percentile",
        color="home_conference",
        markers=True,
        title="Conference Chaos Percentile by Week"
    )
    st.plotly_chart(fig_conf_percentile, use_container_width=True)

    # conference_tab.py
    from components.utils import momentum_chart

    # conf_games = df[df['conference'] == selected_conf].sort_values('week')
    # fig_conf_momentum = momentum_chart(conf_games, selected_conf)
    # st.plotly_chart(fig_conf_momentum, use_container_width=True)
