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
        title="Weekly Chaos Trends by Conference",
        labels={"week": "Week", "chaos_score": "Chaos Score"}
    )
    st.plotly_chart(fig_conf_trend, use_container_width=True)
    st.caption("Each line represents the average chaos score for a given conference over time.")

    # Overall averages
    conf_avg = df.groupby("home_conference")["chaos_score"].mean().reset_index()
    fig_conf_avg = px.bar(
        conf_avg.sort_values("chaos_score", ascending=True),
        x="chaos_score",
        y="home_conference",
        color="home_conference",
        orientation="h",
        title="Average Chaos Score by Conference",
        labels={"chaos_score": "Average Chaos Score", "home_conference": "Conference"}
    )
    st.plotly_chart(fig_conf_avg, use_container_width=True)
    st.caption("The top conferences are consistently more chaotic than the bottom.")

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
        title="Conference Chaos Percentile by Week",
        labels={"week": "Week", "percentile": "Chaos Percentile Rank"}
    )
    st.plotly_chart(fig_conf_percentile, use_container_width=True)
    st.caption("This plot applies percentile ranking to the weekly average chaos score for each conference.")
    # conference_tab.py
    from components.utils import momentum_chart
    selected_conf = st.selectbox("Select a conference:", conf_avg["home_conference"])
    conf_games = df[df['home_conference'] == selected_conf].sort_values('week')
    fig_conf_momentum = momentum_chart(conf_games, selected_conf)
    st.plotly_chart(fig_conf_momentum, use_container_width=True)
    st.caption("Momentum of the selected conference over time with trend lines.")
