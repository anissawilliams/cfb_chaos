import streamlit as st
import plotly.express as px

def render_team_tab(df_filtered, selected_team, df, hover_data_cols):
    st.subheader("📈 Team Deep Dive")

    view_mode = st.radio(
        "Choose view mode:",
        ["Selected Team", "Top 10 Chaos Teams"],
        horizontal=True
    )

    if view_mode == "Selected Team" and selected_team != "All Teams":
        st.subheader(f"Chaos Timeline — {selected_team}")

        # Timeline
        fig_team_line = px.line(
            df_filtered,
            x="week",
            y="chaos_score",
            markers=True,
            hover_data=["home", "away", "lead_change_count", "explosive_play_delta", "win_prob_volatility"],
            title=f"Chaos Score Timeline for {selected_team}"
        )
        st.plotly_chart(fig_team_line, use_container_width=True)

        # Conference overlay
        team_conf = df_filtered["home_conference"].iloc[0] if "home_conference" in df_filtered.columns else None
        if team_conf:
            conf_weekly = df[df["home_conference"] == team_conf].groupby("week")["chaos_score"].mean().reset_index()
            team_weekly = df_filtered.groupby("week")["chaos_score"].mean().reset_index()

            fig_conf_overlay = px.line(
                conf_weekly,
                x="week",
                y="chaos_score",
                markers=True,
                title=f"{selected_team} vs {team_conf} Average Chaos"
            )
            fig_conf_overlay.add_scatter(
                x=team_weekly["week"],
                y=team_weekly["chaos_score"],
                mode="lines+markers",
                name=selected_team,
                line=dict(color="red", width=3)
            )
            st.plotly_chart(fig_conf_overlay, use_container_width=True)

            # Percentile rank within conference
            conf_avg_scores = df[df["home_conference"] == team_conf].groupby("home")["chaos_score"].mean()
            team_avg = df_filtered["chaos_score"].mean()
            percentile = (conf_avg_scores < team_avg).mean() * 100

            st.metric("📈 Conference Percentile",
                      f"{percentile:.1f}%",
                      delta=f"Avg Chaos {team_avg:.2f} vs Conf Avg {conf_avg_scores.mean():.2f}")

        # Chaos components breakdown
        st.subheader("Chaos Components Breakdown")
        fig_components = px.bar(
            df_filtered,
            x="week",
            y="lead_change_count",
            color="explosive_play_delta",
            hover_data=["win_prob_volatility", "chaos_score"],
            title=f"Lead Changes & Explosive Plays — {selected_team}"
        )
        st.plotly_chart(fig_components, use_container_width=True)

    else:
        st.subheader("Chaos Timeline — Top 10 Teams")
        team_avg = df.groupby("home")["chaos_score"].mean().nlargest(10).index
        df_top = df[(df["home"].isin(team_avg)) | (df["away"].isin(team_avg))]

        fig_top = px.line(
            df_top,
            x="week",
            y="chaos_score",
            color="home",
            markers=True,
            hover_data=["away", "lead_change_count", "explosive_play_delta", "win_prob_volatility"],
            title="Chaos Scores for Top 10 Teams"
        )
        st.plotly_chart(fig_top, use_container_width=True)

    # Summary metrics (always shown)
    avg_chaos = df_filtered["chaos_score"].mean()
    max_chaos = df_filtered["chaos_score"].max()
    st.metric("📊 Avg Chaos", f"{avg_chaos:.2f}")
    st.metric("🔥 Max Chaos", f"{max_chaos:.2f}")
