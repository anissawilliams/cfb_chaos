import streamlit as st
import pandas as pd
from datetime import datetime

def render_downloads(df, df_filtered):
    st.subheader("📥 Export Data & Reports")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Full Dataset",
            data=csv,
            file_name=f"chaos_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:
        csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="🎯 Filtered Data",
            data=csv_filtered,
            file_name=f"chaos_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col3:
        teams_all = []
        for _, row in df.iterrows():
            teams_all.append({"team": row["home"], "chaos_score": row["chaos_score"]})
            teams_all.append({"team": row["away"], "chaos_score": row["chaos_score"]})
        team_df = pd.DataFrame(teams_all)
        leaderboard = team_df.groupby("team").agg({"chaos_score": ["mean", "std", "count"]}).reset_index()
        leaderboard.columns = ["team", "avg_chaos", "chaos_std", "game_count"]

        leaderboard_csv = leaderboard.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="🏆 Leaderboard",
            data=leaderboard_csv,
            file_name=f"chaos_leaderboard_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col4:
        upset_csv = df[df['is_upset'] == True].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="😱 Upset Report",
            data=upset_csv,
            file_name=f"upsets_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.caption("🏈 College Football Chaos Dashboard v3.0 | Refactored with Sentiment Analysis")
