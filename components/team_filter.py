import streamlit as st

def team_filter(df):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🎯 Team Analysis")
        teams = sorted(set(df["home"]).union(set(df["away"])))
        selected = st.selectbox("Select a Team", ["All Teams"] + teams, key="team_selector")

    if selected != "All Teams":
        df_filtered = df[(df["home"] == selected) | (df["away"] == selected)].copy()
    else:
        df_filtered = df.copy()

    if len(df_filtered) == 0:
        st.warning(f"No games found for {selected}")
        st.stop()

    return df_filtered, selected
