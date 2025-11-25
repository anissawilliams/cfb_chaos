import streamlit as st

def render_top_metrics(df):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg = df["chaos_score"].mean()
        st.metric("📊 Average Chaos", f"{avg:.2f}",
                  delta=f"{(avg - df['chaos_score'].median()):.2f} vs median")

    with col2:
        max_game = df.loc[df["chaos_score"].idxmax()]
        st.metric("🔥 Peak Chaos", f"{max_game['chaos_score']:.2f}",
                  delta=f"{max_game['home'][:10]} vs {max_game['away'][:10]}")

    with col3:
        upset_count = df['is_upset'].sum()
        upset_pct = upset_count / len(df) * 100 if len(df) > 0 else 0
        st.metric("😱 Upset Games", f"{upset_pct:.1f}%",
                  delta=f"{upset_count} potential upsets")

    with col4:
        most_vol_week = df.groupby("week")["chaos_score"].std().idxmax()
        st.metric("🌪️ Most Volatile Week", f"Week {most_vol_week}",
                  delta=f"σ = {df[df['week'] == most_vol_week]['chaos_score'].std():.2f}")
