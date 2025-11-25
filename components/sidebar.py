import streamlit as st

def chaos_score_tuner():
    st.sidebar.header("⚙️ Chaos Score Tuner")
    lead = st.sidebar.slider("Lead Change Weight", 0.0, 1.0, 0.4)
    explosive = st.sidebar.slider("Explosive Play Weight", 0.0, 1.0, 0.3)
    volatility = st.sidebar.slider("Volatility Weight", 0.0, 1.0, 0.3)

    total = lead + explosive + volatility
    return lead/total, explosive/total, volatility/total
