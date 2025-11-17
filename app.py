import streamlit as st
from chaos_predictor import chaos_predictor_tab
from sentiment_analysis import sentiment_analysis_tab

# Assume df and season_avg are defined earlier
tabs = st.tabs(["Overview", "Stats", "Other", "Chaos Predictor", "Sentiment Analysis"])

with tabs[3]:
    chaos_predictor_tab(df, season_avg)

with tabs[4]:
    sentiment_analysis_tab(df)

st.markdown("---")
