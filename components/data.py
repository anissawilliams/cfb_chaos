import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("chaos_data.csv")
    if 'chaos_score' not in df.columns:
        df["chaos_score"] = (
            0.4 * df["lead_change_count"] +
            0.3 * df["explosive_play_delta"] +
            0.3 * df["win_prob_volatility"]
        )
    power5 = ['SEC', 'Big Ten', 'ACC', 'Big 12', 'Pac-12']
    return df[df['home_conference'].isin(power5)].copy()

@st.cache_resource
def train_chaos_model(features, target):
    return LinearRegression().fit(features, target)
