import streamlit as st
import pandas as pd
import numpy as np
from components.data import train_chaos_model  # reuse your cached model

def render_predictor_tab(df, season_avg):
    st.subheader("🔮 Advanced Chaos Predictor")

    col1, col2, col3 = st.columns([1, 1, 1])

    # --- Input Form ---
    with col1:
        with st.form("predictor_form"):
            st.write("**Input Game Metrics:**")
            lead_input = st.number_input("Lead Changes", min_value=0, max_value=15, value=3)
            explosive_input = st.number_input("Explosive Play Delta", min_value=0, max_value=30, value=8)
            volatility_input = st.slider("Win Prob Volatility", 0.0, 0.6, 0.2)

            predict_button = st.form_submit_button("⚡ Predict Chaos", type="primary")

    # --- Prediction Output ---
    with col2:
        if predict_button:
            features = df[["lead_change_count", "explosive_play_delta", "win_prob_volatility"]]
            target = df["chaos_score"]
            model = train_chaos_model(features, target)

            input_df = pd.DataFrame(
                [[lead_input, explosive_input, volatility_input]],
                columns=["lead_change_count", "explosive_play_delta", "win_prob_volatility"]
            )
            predicted_chaos = model.predict(input_df)[0]

            # Categorize prediction
            if predicted_chaos < df["chaos_score"].quantile(0.33):
                level, color = "Low 😴", "#43e97b"
            elif predicted_chaos < df["chaos_score"].quantile(0.67):
                level, color = "Medium 🔥", "#4facfe"
            else:
                level, color = "High 🌋", "#f5576c"

            st.markdown(f"""
            <div style='background: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
                <h2 style='color: white; margin: 0;'>{predicted_chaos:.2f}</h2>
                <p style='color: white; margin: 5px 0 0 0;'>Predicted Level: {level}</p>
            </div>
            """, unsafe_allow_html=True)

            pct_vs_avg = ((predicted_chaos - season_avg) / season_avg) * 100
            st.info(f"📊 {abs(pct_vs_avg):.1f}% {'above' if pct_vs_avg > 0 else 'below'} season average")

    # --- Similar Historical Games ---
    with col3:
        if predict_button:
            st.write("**🎯 Similar Historical Games:**")
            df_temp = df.copy()
            df_temp["similarity"] = np.abs(df_temp["chaos_score"] - predicted_chaos)
            similar_games = df_temp.nsmallest(5, "similarity")[["home", "away", "chaos_score", "week"]]

            for _, game in similar_games.iterrows():
                st.caption(f"• {game['home']} vs {game['away']} — {game['chaos_score']:.2f} (Wk {int(game['week'])})")
