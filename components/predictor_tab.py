# predictor_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.data import train_chaos_model  # reuse your cached model

def render_predictor_tab(df, season_avg):
    st.subheader("🔮 Advanced Chaos Predictor")
    st.caption(
        "This predictor demonstrates how chaos metrics can forecast game unpredictability. "
        "By quantifying lead changes, explosive plays, and volatility, we can anticipate "
        "whether a matchup will be calm, fiery, or volcanic."
    )

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

    # --- Historical Matchups & Rivalries ---
    if predict_button:
        st.subheader("📜 Historical Matchup Chaos")
        st.caption(
            "Recurring matchups reveal that chaos has memory. Rivalries that repeat every year "
            "often show consistent chaos patterns, helping us forecast future unpredictability."
        )

        # Build matchup identifier

        df["matchup"] = df.apply(
            lambda row: " vs ".join(sorted([row["home"], row["away"]])),
            axis=1
        )

        matchup_counts = df["matchup"].value_counts()
        recurring_matchups = matchup_counts[matchup_counts >= 3].index

        if len(recurring_matchups) > 0:
            for matchup in recurring_matchups[:3]:
                history = df[df["matchup"] == matchup][["start_date", "chaos_score"]].sort_values("start_date")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history["start_date"],
                    y=history["chaos_score"],
                    mode="lines+markers",
                    name="Historical Chaos",
                    line=dict(color="#4facfe", width=3)
                ))
                fig.add_hline(
                    y=predicted_chaos,
                    line=dict(color="#f5576c", dash="dash"),
                    annotation_text=f"Predicted Chaos {predicted_chaos:.2f}",
                    annotation_position="top left"
                )

                fig.update_layout(
                    title=f"{matchup} Chaos History vs Prediction",
                    xaxis_title="Date",
                    yaxis_title="Chaos Score",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Rivalry chaos leaderboard
            rivalry_df = df[df["matchup"].isin(recurring_matchups)].groupby("matchup").agg(
                avg_chaos=("chaos_score", "mean"),
                games_played=("chaos_score", "count")
            ).reset_index().sort_values("avg_chaos", ascending=False)

            st.write("**🔥 Top Chaotic Rivalries (Avg Chaos):**")
            st.dataframe(
                rivalry_df.head(5),
                column_config={
                    "matchup": st.column_config.TextColumn("Matchup", width="medium"),
                    "avg_chaos": st.column_config.NumberColumn("Avg Chaos", format="%.2f"),
                    "games_played": st.column_config.NumberColumn("Games Played", format="%d")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No recurring matchups with at least 3 games found in dataset.")
