def sentiment_analysis_tab(df):
    st.subheader("💬 Fan Sentiment Analysis")

    if 'sentiment_score' not in df.columns or 'sentiment_label' not in df.columns:
        st.info("Sentiment data loaded but missing required columns (sentiment_score, sentiment_label)")
        return

    col1, col2 = st.columns(2)
    with col1:
        plot_sentiment_distribution(df)
        show_conference_sentiment(df)

    with col2:
        plot_chaos_sentiment_correlation(df)

    show_most_discussed_games(df)
    plot_sentiment_trends(df)


def plot_sentiment_distribution(df):
    sentiment_counts = df['sentiment_label'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Overall Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map={"Positive": "#43e97b", "Neutral": "#4facfe", "Negative": "#f5576c"},
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)


def show_conference_sentiment(df):
    conf_sentiment = df.groupby('home_conference')['sentiment_score'].mean().sort_values(
        ascending=False).reset_index()
    conf_sentiment.columns = ['Conference', 'Avg Sentiment']
    st.write("**Conference Fan Sentiment**")
    st.dataframe(conf_sentiment, hide_index=True, use_container_width=True)


def plot_chaos_sentiment_correlation(df):
    fig = px.scatter(
        df,
        x='chaos_score',
        y='sentiment_score',
        color='sentiment_label',
        size='comment_count' if 'comment_count' in df.columns else None,
        hover_data=['home', 'away', 'week'],
        title="Chaos vs Fan Sentiment",
        labels={'chaos_score': 'Chaos Score', 'sentiment_score': 'Sentiment Score'},
        color_discrete_map={"Positive": "#43e97b", "Neutral": "#4facfe", "Negative": "#f5576c"}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    correlation = df[['chaos_score', 'sentiment_score']].corr().iloc[0, 1]
    st.metric("Chaos-Sentiment Correlation", f"{correlation:.3f}")
    if correlation > 0.3:
        st.info("🔥 Strong positive correlation: More chaotic games = More excited fans!")
    elif correlation < -0.3:
        st.info("😴 Negative correlation: Chaotic games = Mixed fan reactions")
    else:
        st.info("🤷 Weak correlation: Sentiment varies independently of chaos")


def show_most_discussed_games(df):
    st.subheader("🗣️ Most Discussed Games")
    if 'comment_count' not in df.columns:
        return
    most_discussed = df.nlargest(10, 'comment_count')[['home', 'away', 'week', 'chaos_score', 'sentiment_score', 'sentiment_label', 'comment_count']]
    for _, game in most_discussed.iterrows():
        sentiment_class = "sentiment-positive" if game['sentiment_label'] == "Positive" else "sentiment-negative" if game['sentiment_label'] == "Negative" else "sentiment-neutral"
        st.markdown(f"""
        <div class='{sentiment_class}'>
            <strong>{game['home']} vs {game['away']}</strong> (Week {int(game['week'])})<br>
            <small>💬 {int(game['comment_count'])} comments | Chaos: {game['chaos_score']:.2f} | Sentiment: {game['sentiment_score']:.2f}</small>
        </div>
        """, unsafe_allow_html=True)


def plot_sentiment_trends(df):
    st.subheader("📊 Sentiment Trends Over Season")
    weekly_sentiment = df.groupby('week').agg({
        'sentiment_score': 'mean',
        'chaos_score': 'mean',
        'comment_count': 'sum' if 'comment_count' in df.columns else 'count'
    }).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_sentiment['week'], y=weekly_sentiment['sentiment_score'], mode='lines+markers', name='Avg Sentiment', line=dict(color='#43e97b', width=3), yaxis='y'))
    fig.add_trace(go.Scatter(x=weekly_sentiment['week'], y=weekly_sentiment['chaos_score'], mode='lines+markers', name='Avg Chaos', line=dict(color='#f5576c', width=3), yaxis='y2'))

    fig.update_layout(title="Weekly Sentiment vs Chaos Trends", xaxis_title="Week", yaxis=dict(title="Sentiment Score", side='left'), yaxis2=dict(title="Chaos Score", overlaying='y', side='right'), hovermode='x unified', height=400)
    st.plotly_chart(fig, use_container_width=True)
