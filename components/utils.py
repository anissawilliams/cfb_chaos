import pandas as pd

# -----------------------------
# Chaos Score Calculation
# -----------------------------
def calculate_chaos_score(df: pd.DataFrame,
                          lead_weight: float = 0.4,
                          explosive_weight: float = 0.3,
                          volatility_weight: float = 0.3) -> pd.DataFrame:
    """
    Adds a chaos_score column to the dataframe using weighted components.
    Default weights: 0.4 lead changes, 0.3 explosive plays, 0.3 volatility.
    """
    total = lead_weight + explosive_weight + volatility_weight
    lead_weight /= total
    explosive_weight /= total
    volatility_weight /= total

    df["chaos_score"] = (
        lead_weight * df["lead_change_count"] +
        explosive_weight * df["explosive_play_delta"] +
        volatility_weight * df["win_prob_volatility"]
    )
    return df


# -----------------------------
# Upset Detection
# -----------------------------
def add_upset_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'is_upset' column to the dataframe.
    If rankings exist, uses ranked-team logic. Otherwise, uses chaos quantile threshold.
    """
    has_rankings = 'home_rank' in df.columns and 'away_rank' in df.columns

    if has_rankings:
        df['is_upset'] = (
            ((df['home_rank'].notna()) | (df['away_rank'].notna())) &
            (df['chaos_score'] > df['chaos_score'].quantile(0.7))
        )
    else:
        df['is_upset'] = df['chaos_score'] > df['chaos_score'].quantile(0.75)

    return df


# -----------------------------
# Chaos Level Categorization
# -----------------------------
def add_chaos_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a categorical chaos_level column (Low, Medium, High) based on chaos_score quantiles.
    """
    df["chaos_level"] = pd.cut(
        df["chaos_score"],
        bins=[0,
              df["chaos_score"].quantile(0.33),
              df["chaos_score"].quantile(0.67),
              df["chaos_score"].max()],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )
    return df


# -----------------------------
# Shared Color Maps
# -----------------------------
def get_color_map():
    return {
        "Low": "#43e97b",
        "Medium": "#4facfe",
        "High": "#f5576c",
        "Positive": "#43e97b",
        "Neutral": "#4facfe",
        "Negative": "#f5576c"
    }

def get_hover_columns():
    return [
        "home",
        "away",
        "week",
        "lead_change_count",
        "explosive_play_delta",
        "win_prob_volatility",
        "chaos_score"
    ]

def get_conference_colors():
    """
    Returns a dictionary mapping conferences to distinctive colors.
    """
    return {
        "SEC": "#1f77b4",       # deep blue
        "Big Ten": "#ff7f0e",   # orange
        "ACC": "#2ca02c",       # green
        "Big 12": "#d62728",    # red
        "Pac-12": "#9467bd",    # purple
        "Group of 5": "#8c564b",# brown
        "Independent": "#e377c2" # pink
    }
