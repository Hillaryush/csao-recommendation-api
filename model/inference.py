import lightgbm as lgb
import pandas as pd
import numpy as np

# Load trained model
model = lgb.Booster(model_file="model/csao_model.txt")

FEATURE_COLUMNS = [
    "user_id",
    "restaurant_id",
    "cart_total_value",
    "cart_item_count",
    "hour",
    "day_of_week",
    "item_price",
    "category"
]


# ---------------------------
# Popularity Fallback
# ---------------------------
def popularity_ranking(df):
    df = df.copy()
    df["score"] = 1 / df["item_price"]  # cheaper items ranked higher
    return df.sort_values("score", ascending=False)


# ---------------------------
# Diversity Re-ranking
# ---------------------------
def diversity_rerank(df, max_per_category=2):
    selected = []
    category_count = {}

    for _, row in df.iterrows():
        cat = row["category"]

        if category_count.get(cat, 0) < max_per_category:
            selected.append(row)
            category_count[cat] = category_count.get(cat, 0) + 1

    return pd.DataFrame(selected)


# ---------------------------
# Main Prediction Function
# ---------------------------
def predict_scores(feature_df: pd.DataFrame):

    feature_df = feature_df.copy()

    # ------------------------
    # Cold Start
    # ------------------------
    if feature_df["user_id"].iloc[0] > 9000:
        print("Cold start: using popularity fallback")
        ranked = popularity_ranking(feature_df)
        return diversity_rerank(ranked)

    # ------------------------
    # ML Prediction
    # ------------------------
    model_input = feature_df[FEATURE_COLUMNS]
    probs = model.predict(model_input)

    feature_df["probability"] = probs

    # ------------------------
    # Revenue-aware scoring
    # ------------------------
    feature_df["expected_revenue"] = (
        feature_df["probability"] * feature_df["item_price"]
    )

    # IMPORTANT: Keep 'score' for API compatibility
    feature_df["score"] = feature_df["expected_revenue"]

    # Rank by expected revenue
    ranked = feature_df.sort_values("expected_revenue", ascending=False)

    diversified = diversity_rerank(ranked)

    return diversified