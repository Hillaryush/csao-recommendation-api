import lightgbm as lgb
import pandas as pd
import numpy as np
from sqlalchemy import text
from db import SessionLocal


# ---------------------------
# Load Trained Model
# ---------------------------
model = lgb.Booster(model_file="model/csao_model.txt")


# ---------------------------
# Feature Columns
# ---------------------------
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
# Popularity Fallback (Cold Start)
# ---------------------------
def popularity_ranking(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score"] = 1 / df["item_price"]
    return df.sort_values("score", ascending=False)


# ---------------------------
# Diversity Re-ranking
# ---------------------------
def diversity_rerank(df: pd.DataFrame, max_per_category: int = 2) -> pd.DataFrame:
    selected = []
    category_count = {}

    for _, row in df.iterrows():
        cat = row["category"]

        if category_count.get(cat, 0) < max_per_category:
            selected.append(row)
            category_count[cat] = category_count.get(cat, 0) + 1

    return pd.DataFrame(selected)


# ---------------------------
# Log Recommendations to DB
# ---------------------------
def log_recommendations(df: pd.DataFrame):
    db = SessionLocal()

    for _, row in df.iterrows():
        db.execute(
            text("""
                INSERT INTO recommendations_log
                (user_id, restaurant_id, item_id, probability, expected_revenue)
                VALUES (:user_id, :restaurant_id, :item_id, :probability, :expected_revenue)
            """),
            {
                "user_id": int(row["user_id"]),
                "restaurant_id": int(row["restaurant_id"]),
                "item_id": int(row.get("item_id", 0)),
                "probability": float(row.get("probability", 0)),
                "expected_revenue": float(row.get("expected_revenue", 0))
            }
        )

    db.commit()
    db.close()


# ---------------------------
# Main Prediction Function
# ---------------------------
def predict_scores(feature_df: pd.DataFrame) -> pd.DataFrame:

    feature_df = feature_df.copy()

    # ------------------------
    # Cold Start Handling
    # ------------------------
    if feature_df["user_id"].iloc[0] > 9000:
        ranked = popularity_ranking(feature_df)
        diversified = diversity_rerank(ranked)
        log_recommendations(diversified)
        return diversified

    # ------------------------
    # ML Prediction
    # ------------------------
    model_input = feature_df[FEATURE_COLUMNS]
    probabilities = model.predict(model_input)

    feature_df["probability"] = probabilities

    # ------------------------
    # Revenue-Aware Scoring
    # ------------------------
    feature_df["expected_revenue"] = (
        feature_df["probability"] * feature_df["item_price"]
    )

    feature_df["score"] = feature_df["expected_revenue"]

    ranked = feature_df.sort_values("expected_revenue", ascending=False)

    diversified = diversity_rerank(ranked)

    # ------------------------
    # Log to Database
    # ------------------------
    log_recommendations(diversified)

    return diversified