from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import time

from model.inference import predict_scores
from db import SessionLocal
from sqlalchemy import text


# ---------------------------------
# FastAPI App
# ---------------------------------
app = FastAPI(
    title="CSAO Recommendation API",
    version="1.0"
)


# ---------------------------------
# Request Schemas
# ---------------------------------
class CandidateItem(BaseModel):
    item_id: int
    item_price: float
    category: int


class RecommendationRequest(BaseModel):
    user_id: int
    restaurant_id: int
    cart_total_value: float
    cart_item_count: int
    hour: int
    day_of_week: int
    candidate_items: List[CandidateItem]


# ---------------------------------
# Health Check
# ---------------------------------
@app.get("/")
def health_check():
    return {"status": "API running 🚀"}


# ---------------------------------
# Recommendation Endpoint
# ---------------------------------
@app.post("/recommend")
def recommend(request: RecommendationRequest):

    try:
        start_time = time.time()

        # Convert candidate items to DataFrame
        df = pd.DataFrame([item.dict() for item in request.candidate_items])

        # Add user/session features
        df["user_id"] = request.user_id
        df["restaurant_id"] = request.restaurant_id
        df["cart_total_value"] = request.cart_total_value
        df["cart_item_count"] = request.cart_item_count
        df["hour"] = request.hour
        df["day_of_week"] = request.day_of_week

        # Run model inference
        ranked_df = predict_scores(df)

        # Top 8 recommendations
        top_recommendations = ranked_df.head(8)[["item_id", "score"]]

        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "recommendations": top_recommendations.to_dict(orient="records"),
            "latency_ms": latency
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------
# Analytics / Monitoring Endpoint
# ---------------------------------
@app.get("/analytics")
def analytics():

    db = SessionLocal()

    try:
        total_recs = db.execute(
            text("SELECT COUNT(*) FROM recommendations_log")
        ).scalar()

        avg_revenue = db.execute(
            text("SELECT AVG(expected_revenue) FROM recommendations_log")
        ).scalar()

        top_item = db.execute(
            text("""
                SELECT item_id, COUNT(*) as freq
                FROM recommendations_log
                GROUP BY item_id
                ORDER BY freq DESC
                LIMIT 1
            """)
        ).fetchone()

        top_restaurant = db.execute(
            text("""
                SELECT restaurant_id, COUNT(*) as freq
                FROM recommendations_log
                GROUP BY restaurant_id
                ORDER BY freq DESC
                LIMIT 1
            """)
        ).fetchone()

        total_users = db.execute(
            text("SELECT COUNT(DISTINCT user_id) FROM recommendations_log")
        ).scalar()

        return {
            "total_recommendations": total_recs or 0,
            "average_expected_revenue": float(avg_revenue) if avg_revenue else 0,
            "top_item": top_item[0] if top_item else None,
            "top_restaurant": top_restaurant[0] if top_restaurant else None,
            "unique_users_served": total_users or 0
        }

    finally:
        db.close()