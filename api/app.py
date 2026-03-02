from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import time

from model.inference import predict_scores

app = FastAPI(
    title="CSAO Recommendation API",
    version="1.0"
)

class RecommendationRequest(BaseModel):
    user_id: int
    restaurant_id: int
    cart_total_value: float
    cart_item_count: int
    hour: int
    day_of_week: int
    candidate_items: list


@app.get("/")
def health_check():
    return {"status": "API running 🚀"}


@app.post("/recommend")
def recommend(request: RecommendationRequest):

    start_time = time.time()

    df = pd.DataFrame(request.candidate_items)

    df["user_id"] = request.user_id
    df["restaurant_id"] = request.restaurant_id
    df["cart_total_value"] = request.cart_total_value
    df["cart_item_count"] = request.cart_item_count
    df["hour"] = request.hour
    df["day_of_week"] = request.day_of_week

    ranked_df = predict_scores(df)

    top_recommendations = ranked_df.head(8)[["item_id", "score"]]

    latency = round((time.time() - start_time) * 1000, 2)

    return {
        "recommendations": top_recommendations.to_dict(orient="records"),
        "latency_ms": latency
    }