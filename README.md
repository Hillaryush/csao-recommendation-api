🚀 CSAO – AI Revenue Optimized Recommendation System
🔥 Live API
👉 https://csao-recommendation-api.onrender.com/
📌 Project Overview
CSAO is a production-grade AI recommendation system built using LightGBM + FastAPI + PostgreSQL, designed to optimize expected revenue instead of just click-through rate.
Unlike traditional recommenders that rank by probability, this system ranks by:
Expected Revenue = Conversion Probability × Item Price
It also enforces category diversity constraints and handles cold-start users intelligently.
🏗 Architecture
User → FastAPI → Feature Engineering → LightGBM Model →
Revenue Scoring → Diversity Re-ranking →
PostgreSQL Logging → Analytics Endpoint
⚙️ Tech Stack
Backend: FastAPI
ML Model: LightGBM
Database: PostgreSQL (Render Cloud)
ORM: SQLAlchemy
Deployment: Render
Language: Python
✨ Key Features
✅ Revenue-Aware Ranking
Ranks items by expected revenue instead of raw probability.
✅ Diversity Re-Ranking
Prevents over-representation of same category items.
✅ Cold Start Handling
Uses popularity fallback for unseen users.
✅ Production Logging
All recommendations stored in PostgreSQL.
✅ Analytics Endpoint
Real-time system monitoring via /analytics.
📡 API Endpoints
🔹 Health Check
GET /
Response:
{
  "status": "API running 🚀"
}
🔹 Get Recommendations
POST /recommend
Request Example:
{
  "user_id": 1001,
  "restaurant_id": 501,
  "cart_total_value": 800,
  "cart_item_count": 3,
  "hour": 14,
  "day_of_week": 2,
  "candidate_items": [
    {"item_id": 1, "item_price": 120, "category": 1},
    {"item_id": 2, "item_price": 250, "category": 2},
    {"item_id": 3, "item_price": 180, "category": 1}
  ]
}
Response:
{
  "recommendations": [
    {"item_id": 2, "score": 212.4},
    {"item_id": 3, "score": 175.2}
  ],
  "latency_ms": 5.32
}

🔹 Analytics Dashboard
GET /analytics
Response:
{
  "total_recommendations": 120,
  "average_expected_revenue": 138.42,
  "top_item": 3,
  "top_restaurant": 501,
  "unique_users_served": 42
}
📊 Business Logic
Revenue Optimization
Instead of ranking by:
P(click)
We rank by:
P(conversion) × Item Price
This aligns recommendations with business revenue goals.
Diversity Constraint
Limits max items per category to prevent:
Category dominance
Recommendation fatigue
Reduced exploration
Cold Start Strategy
For unseen users:
Fallback to price-based popularity ranking
🚀 Deployment

Hosted on Render
Cloud PostgreSQL integration
Environment-based configuration
Production-ready start command:
uvicorn api.app:app --host 0.0.0.0 --port 10000
