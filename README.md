# 🚀 CSAO – AI Revenue-Optimized Recommendation System

🔥 **Live API:** https://csao-recommendation-api.onrender.com/

---

## 🧠 Overview

CSAO is a **production-grade recommendation system** designed to maximize **business revenue**, not just user engagement.

Unlike traditional recommenders that optimize click probability, CSAO ranks items using:

👉 **Expected Revenue = Conversion Probability × Item Price**

This aligns recommendations directly with **real-world monetization goals**.

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **ML Model:** LightGBM
* **Database:** PostgreSQL (Render Cloud)
* **ORM:** SQLAlchemy
* **Language:** Python

---

## 🏗️ System Architecture

User Request
→ FastAPI
→ Feature Engineering
→ LightGBM Model
→ Revenue Scoring
→ Diversity Re-Ranking
→ PostgreSQL Logging
→ Analytics Endpoint

---

## ✨ Key Features

### 💰 Revenue-Aware Ranking

Ranks items by **expected revenue**, not just probability.

### 🎯 Diversity Re-Ranking

Prevents over-recommendation of same-category items.

### 🧊 Cold Start Handling

Fallback mechanism using **popularity-based ranking**.

### 📊 Production Logging

Stores all recommendations in PostgreSQL for analysis.

### 📈 Analytics Endpoint

Tracks:

* Total recommendations
* Average expected revenue
* Top-performing items

---

## 📡 API Endpoints

### 🔹 Health Check

```http
GET /
```

Response:

```json
{ "status": "API running 🚀" }
```

---

### 🔹 Get Recommendations

```http
POST /recommend
```

Response:

```json
{
  "recommendations": [
    { "item_id": 2, "score": 212.4 },
    { "item_id": 3, "score": 175.2 }
  ],
  "latency_ms": 5.32
}
```

---

### 🔹 Analytics

```http
GET /analytics
```

---

## 📊 Core Business Logic

### 🚀 Revenue Optimization

Traditional:

```
Rank = P(click)
```

CSAO:

```
Rank = P(conversion) × Price
```

---

### 🎯 Diversity Constraint

* Limits items per category
* Prevents recommendation fatigue
* Improves exploration

---

### 🧊 Cold Start Strategy

* Handles unseen users
* Uses popularity-based fallback

---

## 🚀 Deployment

* Hosted on Render Cloud
* PostgreSQL integration
* Production-ready API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 10000
```

---

## 📌 Why This Project Stands Out

* Combines **ML + backend + system design**
* Focuses on **business metrics (revenue)**
* Simulates **real-world production systems**

---

## 👨‍💻 Author

**Ayush Patel**
Backend & ML Developer
