import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from ranking_metrics import precision_at_k, recall_at_k, ndcg_at_k

# ---------------------------
# 1. Generate Synthetic Data
# ---------------------------

np.random.seed(42)

n_samples = 5000

data = pd.DataFrame({
    "user_id": np.random.randint(1, 1000, n_samples),
    "restaurant_id": np.random.randint(1, 200, n_samples),
    "cart_total_value": np.random.uniform(100, 1000, n_samples),
    "cart_item_count": np.random.randint(1, 5, n_samples),
    "hour": np.random.randint(0, 24, n_samples),
    "day_of_week": np.random.randint(0, 7, n_samples),
    "item_price": np.random.uniform(50, 300, n_samples),
    "category": np.random.randint(1, 4, n_samples),
})

# Simulated target (probability based on some logic)
data["label"] = (
    (data["item_price"] < 200).astype(int) +
    (data["cart_item_count"] < 3).astype(int) +
    (data["hour"] > 18).astype(int)
)

data["label"] = (data["label"] > 1).astype(int)

# ---------------------------
# 2. Train-Test Split
# ---------------------------

X = data.drop(columns=["label"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 3. Train LightGBM
# ---------------------------

model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31
)

model.fit(X_train, y_train)

# ---------------------------
# 4. Evaluate
# ---------------------------

preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)

print("AUC Score:", auc)

# ---------------------------
# 5. Save Model Properly
# ---------------------------

model.booster_.save_model("model/csao_model.txt")

print("Model saved successfully!")

# ---------------------------
# 6. Evaluate with Ranking Metrics
# ---------------------------

# Convert test data to numpy arrays
y_scores = preds
y_true = y_test.values

print("Precision@8:", precision_at_k(y_true, y_scores, k=8))
print("Recall@8:", recall_at_k(y_true, y_scores, k=8))
print("NDCG@8:", ndcg_at_k(y_true, y_scores, k=8))
# ---------------------------
# Business Impact Simulation
# ---------------------------

# Assume:
avg_addon_price = 120  # average add-on price
baseline_acceptance_rate = 0.20  # 20% baseline without model

# Model acceptance = top-K precision (proxy)
model_acceptance_rate = precision_at_k(y_true, y_scores, k=8)

# AOV Lift
aov_lift = (model_acceptance_rate - baseline_acceptance_rate) * avg_addon_price

print("\n--- Business Impact Simulation ---")
print("Baseline Acceptance Rate:", baseline_acceptance_rate)
print("Model Acceptance Rate:", model_acceptance_rate)
print("Estimated AOV Lift per Order: ₹", round(aov_lift, 2))

# Revenue impact simulation
daily_orders = 50000  # example scale
daily_revenue_increase = aov_lift * daily_orders

print("Estimated Daily Revenue Lift: ₹", round(daily_revenue_increase, 2))
# ---------------------------
# Segment-wise Analysis
# ---------------------------

print("\n--- Segment-wise Business Impact ---")

test_df = X_test.copy()
test_df["label"] = y_test.values
test_df["score"] = y_scores

# Time segmentation
def time_segment(hour):
    if 11 <= hour <= 15:
        return "Lunch"
    elif 18 <= hour <= 22:
        return "Dinner"
    else:
        return "Other"

test_df["time_segment"] = test_df["hour"].apply(time_segment)

# Cart value segmentation
def value_segment(value):
    if value < 300:
        return "Low"
    elif value < 700:
        return "Medium"
    else:
        return "High"

test_df["value_segment"] = test_df["cart_total_value"].apply(value_segment)

# Function to compute segment precision
def segment_precision(df, segment_col, k=8):
    results = {}
    for segment in df[segment_col].unique():
        seg_df = df[df[segment_col] == segment]
        if len(seg_df) < k:
            continue
        
        y_true_seg = seg_df["label"].values
        y_scores_seg = seg_df["score"].values
        
        precision = precision_at_k(y_true_seg, y_scores_seg, k=min(k, len(seg_df)))
        results[segment] = round(precision, 3)
    
    return results

print("Precision@8 by Time Segment:", segment_precision(test_df, "time_segment"))
print("Precision@8 by Cart Value Segment:", segment_precision(test_df, "value_segment"))