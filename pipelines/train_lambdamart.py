"""
train_lambdamart.py
-------------------
Trains a LambdaMART model using LightGBM on your training data
and saves it to the models folder so the ranking service can use it.

Run with:
    python -m pipelines.train_lambdamart
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMRanker
from configs.settings import TRAINING_DATA_PATH, MODEL_PATH, FEATURE_COLUMNS, MODEL_PARAMS


# ----------------------------------------------------------------
# Step 1 — Load training data
# ----------------------------------------------------------------

print("\n" + "="*50)
print("  LAMBDAMART TRAINING STARTED")
print("="*50)

print("\nStep 1 — Loading training data...\n")

train_df = pd.read_csv(TRAINING_DATA_PATH)

print(f"  Loaded {len(train_df)} rows")
print(f"  Queries: {train_df['query'].nunique()} unique queries")
print(f"  Columns: {list(train_df.columns)}")
print("\nSample data:")
print(train_df.head())


# ----------------------------------------------------------------
# Step 2 — Build feature matrix and labels
# ----------------------------------------------------------------

print("\nStep 2 — Building feature matrix...\n")

X = train_df[FEATURE_COLUMNS]
y = train_df["relevance"]

print(f"  Feature matrix shape: {X.shape}")
print(f"  Labels shape: {y.shape}")
print(f"  Relevance range: {y.min()} to {y.max()}")


# ----------------------------------------------------------------
# Step 3 — Create query groups
# ----------------------------------------------------------------

print("\nStep 3 — Creating query groups...\n")

group = (
    train_df
    .groupby("query")
    .size()
    .to_numpy()
)

print(f"  {len(group)} query groups")
print(f"  Docs per query: {group}")


# ----------------------------------------------------------------
# Step 4 — Train the LambdaMART model
# ----------------------------------------------------------------

print("\nStep 4 — Training LambdaMART model...\n")

ranker = LGBMRanker(**MODEL_PARAMS)

ranker.fit(
    X,
    y,
    group=group,
)

print("\nModel trained successfully!")


# ----------------------------------------------------------------
# Step 5 — Preview predictions on training data
# ----------------------------------------------------------------

print("\nStep 5 — Previewing predictions...\n")

predictions = ranker.predict(X)
train_df["predicted_score"] = predictions

ranked_results = train_df.sort_values(by="predicted_score", ascending=False)

print("Top 10 AI-ranked results:")
print(
    ranked_results[
        ["query", "bm25_score", "rating", "review_count", "predicted_score"]
    ].head(10)
)


# ----------------------------------------------------------------
# Step 6 — Save the model
# ----------------------------------------------------------------

print("\nStep 6 — Saving model...\n")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(ranker, f)

print(f"  Model saved to: {MODEL_PATH}")

print("\n" + "="*50)
print("  TRAINING COMPLETE")
print("="*50 + "\n")