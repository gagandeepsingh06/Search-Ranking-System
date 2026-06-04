"""
generate_training_data.py
-------------------------
Generates better training data for LambdaMART
from your existing cleaned_products.csv.

Instead of manually labeling, we automatically assign
relevance scores based on rating, review_count, and price.

Run with:
    python -m pipelines.generate_training_data
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from services.bm25_service import BM25Service
from configs.settings import DATA_PATH

# ----------------------------------------------------------------
# Settings
# ----------------------------------------------------------------

OUTPUT_PATH = "data/processed/training_data_v2.csv"

# Queries to generate training data for
QUERIES = [
    "cap", "bag", "shoes", "shirt", "watch",
    "phone", "laptop", "earphones", "wallet", "belt",
    "jacket", "dress", "saree", "kurta", "sandals",
]

TOP_K = 20  # candidates per query


# ----------------------------------------------------------------
# Auto relevance scoring function
# ----------------------------------------------------------------

def compute_relevance(row):
    """
    Automatically assign a relevance score (0-3) based on
    product quality signals — rating, reviews, bestseller.

    3 = highly relevant (great rating + many reviews)
    2 = relevant
    1 = somewhat relevant
    0 = not relevant
    """
    score = 0

    # Rating signal
    rating = row.get("rating", 0) or 0
    if rating >= 4.0:
        score += 2
    elif rating >= 3.0:
        score += 1

    # Review count signal — correct column name is "review_count"
    reviews = row.get("review_count", 0) or 0
    if reviews >= 100:
        score += 1

    # Bestseller signal — correct column name is "isBestSeller"
    is_best = row.get("isBestSeller", False) or False
    if is_best:
        score += 1

    # Cap at 3
    return min(score, 3)


# ----------------------------------------------------------------
# Main script
# ----------------------------------------------------------------

print("\n" + "="*50)
print("  GENERATING TRAINING DATA")
print("="*50)

# Load BM25 service
print("\nLoading BM25 service...\n")
bm25 = BM25Service(DATA_PATH)
df = bm25.df

print(f"Loaded {len(df)} products")
print(f"Generating data for {len(QUERIES)} queries...\n")

all_rows = []

for query in QUERIES:
    print(f"  Processing query: '{query}'")

    # Get top candidates using BM25
    top_n, scores = bm25.retrieve_candidates(query, top_k=TOP_K)

    for idx in top_n:
        row = df.iloc[idx]

        # Get feature values using correct column names
        bm25_score   = float(scores[idx])
        rating       = float(row.get("rating", 0) or 0)
        review_count = float(row.get("review_count", 0) or 0)  # ✓ fixed
        price        = float(row.get("price", 0) or 0)
        is_best      = int(bool(row.get("isBestSeller", False)))  # ✓ fixed
        bought_last  = float(row.get("boughtInLastMonth", 0) or 0)  # ✓ fixed

        # Auto-assign relevance
        relevance = compute_relevance({
            "rating":       rating,
            "review_count": review_count,
            "isBestSeller": is_best,
        })
        
        # Add 15% random click noise to simulate real-world noisy e-commerce interactions
        if np.random.rand() < 0.15:
            relevance = int(np.random.choice([0, 1, 2, 3]))

        all_rows.append({
            "query":             query,
            "product_index":     int(idx),
            "relevance":         relevance,
            "bm25_score":        bm25_score,
            "rating":            rating,
            "review_count":      review_count,
            "price":             price,
            "is_best_seller":    is_best,
            "bought_last_month": bought_last,
        })

# ----------------------------------------------------------------
# Save
# ----------------------------------------------------------------

training_df = pd.DataFrame(all_rows)

print(f"\n  Generated {len(training_df)} training rows")
print(f"  Queries: {training_df['query'].nunique()}")
print(f"  Relevance distribution:\n{training_df['relevance'].value_counts()}")
print(f"\n  Feature check:")
print(training_df[["bm25_score", "rating", "review_count", "is_best_seller"]].describe())

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
training_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n  Saved to: {OUTPUT_PATH}")

print("\n" + "="*50)
print("  DONE")
print("="*50 + "\n")