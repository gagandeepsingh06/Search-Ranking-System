"""
clean_data.py
-------------
Cleans the product dataset for better search quality.

What this does:
    1. Removes products with no rating AND no reviews
    2. Removes duplicate titles
    3. Fills missing prices with category average
    4. Normalizes column names
    5. Saves a clean version

Run with:
    python -m pipelines.clean_data
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from configs.settings import DATA_PATH

OUTPUT_PATH = "data/processed/cleaned_products_v2.csv"

print("\n" + "="*50)
print("  DATA CLEANING STARTED")
print("="*50)

# ----------------------------------------------------------------
# Step 1 — Load
# ----------------------------------------------------------------
print("\nStep 1 — Loading dataset...\n")
df = pd.read_csv(DATA_PATH)
print(f"  Original: {len(df)} rows")
print(f"  Categories:\n{df['category'].value_counts()}")

# ----------------------------------------------------------------
# Step 2 — Remove products with no rating AND no reviews
# ----------------------------------------------------------------
print("\nStep 2 — Removing unrated products...\n")
df = df[(df['rating'] > 0) | (df['review_count'] > 0)]
print(f"  After removing unrated: {len(df)} rows")

# ----------------------------------------------------------------
# Step 3 — Remove duplicate titles
# ----------------------------------------------------------------
print("\nStep 3 — Removing duplicate titles...\n")
df = df.drop_duplicates(subset=["title"])
print(f"  After removing duplicates: {len(df)} rows")

# ----------------------------------------------------------------
# Step 4 — Fill missing prices with category average
# ----------------------------------------------------------------
print("\nStep 4 — Fixing missing prices...\n")
df["price"] = df["price"].replace(0, np.nan)
df["price"] = df.groupby("category")["price"].transform(
    lambda x: x.fillna(x.mean())
)
df["price"] = df["price"].fillna(df["price"].mean())
df["price"] = df["price"].round(2)
print(f"  Prices fixed. Missing: {df['price'].isna().sum()}")

# ----------------------------------------------------------------
# Step 5 — Normalize ratings to 0-5
# ----------------------------------------------------------------
print("\nStep 5 — Normalizing ratings...\n")
df["rating"] = df["rating"].clip(0, 5)
print(f"  Rating range: {df['rating'].min()} to {df['rating'].max()}")

# ----------------------------------------------------------------
# Step 6 — Add quality score column
# ----------------------------------------------------------------
print("\nStep 6 — Adding quality score...\n")

# Normalize review_count to 0-1
max_reviews = df["review_count"].max()
df["review_norm"] = df["review_count"] / max_reviews if max_reviews > 0 else 0

# Quality score = 60% rating + 40% review popularity
df["quality_score"] = (
    0.6 * (df["rating"] / 5.0) +
    0.4 * df["review_norm"]
).round(4)

print(f"  Quality score range: {df['quality_score'].min()} to {df['quality_score'].max()}")

# ----------------------------------------------------------------
# Step 7 — Show category breakdown
# ----------------------------------------------------------------
print("\nStep 7 — Final category breakdown:\n")
print(df["category"].value_counts())

# ----------------------------------------------------------------
# Step 8 — Save
# ----------------------------------------------------------------
print(f"\nStep 8 — Saving clean dataset...\n")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df = df.drop(columns=["review_norm"])  # drop temp column
df.to_csv(OUTPUT_PATH, index=False)
print(f"  Saved to: {OUTPUT_PATH}")
print(f"  Final row count: {len(df)}")

print("\n" + "="*50)
print("  CLEANING COMPLETE")
print("="*50 + "\n")