"""
settings.py
-----------
All settings for the project in one place.
Instead of hardcoding paths and values everywhere,
we import them from here.

If you move a file or change a setting,
you only need to change it here — not in every file.
"""

import os

# ----------------------------------------------------------------
# Base path — root of your project
# Everything else is relative to this
# ----------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------
# Data paths
# ----------------------------------------------------------------

# Your main product dataset (used by BM25Service)
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_products_v2.csv")

# Your training data (used by the training script)
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "training_data_v2.csv")


# ----------------------------------------------------------------
# Model paths
# ----------------------------------------------------------------

# Where the trained LambdaMART model is saved/loaded from
MODEL_PATH = os.path.join(BASE_DIR, "models", "lambdamart_ranker.pkl")


# ----------------------------------------------------------------
# Search settings
# ----------------------------------------------------------------

# How many BM25 candidates to retrieve before re-ranking
BM25_TOP_K = 50

# How many final results to return to the user
SEARCH_TOP_K = 10


# ----------------------------------------------------------------
# Training settings
# ----------------------------------------------------------------

# Features used to train the LambdaMART model
FEATURE_COLUMNS = [
    "bm25_score",
    "rating",
    "review_count",
    "price",
    "is_best_seller",
    "bought_last_month",
]

# LambdaMART model parameters (regularized to prevent overfitting)
MODEL_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "n_estimators": 15,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_child_samples": 10,
    "reg_lambda": 1.0,
}


# ----------------------------------------------------------------
# API settings
# ----------------------------------------------------------------

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "AI Based Search Ranking System"
API_VERSION = "1.0.0"