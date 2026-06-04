"""
ranking_service.py
------------------
This file handles the LAST step of search — re-ranking.

BM25 gives us a rough list of candidates.
This service re-ranks them using a weighted scoring formula
that also penalises results from unrelated categories.
"""

import logging
import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Keywords that signal a query is about clothing/accessories
# If query contains these, hair care products get penalised
CLOTHING_KEYWORDS = [
    "cap", "hat", "bag", "shoes", "shirt", "watch", "jacket",
    "dress", "saree", "kurta", "sandals", "wallet", "belt",
    "boots", "trouser", "jeans", "socks", "gloves", "scarf",
]

# Categories that should be penalised for clothing queries
HAIR_CARE_CATEGORY = "बालों की देखभाल"


class RankingService:

    def __init__(self, model_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """
        model_path: path to trained LightGBM ranker (optional).
        df:         product dataframe — used for category-aware ranking.
        """
        self.model = None
        self.model_path = model_path
        self.df = df  # we need this to check product categories

        if model_path:
            self._load_model(model_path)
        else:
            logger.warning("No model loaded. Using weighted scoring formula.")

    def rerank(
        self,
        X: np.ndarray,
        doc_indices: List[int],
        bm25_scores: Optional[List[float]] = None,
        query: Optional[str] = None,  # original query for category boosting
    ) -> List[Tuple[int, float]]:
        """
        Re-rank candidates.
        If self.model is loaded, we use the LightGBM LambdaMART ranker.
        Otherwise, we fall back to a weighted heuristic scoring formula.
        Penalises hair care products when query is about clothing/accessories.
        """

        if not doc_indices:
            return []

        # Check if this query is about clothing/accessories
        is_clothing_query = False
        if query:
            query_lower = query.lower()
            is_clothing_query = any(kw in query_lower for kw in CLOTHING_KEYWORDS)

        # ------------------------------------------------------------
        # SEARCH MODE 1: True ML LambdaMART Ranker
        # ------------------------------------------------------------
        if self.model is not None and self.df is not None:
            try:
                # Build the 6-feature matrix expected by the LightGBM model
                X_ml = []
                for i, doc_idx in enumerate(doc_indices):
                    row = self.df.iloc[doc_idx]
                    
                    # 1. bm25_score
                    bm25_val = float(bm25_scores[i]) if (bm25_scores is not None and i < len(bm25_scores)) else 0.0
                    
                    # 2. rating
                    rating_val = float(row.get("rating", 0.0))
                    if np.isnan(rating_val):
                        rating_val = 0.0
                        
                    # 3. review_count
                    reviews_val = float(row.get("review_count", 0.0))
                    if np.isnan(reviews_val):
                        reviews_val = 0.0
                        
                    # 4. price
                    price_val = float(row.get("price", 0.0))
                    if np.isnan(price_val):
                        price_val = 0.0
                        
                    # 5. is_best_seller
                    bs_val = row.get("isBestSeller", False)
                    is_best = 1.0 if (pd.notna(bs_val) and bool(bs_val)) else 0.0
                    
                    # 6. bought_last_month
                    bought_val = row.get("boughtInLastMonth", 0.0)
                    bought = float(bought_val) if pd.notna(bought_val) else 0.0
                    X_ml.append([bm25_val, rating_val, reviews_val, price_val, is_best, bought])
                
                # Perform model prediction using DataFrame with matching feature names to avoid UserWarning
                X_df = pd.DataFrame(X_ml, columns=[
                    "bm25_score",
                    "rating",
                    "review_count",
                    "price",
                    "is_best_seller",
                    "bought_last_month"
                ])
                predictions = self.model.predict(X_df)
                
                scored = []
                for i, doc_idx in enumerate(doc_indices):
                    pred_score = float(predictions[i])
                    
                    # Penalise hair care products for clothing queries
                    if is_clothing_query:
                        try:
                            category = self.df.iloc[doc_idx]["category"]
                            if category == HAIR_CARE_CATEGORY:
                                pred_score -= 1.0  # Apply negative penalty to ML score
                        except:
                            pass
                    
                    scored.append((doc_idx, round(pred_score, 4)))
                    
                # Sort best first
                ranked = sorted(scored, key=lambda x: x[1], reverse=True)
                logger.info(f"Re-ranked {len(ranked)} candidates using LambdaMART model.")
                return ranked
                
            except Exception as e:
                logger.error(f"Error during model prediction: {e}. Falling back to weighted formula.")
                # Fall back to weighted formula if model prediction fails
                pass

        # ------------------------------------------------------------
        # SEARCH MODE 2: Heuristic Weighted Scoring Fallback
        # ------------------------------------------------------------
        scored = []
        for i, doc_idx in enumerate(doc_indices):
            row = X[i]

            # Feature vector layout of FeatureService:
            # [0] bm25_score, [1] coverage, [4] exact, [5] title, [6] avg_tf
            bm25     = float(row[0])
            coverage = float(row[1])
            exact    = float(row[4])
            title    = float(row[5])
            avg_tf   = float(row[6])

            # Normalize BM25 to 0-1 range
            bm25_norm = bm25 / 10.0 if bm25 > 0 else 0.0

            # Base weighted score
            final_score = (
                0.5 * bm25_norm +
                0.2 * coverage +
                0.1 * exact +
                0.1 * title +
                0.1 * avg_tf
            )

            # Penalise hair care products for clothing queries
            if is_clothing_query and self.df is not None:
                try:
                    category = self.df.iloc[doc_idx]["category"]
                    if category == HAIR_CARE_CATEGORY:
                        final_score *= 0.3  # reduce score by 70%
                except:
                    pass

            scored.append((doc_idx, round(final_score, 4)))

        # Sort best first
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        logger.debug(f"Re-ranked {len(ranked)} candidates using weighted formula.")
        return ranked

    def is_ready(self) -> bool:
        return self.model is not None

    def _load_model(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        ext = os.path.splitext(path)[-1].lower()

        try:
            if ext == ".txt":
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=path)
                logger.info(f"Loaded LightGBM model from {path}")
            elif ext in (".pkl", ".pickle"):
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded pickled model from {path}")
            else:
                raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pkl")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def save_model(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("No model to save — train or load one first.")
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {path}")