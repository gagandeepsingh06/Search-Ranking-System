"""
search_api.py
-------------
FastAPI backend for the AI search ranking system.
Exposes a /search endpoint that the Streamlit UI calls.

Run with:
    uvicorn api.search_api:app --reload
"""

import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from configs.settings import DATA_PATH, MODEL_PATH, BM25_TOP_K, SEARCH_TOP_K
from services.pipeline import SearchPipeline


# ----------------------------------------------------------------
# Start FastAPI app
# ----------------------------------------------------------------

app = FastAPI(
    title="AI Based Search Ranking System",
    version="1.0.0",
)

# Enable CORS for frontend integration
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pipeline once when the server starts
# (not on every request — that would be too slow)
print("\nLoading pipeline...\n")
pipeline = SearchPipeline(
    data_path=DATA_PATH,
    model_path=MODEL_PATH,
    bm25_top_k=BM25_TOP_K,
)
print("Pipeline ready!")


# ----------------------------------------------------------------
# Request and Response models
# ----------------------------------------------------------------

from services.translator_service import TranslatorService

class SearchRequest(BaseModel):
    # What the user typed in the search bar
    query: str
    # How many results to return (default 10)
    top_k: int = SEARCH_TOP_K
    # Toggle between AI search (LambdaMART) and Traditional
    use_ai: bool = True
    # Search mode: "keyword", "semantic", "hybrid"
    search_mode: str = "keyword"


class SearchResult(BaseModel):
    product_index: int
    score: float
    title_original: str
    title_translated: str
    category_original: str
    category_translated: str
    price: float
    rating: float
    review_count: int
    is_best_seller: bool
    bought_last_month: int
    features: dict  # Explanation features


class SearchResponse(BaseModel):
    query: str
    use_ai: bool
    search_mode: str
    results: List[SearchResult]
    total: int


# ----------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------

@app.get("/")
def home():
    """Health check — confirms the API is running."""
    return {"status": "running", "message": "AI Search Ranking API is ready!"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Main search endpoint.
    Accepts a query, runs it through the full pipeline,
    and returns ranked results with rich product details and explainability.
    """
    # Validate and handle empty/whitespace queries gracefully
    clean_query = request.query.strip()
    if not clean_query:
        return SearchResponse(
            query=request.query,
            use_ai=request.use_ai,
            search_mode=request.search_mode,
            results=[],
            total=0,
        )

    # Tokenize the query
    query_tokens = clean_query.lower().split()
    query_string = " ".join(query_tokens)
    mode = request.search_mode.lower()

    # 1. Retrieve ranked indices based on search mode and AI toggle
    if mode == "semantic":
        if request.use_ai:
            ranked = pipeline.search_semantic(query_string, top_k=request.top_k)
        else:
            doc_indices, sim_scores = pipeline.semantic.search(query_string, top_k=request.top_k)
            ranked = [(int(idx), float(score)) for idx, score in zip(doc_indices, sim_scores)]
            
    elif mode == "hybrid":
        if request.use_ai:
            ranked = pipeline.search_hybrid(query_string, top_k=request.top_k)
        else:
            # Replicate hybrid merging without LambdaMART ranking
            top_n_bm25, bm25_scores_all = pipeline.bm25.retrieve_candidates(query_string, top_k=pipeline.bm25_top_k)
            sem_indices, sem_scores = pipeline.semantic.search(query_string, top_k=pipeline.bm25_top_k)
            
            doc_scores = {}
            for rank, idx in enumerate(top_n_bm25):
                idx = int(idx)
                rrf_score = 1.0 / (60 + rank + 1)
                doc_scores[idx] = doc_scores.get(idx, 0.0) + 0.4 * rrf_score
            for rank, (idx, sim) in enumerate(zip(sem_indices, sem_scores)):
                rrf_score = 1.0 / (60 + rank + 1)
                doc_scores[idx] = doc_scores.get(idx, 0.0) + 0.6 * rrf_score
                
            merged = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:request.top_k]
            ranked = [(int(idx), float(score)) for idx, score in merged]
            
    else:  # Default: "keyword" / BM25
        if request.use_ai:
            ranked = pipeline.search(query_tokens, top_k=request.top_k)
        else:
            top_n, scores = pipeline.bm25.retrieve_candidates(query_string, top_k=request.top_k)
            ranked = [(int(idx), float(scores[idx])) for idx in top_n]

    # Get data-frame reference
    df = pipeline.bm25.df

    # 2. Format results with full product metadata and on-the-fly translations
    results = []
    for idx, score in ranked:
        if idx >= len(df):
            continue

        row = df.iloc[idx]

        # Extract features for explanation
        try:
            # Get actual BM25 score of the document for explanation
            hindi_query = TranslatorService.translate_query(query_string)
            tokenized_query = hindi_query.lower().split()
            bm25_score = float(pipeline.bm25.bm25.get_batch_scores(tokenized_query, [idx])[0])
        except:
            bm25_score = float(score)

        features_dict = pipeline.explain(query_tokens, idx, bm25_score)

        # Get values with fallbacks
        title_orig = str(row.get("title", ""))
        cat_orig = str(row.get("category", ""))
        
        # Translate to English for UI
        title_trans = TranslatorService.translate_hi_to_en(title_orig)
        cat_trans = TranslatorService.translate_hi_to_en(cat_orig)

        # Clean values
        try:
            price_val = float(row.get("price", 0.0))
            if pd.isna(price_val):
                price_val = 0.0
        except:
            price_val = 0.0

        try:
            rating_val = float(row.get("rating", 0.0))
            if pd.isna(rating_val):
                rating_val = 0.0
        except:
            rating_val = 0.0

        try:
            reviews_val = int(row.get("review_count", 0))
            if pd.isna(reviews_val):
                reviews_val = 0
        except:
            reviews_val = 0

        # Handle isBestSeller as bool
        best_seller = False
        if "isBestSeller" in row:
            bs_val = row["isBestSeller"]
            if pd.notna(bs_val):
                best_seller = bool(bs_val)

        # Handle boughtInLastMonth
        bought_val = 0
        if "boughtInLastMonth" in row:
            b_val = row["boughtInLastMonth"]
            if pd.notna(b_val):
                try:
                    bought_val = int(b_val)
                except:
                    bought_val = 0

        results.append(
            SearchResult(
                product_index=int(idx),
                score=round(score, 4),
                title_original=title_orig,
                title_translated=title_trans,
                category_original=cat_orig,
                category_translated=cat_trans,
                price=price_val,
                rating=rating_val,
                review_count=reviews_val,
                is_best_seller=best_seller,
                bought_last_month=bought_val,
                features=features_dict,
            )
        )

    return SearchResponse(
        query=request.query,
        use_ai=request.use_ai,
        search_mode=request.search_mode,
        results=results,
        total=len(results),
    )