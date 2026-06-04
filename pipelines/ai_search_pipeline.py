"""
ai_search_pipeline.py
---------------------
Standalone AI search pipeline script.

This is the orchestration layer that chains together
all 3 services into one clean search flow:

    User Query
        ↓
    BM25 Retrieval
        ↓
    Feature Engineering
        ↓
    LambdaMART Ranking
        ↓
    Final Ranked Results

Run with:
    python -m pipelines.ai_search_pipeline
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from configs.settings import DATA_PATH, MODEL_PATH, BM25_TOP_K, SEARCH_TOP_K
from services.pipeline import SearchPipeline
from deep_translator import GoogleTranslator


def translate_to_english(text: str) -> str:
    """Translate Hindi text to English for display."""
    try:
        return GoogleTranslator(source="hi", target="en").translate(str(text))
    except:
        return text


def ai_search(query: str, top_k: int = SEARCH_TOP_K) -> list:
    """
    Complete AI search function.

    Args:
        query:  English search query (e.g. "red cap")
        top_k:  Number of final results to return

    Returns:
        List of dicts with product info + AI relevance score
    """

    # --------------------------------------------------
    # Step 1 — Initialize the pipeline
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("  AI SEARCH PIPELINE")
    print("=" * 60)

    print(f"\n  Query:  '{query}'")
    print(f"  Top-K:  {top_k}")

    print("\n  Loading pipeline...")
    pipeline = SearchPipeline(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        bm25_top_k=BM25_TOP_K,
    )
    df = pipeline.bm25.df
    print("  Pipeline ready!")

    # --------------------------------------------------
    # Step 2 — Run the search
    # --------------------------------------------------
    print("\n  Running search...")
    query_tokens = query.lower().split()
    ranked = pipeline.search(query_tokens, top_k=top_k)

    if not ranked:
        print("  No results found.")
        return []

    # --------------------------------------------------
    # Step 3 — Format results with product details
    # --------------------------------------------------
    print(f"\n  Found {len(ranked)} results. Formatting...\n")

    results = []
    for rank, (doc_idx, score) in enumerate(ranked, start=1):
        if score <= 0:
            continue

        row = df.iloc[doc_idx]

        title_hi = str(row.get("title", f"Product {doc_idx}"))
        title_en = translate_to_english(title_hi)
        category_hi = str(row.get("category", ""))
        category_en = translate_to_english(category_hi)

        result = {
            "rank": rank,
            "product_index": int(doc_idx),
            "title": title_en,
            "title_original": title_hi,
            "category": category_en,
            "price": float(row.get("price", 0) or 0),
            "rating": float(row.get("rating", 0) or 0),
            "review_count": int(row.get("review_count", 0) or 0),
            "ai_score": round(score, 4),
        }
        results.append(result)

    return results


def display_results(results: list):
    """Pretty-print search results to the console."""

    if not results:
        print("  No results to display.")
        return

    print("-" * 60)
    for r in results:
        print(f"  #{r['rank']}  {r['title']}")
        print(f"       Category: {r['category']}")
        print(f"       Price: ₹{r['price']:.2f}  |  Rating: {r['rating']:.1f}  |  Reviews: {r['review_count']}")
        print(f"       AI Score: {r['ai_score']}")
        print()
    print("-" * 60)
    print(f"  Total: {len(results)} results")
    print("=" * 60 + "\n")


# --------------------------------------------------
# Main — interactive search
# --------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  AI-POWERED SEARCH RANKING SYSTEM")
    print("  Type a product query and press Enter")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        query = input("\n  🔍 Search: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 👋\n")
            break

        if not query:
            print("  Please enter a search term.")
            continue

        results = ai_search(query, top_k=10)
        display_results(results)
