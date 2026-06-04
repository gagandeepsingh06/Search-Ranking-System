"""
test_pipeline.py
----------------
End-to-end tests for the full SearchPipeline.

These tests check that all 3 services work together correctly —
from BM25 retrieval all the way through to the final ranked output.

Run with:
    python -m tests.test_pipeline
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import tempfile
from services.pipeline import SearchPipeline


# Create a small temporary CSV file to test against
def create_temp_csv():
    data = {
        "search_text": [
            "machine learning ranking algorithms",
            "information retrieval bm25 ranking",
            "deep learning neural networks",
            "search engine ranking relevance",
            "natural language processing nlp",
        ]
    }
    df = pd.DataFrame(data)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name


DATA_PATH = create_temp_csv()


def test_pipeline_returns_results():
    pipeline = SearchPipeline(DATA_PATH)
    results = pipeline.search(["machine", "learning"], top_k=3)
    assert len(results) > 0
    assert len(results) <= 3
    print(f"  ✓ pipeline returned {len(results)} results")


def test_results_are_sorted_best_first():
    pipeline = SearchPipeline(DATA_PATH)
    results = pipeline.search(["ranking"], top_k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    print("  ✓ results are sorted best-first")


def test_explain_returns_all_features():
    pipeline = SearchPipeline(DATA_PATH)
    explanation = pipeline.explain(["machine", "learning"], doc_index=0, bm25_score=3.5)
    assert "bm25_score" in explanation
    assert len(explanation) == 8
    print(f"  ✓ explain() returned {len(explanation)} features correctly")


def test_empty_query_does_not_crash():
    pipeline = SearchPipeline(DATA_PATH)
    try:
        results = pipeline.search([], top_k=5)
        print(f"  ✓ empty query handled, got {len(results)} results")
    except Exception as e:
        assert False, f"Empty query crashed: {e}"


def test_top_k_is_respected():
    pipeline = SearchPipeline(DATA_PATH, bm25_top_k=10)
    results = pipeline.search(["ranking"], top_k=2)
    assert len(results) <= 2
    print(f"  ✓ top_k=2 respected, got {len(results)} results")


if __name__ == "__main__":
    tests = [
        test_pipeline_returns_results,
        test_results_are_sorted_best_first,
        test_explain_returns_all_features,
        test_empty_query_does_not_crash,
        test_top_k_is_respected,
    ]

    print(f"\nRunning {len(tests)} pipeline tests...\n")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__} FAILED: {e}")

    print(f"\n{'='*40}")
    print(f"  {passed}/{len(tests)} tests passed")
    print(f"{'='*40}\n")