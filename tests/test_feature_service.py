import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from services.feature_service import FeatureService, NUM_FEATURES, FEATURE_NAMES


# A small fake corpus to test against
CORPUS = [
    ["machine", "learning", "ranking", "algorithms"],
    ["information", "retrieval", "bm25", "ranking"],
    ["deep", "learning", "neural", "networks"],
    ["search", "engine", "ranking", "relevance"],
    ["natural", "language", "processing", "nlp"],
]

QUERY = ["machine", "learning", "ranking"]


def test_feature_vector_shape():
    svc = FeatureService(CORPUS)
    vec = svc.extract_features(QUERY, doc_index=0, bm25_score=3.5)
    assert vec.shape == (NUM_FEATURES,)
    print(f"  ✓ feature vector shape is correct: {vec.shape}")


def test_feature_vector_values():
    svc = FeatureService(CORPUS)
    vec = svc.extract_features(QUERY, doc_index=0, bm25_score=3.5)
    assert vec[0] == 3.5                        # bm25 score passed through correctly
    assert vec[1] == 1.0                        # all query words found in doc
    assert vec[2] == float(len(CORPUS[0]))      # doc length correct
    assert vec[3] == float(len(QUERY))          # query length correct
    print(f"  ✓ feature values are correct")


def test_feature_matrix_shape():
    svc = FeatureService(CORPUS)
    candidates = [(0, 3.5), (1, 2.1), (3, 1.8)]
    X, doc_indices = svc.build_feature_matrix(QUERY, candidates)
    assert X.shape == (3, NUM_FEATURES)
    assert doc_indices == [0, 1, 3]
    print(f"  ✓ feature matrix shape is correct: {X.shape}")


def test_empty_candidates():
    svc = FeatureService(CORPUS)
    X, doc_indices = svc.build_feature_matrix(QUERY, [])
    assert X.shape == (0, NUM_FEATURES)
    assert doc_indices == []
    print("  ✓ empty candidates handled correctly")


def test_zero_coverage():
    svc = FeatureService(CORPUS)
    # query words don't exist in doc[4] at all
    vec = svc.extract_features(["machine", "learning"], doc_index=4, bm25_score=0.0)
    assert vec[1] == 0.0
    print("  ✓ zero coverage works correctly")


if __name__ == "__main__":
    tests = [
        test_feature_vector_shape,
        test_feature_vector_values,
        test_feature_matrix_shape,
        test_empty_candidates,
        test_zero_coverage,
    ]

    print(f"\nRunning {len(tests)} tests for FeatureService...\n")
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