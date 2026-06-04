import logging
import numpy as np
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "bm25_score",
    "query_term_coverage",
    "doc_length",
    "query_length",
    "exact_match",
    "title_match",
    "avg_term_freq",
    "doc_length_norm",
]

NUM_FEATURES = len(FEATURE_NAMES)


class FeatureService:
    """Extracts lexical and structural features for search query-document pairs."""

    def __init__(self, corpus: List[List[str]]):
        if not corpus:
            raise ValueError("Corpus is empty — please pass at least one document.")

        self.corpus = corpus
        self._mean_doc_length = float(np.mean([len(doc) for doc in corpus]))
        logger.info(f"FeatureService ready | {len(corpus)} docs | avg length = {self._mean_doc_length:.1f} words")

    def extract_features(
        self,
        query_tokens: List[str],
        doc_index: int,
        bm25_score: float,
    ) -> np.ndarray:
        doc_tokens = self.corpus[doc_index]
        doc_words = set(doc_tokens)
        query_words = set(query_tokens)

        bm25 = bm25_score
        coverage = len(query_words & doc_words) / len(query_words) if query_words else 0.0
        doc_len = float(len(doc_tokens))
        query_len = float(len(query_tokens))
        
        exact = 1.0 if (query_tokens and " ".join(query_tokens) in " ".join(doc_tokens)) else 0.0
        title = 1.0 if (doc_tokens and query_tokens and doc_tokens[0] == query_tokens[0]) else 0.0
        
        term_freqs = [doc_tokens.count(word) for word in query_tokens if word in doc_words]
        avg_tf = float(np.mean(term_freqs)) if term_freqs else 0.0
        doc_len_norm = doc_len / self._mean_doc_length if self._mean_doc_length > 0 else 0.0

        return np.array(
            [bm25, coverage, doc_len, query_len, exact, title, avg_tf, doc_len_norm],
            dtype=np.float32,
        )

    def build_feature_matrix(
        self,
        query_tokens: List[str],
        candidates: List[Tuple[int, float]],
    ) -> Tuple[np.ndarray, List[int]]:
        if not candidates:
            return np.empty((0, NUM_FEATURES), dtype=np.float32), []

        rows = []
        doc_indices = []

        for doc_idx, bm25_score in candidates:
            vec = self.extract_features(query_tokens, doc_idx, bm25_score)
            rows.append(vec)
            doc_indices.append(doc_idx)

        X = np.vstack(rows)
        logger.debug(f"Built feature matrix: {X.shape[0]} candidates × {X.shape[1]} features")
        return X, doc_indices

    @staticmethod
    def feature_names() -> List[str]:
        return FEATURE_NAMES.copy()

    def summarise(self, X: np.ndarray) -> Dict[str, Dict[str, float]]:
        summary = {}
        for i, name in enumerate(FEATURE_NAMES):
            column = X[:, i]
            summary[name] = {
                "mean": float(column.mean()),
                "min":  float(column.min()),
                "max":  float(column.max()),
            }
        return summary