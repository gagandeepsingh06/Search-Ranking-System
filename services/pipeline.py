import logging
from typing import List, Tuple, Optional

from services.bm25_service import BM25Service
from services.feature_service import FeatureService
from services.ranking_service import RankingService
from services.semantic_service import SemanticService

logger = logging.getLogger(__name__)


class SearchPipeline:
    """Orchestration layer connecting BM25, FAISS semantic search, and LambdaMART re-ranking."""

    def __init__(
        self,
        data_path: str,
        model_path: Optional[str] = None,
        bm25_top_k: int = 50,
    ):
        self.bm25_top_k = bm25_top_k
        self.bm25 = BM25Service(data_path)
        self.features = FeatureService(self.bm25.tokenized_corpus)
        self.ranker = RankingService(model_path, df=self.bm25.df)
        
        corpus_texts = self.bm25.df["search_text"].tolist()
        self.semantic = SemanticService(corpus_texts)
        logger.info("SearchPipeline ready (BM25 + Semantic + LambdaMART).")

    def search(self, query_tokens: List[str], top_k: int = 10):
        """Retrieve candidates via BM25 and re-rank with LambdaMART."""
        query_string = " ".join(query_tokens)
        candidates = self.bm25.retrieve_candidates(query_string, top_k=self.bm25_top_k)
        top_n, scores = candidates

        if len(top_n) == 0:
            return []

        candidate_list = [(int(idx), float(scores[idx])) for idx in top_n]
        X, doc_indices = self.features.build_feature_matrix(query_tokens, candidate_list)
        bm25_scores = [scores[idx] for idx in top_n]

        ranked = self.ranker.rerank(
            X,
            doc_indices,
            bm25_scores=bm25_scores,
            query=query_string,
        )
        return ranked[:top_k]

    def search_semantic(self, query: str, top_k: int = 10):
        """Retrieve candidates via semantic similarity and re-rank with LambdaMART."""
        doc_indices, sim_scores = self.semantic.search(query, top_k=self.bm25_top_k)
        if not doc_indices:
            return []

        candidate_list = [(idx, score) for idx, score in zip(doc_indices, sim_scores)]
        query_tokens = query.lower().split()
        X, feat_indices = self.features.build_feature_matrix(query_tokens, candidate_list)

        ranked = self.ranker.rerank(
            X,
            feat_indices,
            bm25_scores=sim_scores,
            query=query,
        )
        return ranked[:top_k]

    def search_hybrid(self, query: str, top_k: int = 10, bm25_weight: float = 0.4, semantic_weight: float = 0.6):
        """Hybrid search combining BM25 keyword matching and semantic search via RRF."""
        query_tokens = query.lower().split()
        query_string = " ".join(query_tokens)

        top_n_bm25, bm25_scores_all = self.bm25.retrieve_candidates(query_string, top_k=self.bm25_top_k)
        sem_indices, sem_scores = self.semantic.search(query, top_k=self.bm25_top_k)

        doc_scores = {}
        for rank, idx in enumerate(top_n_bm25):
            idx = int(idx)
            rrf_score = 1.0 / (60 + rank + 1)
            doc_scores[idx] = doc_scores.get(idx, 0.0) + bm25_weight * rrf_score

        for rank, (idx, sim) in enumerate(zip(sem_indices, sem_scores)):
            rrf_score = 1.0 / (60 + rank + 1)
            doc_scores[idx] = doc_scores.get(idx, 0.0) + semantic_weight * rrf_score

        merged = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:self.bm25_top_k]
        if not merged:
            return []

        doc_indices = [idx for idx, _ in merged]
        combined_scores = [score for _, score in merged]

        candidate_list = [(idx, score) for idx, score in zip(doc_indices, combined_scores)]
        X, feat_indices = self.features.build_feature_matrix(query_tokens, candidate_list)

        ranked = self.ranker.rerank(
            X,
            feat_indices,
            bm25_scores=combined_scores,
            query=query,
        )
        return ranked[:top_k]

    def explain(self, query_tokens: List[str], doc_index: int, bm25_score: float):
        vec = self.features.extract_features(query_tokens, doc_index, bm25_score)
        return dict(zip(self.features.feature_names(), vec.tolist()))