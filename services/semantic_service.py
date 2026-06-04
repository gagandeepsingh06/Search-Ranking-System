import os
import logging
import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class SemanticService:
    """Handles multilingual embedding computation and FAISS vector index matching."""

    def __init__(self, corpus_texts, embeddings_dir="data/embeddings"):
        self.corpus_texts = corpus_texts
        self.embeddings_dir = embeddings_dir
        self.embeddings_path = os.path.join(embeddings_dir, "product_embeddings.npy")
        self.index_path = os.path.join(embeddings_dir, "faiss_index.bin")

        print("\n[SemanticService] Loading sentence transformer model...")
        self.model = SentenceTransformer(MODEL_NAME)
        print(f"[SemanticService] Model '{MODEL_NAME}' loaded.")

        if os.path.exists(self.embeddings_path) and os.path.exists(self.index_path):
            try:
                self._load_index()
                # Verify that the cached index matches the current size of the product corpus
                if self.index.ntotal != len(self.corpus_texts):
                    print(f"[SemanticService] Index size mismatch: cached {self.index.ntotal} vs corpus {len(self.corpus_texts)}. Rebuilding index...")
                    self._build_index()
            except Exception as e:
                print(f"[SemanticService] Error loading cached files ({e}). Rebuilding index...")
                self._build_index()
        else:
            self._build_index()

    def _build_index(self):
        print(f"\n[SemanticService] Computing embeddings for {len(self.corpus_texts)} products...")
        
        self.embeddings = self.model.encode(
            self.corpus_texts,
            show_progress_bar=True,
            batch_size=128,
            normalize_embeddings=True,
        )

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

        os.makedirs(self.embeddings_dir, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)
        faiss.write_index(self.index, self.index_path)

        print(f"[SemanticService] Index built: {self.index.ntotal} vectors, {dim} dimensions.")

    def _load_index(self):
        print("[SemanticService] Loading pre-built embeddings and FAISS index...")
        self.embeddings = np.load(self.embeddings_path)
        self.index = faiss.read_index(self.index_path)
        print(f"[SemanticService] Loaded: {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 50):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, top_k)

        doc_indices = indices[0].tolist()
        sim_scores = scores[0].tolist()

        valid = [(idx, score) for idx, score in zip(doc_indices, sim_scores) if idx >= 0]
        doc_indices = [v[0] for v in valid]
        sim_scores = [v[1] for v in valid]

        logger.debug(f"Semantic search: {len(doc_indices)} candidates retrieved.")
        return doc_indices, sim_scores

    def rebuild_index(self):
        self._build_index()
