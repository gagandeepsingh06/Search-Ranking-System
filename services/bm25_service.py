import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from services.translator_service import TranslatorService
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class BM25Service:
    """Handles BM25 sparse keyword index creation and retrieval."""

    def __init__(self, data_path: str):
        print("\nLoading dataset...")
        self.df = pd.read_csv(data_path)
        print("Dataset loaded successfully!")

        print("Tokenizing corpus...")
        self.tokenized_corpus = [
            word_tokenize(text.lower())
            for text in self.df["search_text"]
        ]
        print("Tokenization completed!")

        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 index created successfully!")

    def retrieve_candidates(self, query: str, top_k: int = 20):
        print("\n[BM25 Retrieval Started]")
        
        # Translate query to target language (Hindi)
        hindi_query = TranslatorService.translate_query(query)
        tokenized_query = hindi_query.lower().split()
        
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        
        print(f"Top {top_k} candidates retrieved!")
        return top_n, scores