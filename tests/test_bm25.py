from services.bm25_service import BM25Service

bm25_service = BM25Service(
    "data/processed/cleaned_products.csv"
)

results, scores = bm25_service.retrieve_candidates(
    "कैप"
)

print(results[:5])