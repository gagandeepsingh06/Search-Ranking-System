# 🔍 RankAI — AI-Powered Search Ranking Engine

> A production-grade search engine that combines **BM25 keyword retrieval**, **FAISS semantic vector search**, and **LambdaMART Learning-to-Rank** to deliver intelligent, explainable product ranking — built with **React + FastAPI + LightGBM + Sentence Transformers**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-LambdaMART-green?style=flat)](https://lightgbm.readthedocs.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=flat)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

---

## 🎯 Problem Statement

Traditional e-commerce search systems rely solely on **keyword matching** (e.g., SQL `LIKE` queries or basic TF-IDF), which fails when:
- Users search with **natural language** (e.g., *"something warm for head in winter"*)
- Exact keywords don't appear in product titles
- Products with high ratings/reviews are buried below irrelevant matches

**RankAI** solves this by implementing a **3-stage intelligent retrieval pipeline** that understands both exact keywords and semantic meaning, then re-ranks results using a machine learning model trained on product quality signals.

---

## ⚡ Key Features

| Feature | Description |
|---|---|
| 🔤 **BM25 Keyword Search** | Okapi BM25 sparse retrieval for exact term matching across 14,000+ products |
| 🧠 **Semantic Search (FAISS)** | Multilingual Sentence Transformer embeddings + Facebook FAISS vector index for conceptual matching |
| 🔀 **Hybrid Search (RRF)** | Reciprocal Rank Fusion merges keyword + semantic candidates — the industry-standard approach used by Elasticsearch |
| 🏆 **LambdaMART Re-Ranking** | LightGBM `LGBMRanker` trained with `lambdarank` objective on 6 product quality features |
| 🔍 **Explainable AI (XAI)** | Every result card has an "Explain" drawer showing 8 feature vectors with visual progress bars |
| 🌐 **Cross-Lingual Search** | English queries search a Hindi product corpus via multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) |
| 🎨 **Premium React UI** | Glassmorphic dark theme, micro-animations, responsive design, real-time mode switching |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    User Query (React Frontend)                   │
│                     http://localhost:5173                         │
└────────────────────────────┬─────────────────────────────────────┘
                             │ POST /search
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FastAPI Backend (:8000)                          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │   Keyword     │  │   Semantic   │  │       Hybrid           │  │
│  │  (BM25 Index) │  │ (FAISS+SBERT)│  │  (RRF Fusion)          │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬─────────────┘  │
│         │                 │                      │                │
│         └─────────────────┴──────────────────────┘                │
│                           │                                       │
│                    Top-50 Candidates                              │
│                           │                                       │
│              ┌────────────▼────────────┐                          │
│              │   Feature Extraction     │                          │
│              │   (8 Lexical Features)   │                          │
│              └────────────┬────────────┘                          │
│                           │                                       │
│              ┌────────────▼────────────┐                          │
│              │   LambdaMART Re-Ranker  │                          │
│              │   (LightGBM LGBMRanker) │                          │
│              └────────────┬────────────┘                          │
│                           │                                       │
│                  Ranked Results + XAI                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19 + Vite, Lucide Icons, Vanilla CSS (Glassmorphism) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Sparse Retrieval** | BM25 (rank-bm25), NLTK Tokenization |
| **Dense Retrieval** | Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`), Facebook FAISS |
| **ML Ranking** | LightGBM (`LGBMRanker`, LambdaMART objective, NDCG metric) |
| **Data** | Amazon India Product Dataset (~14,000 products, 11 attributes) |
| **Translation** | Custom Hindi↔English bidirectional translation with caching |

---

## 📁 Project Structure

```
AI-Based-Search-Ranking-System/
│
├── api/
│   └── search_api.py              # FastAPI endpoints (3 search modes + CORS)
│
├── frontend/                      # React (Vite) Web Application
│   ├── src/
│   │   ├── App.jsx                # Search dashboard with mode toggles & XAI drawer
│   │   ├── index.css              # Midnight Glow glassmorphic design system
│   │   └── main.jsx               # React entry point
│   ├── index.html                 # SEO-optimized HTML shell
│   └── package.json
│
├── services/                      # Core Search Engine Services
│   ├── bm25_service.py            # BM25 sparse index construction & retrieval
│   ├── semantic_service.py        # Sentence Transformer + FAISS dense retrieval
│   ├── feature_service.py         # 8-dimension feature vector extraction
│   ├── ranking_service.py         # LambdaMART ML inference + category boosting
│   ├── pipeline.py                # Unified orchestrator (keyword/semantic/hybrid)
│   └── translator_service.py      # Hindi↔English translation with LRU cache
│
├── pipelines/                     # Data & Model Training Pipelines
│   ├── clean_data.py              # Raw data cleaning & preprocessing
│   ├── generate_training_data.py  # Relevance label generation for LTR
│   └── train_lambdamart.py        # LightGBM LambdaMART model training
│
├── models/
│   └── lambdamart_ranker.pkl      # Trained LGBMRanker model weights
│
├── data/
│   ├── raw/                       # Original Amazon product CSV
│   ├── processed/                 # Cleaned corpus & training data
│   └── embeddings/                # Cached FAISS index & product embeddings
│
├── notebooks/                     # Jupyter Notebooks (EDA, BM25, LambdaMART)
├── configs/settings.py            # Centralized configuration
├── run.py                         # Single-command dual-server launcher
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-search-ranking-system.git
cd ai-search-ranking-system

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install Python dependencies
pip install -r requirements.txt
pip install sentence-transformers faiss-cpu

# 4. Install frontend dependencies
cd frontend && npm install && cd ..

# 5. Launch the system
python run.py
```

The system will start:
- **Backend API:** http://localhost:8000
- **React Frontend:** http://localhost:5173

> **Note:** On first launch, the system downloads the multilingual embedding model (~470MB) and computes product embeddings (~4 min). Subsequent launches load from cache in <1 second.

---

## 📊 Search Modes Comparison

| Query | Keyword (BM25) | Semantic (FAISS) | Hybrid (RRF) |
|---|---|---|---|
| `"cap"` | ✅ Exact matches for caps, hats | ✅ Caps + related headwear | ✅ Best of both |
| `"something warm for head"` | ❌ Returns irrelevant (no keyword match) | ✅ Returns Winter Beanie Hat | ✅ Returns Winter Beanie Hat |
| `"bag"` | ⚠️ Partial matches | ✅ PU Bag, Kit Bag, Refill Bag | ✅ Combined ranking |

This demonstrates why **semantic search is essential** — keyword-only search completely fails on natural language queries.

---

## 🧠 ML Model Details

### LambdaMART (Learning-to-Rank)

| Parameter | Value |
|---|---|
| **Algorithm** | LambdaMART (Gradient Boosted Decision Trees) |
| **Framework** | LightGBM `LGBMRanker` |
| **Objective** | `lambdarank` |
| **Metric** | NDCG (Normalized Discounted Cumulative Gain) |
| **Boosting** | GBDT, 100 estimators |
| **Input Features** | `bm25_score`, `rating`, `review_count`, `price`, `is_best_seller`, `bought_last_month` |

### Sentence Transformer (Dense Embeddings)

| Parameter | Value |
|---|---|
| **Model** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Dimensions** | 384 |
| **Languages** | 50+ (English, Hindi, and more) |
| **Index** | FAISS `IndexFlatIP` (Inner Product / Cosine Similarity) |
| **Corpus Size** | 14,394 product embeddings |

---

## 🔍 Explainability (XAI)

Every search result includes an **Explain** drawer that visualizes 8 feature dimensions:

| Feature | Description |
|---|---|
| Retrieval Score | BM25 / Semantic / Hybrid score from candidate retrieval |
| Query Coverage | Fraction of query terms found in the product text |
| Product Text Length | Word count of title + category |
| Query Length | Number of search terms |
| Exact Phrase Match | Whether the full query appears verbatim |
| First Term Match | Whether the product starts with the query's first word |
| Avg Term Frequency | Mean occurrence count of query terms in the document |
| Relative Length Ratio | Document length relative to corpus average |

---

## 🧪 API Reference

### `POST /search`

**Request:**
```json
{
  "query": "winter cap",
  "top_k": 10,
  "use_ai": true,
  "search_mode": "hybrid"
}
```

| Field | Type | Description |
|---|---|---|
| `query` | string | Search query (English) |
| `top_k` | int | Number of results (default: 10) |
| `use_ai` | bool | Enable LambdaMART re-ranking |
| `search_mode` | string | `"keyword"`, `"semantic"`, or `"hybrid"` |

**Response:**
```json
{
  "query": "winter cap",
  "use_ai": true,
  "search_mode": "hybrid",
  "total": 10,
  "results": [
    {
      "product_index": 2084,
      "score": 10.6465,
      "title_translated": "Winter Beanie Hat scarf set...",
      "price": 475.0,
      "rating": 4.2,
      "review_count": 156,
      "is_best_seller": false,
      "features": { "bm25_score": 4.23, "query_term_coverage": 0.85, ... }
    }
  ]
}
```

---

## 📜 License

This project is for educational and portfolio purposes.

---

**Built with ❤️ using React, FastAPI, LightGBM, Sentence Transformers & FAISS**
