import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Cpu, 
  FileText, 
  AlertCircle, 
  ChevronDown, 
  ChevronUp, 
  TrendingUp, 
  Star, 
  Tag, 
  Info,
  Layers,
  Sparkles,
  Shuffle
} from 'lucide-react';

// Friendly descriptions of the 8 ranking features used by our LambdaMART model
const FEATURE_DESCRIPTIONS = {
  bm25_score: {
    label: "Retrieval Score",
    desc: "Relevance score computed by the candidate retrieval step (BM25 raw score, semantic similarity, or RRF combined score).",
    format: (val) => val.toFixed(4)
  },
  query_term_coverage: {
    label: "Query Coverage",
    desc: "Fraction of search terms present in this product description.",
    format: (val) => `${(val * 100).toFixed(0)}%`
  },
  doc_length: {
    label: "Product Text Length",
    desc: "Total word count of this product's title and category.",
    format: (val) => `${val.toFixed(0)} words`
  },
  query_length: {
    label: "Query Length",
    desc: "Number of terms in the user's search query.",
    format: (val) => `${val.toFixed(0)} words`
  },
  exact_match: {
    label: "Exact Phrase Match",
    desc: "Does the complete search phrase appear exactly in the product text?",
    format: (val) => val > 0 ? "Yes" : "No"
  },
  title_match: {
    label: "First Term Match",
    desc: "Does the product text start with the exact first term of the query?",
    format: (val) => val > 0 ? "Yes" : "No"
  },
  avg_term_freq: {
    label: "Avg Term Frequency",
    desc: "Average number of times query terms appear within this product description.",
    format: (val) => val.toFixed(2)
  },
  doc_length_norm: {
    label: "Relative Length Ratio",
    desc: "How long this product text is compared to the corpus average (1.0 = average).",
    format: (val) => `${val.toFixed(2)}x`
  }
};

export default function App() {
  const [query, setQuery] = useState('');
  const [useAi, setUseAi] = useState(true);
  const [searchMode, setSearchMode] = useState('hybrid'); // "keyword", "semantic", "hybrid"
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Track which result cards have their explanations expanded
  const [expandedCards, setExpandedCards] = useState({});

  // Helper to toggle expanded explanation state
  const toggleExplain = (index) => {
    setExpandedCards(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  // Perform search request to FastAPI backend
  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setExpandedCards({}); // Reset expanded cards on new search

    try {
      const response = await fetch('http://localhost:8000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          top_k: parseInt(topK),
          use_ai: useAi,
          search_mode: searchMode
        })
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data.results || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the search API. Please verify that the FastAPI backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // Trigger search on configuration toggle changes
  useEffect(() => {
    if (query.trim()) {
      handleSearch();
    }
  }, [useAi, searchMode, topK]);

  const getModeDescription = () => {
    if (searchMode === 'keyword') return "BM25 keyword search matching exact terms.";
    if (searchMode === 'semantic') return "FAISS vector search finding conceptual matches regardless of words.";
    return "Hybrid fusion combining BM25 keyword and FAISS semantic scores.";
  };

  return (
    <div className="container">
      {/* Header / Navigation */}
      <header className="header-nav">
        <div className="brand">
          <div className="brand-icon">R</div>
          <span className="brand-text">RankAI</span>
        </div>
        <div className="badge-pill">FAISS Semantic + LambdaMART</div>
      </header>

      {/* Hero Section */}
      <section className="hero-sec">
        <h1 className="hero-title">
          Search that <span className="gradient-text">understands</span> you
        </h1>
        <p className="hero-subtitle">
          Intelligent product retrieval utilizing BM25, FAISS semantic vector embeddings, and LambdaMART machine learning re-ranking.
        </p>
      </section>

      {/* Main Search Panel */}
      <div className="glass-panel">
        <form onSubmit={handleSearch}>
          <div className="search-wrapper">
            <div className="search-input-container">
              <Search size={20} className="search-icon-left" />
              <input
                type="text"
                className="search-input"
                placeholder="Search products (e.g. winter cap, running shoes, phone, bag)..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
          </div>

          <div className="controls-row">
            {/* Search Retrieval Engine Mode */}
            <div className="toggle-group-container">
              <span className="control-label">Retrieval Engine:</span>
              <div className="toggle-group">
                <div 
                  className={`toggle-option ${searchMode === 'keyword' ? 'active' : ''}`}
                  onClick={() => setSearchMode('keyword')}
                  title="Traditional BM25 exact keyword match"
                >
                  <Layers size={14} />
                  <span>Keyword</span>
                </div>
                <div 
                  className={`toggle-option ${searchMode === 'semantic' ? 'active' : ''}`}
                  onClick={() => setSearchMode('semantic')}
                  title="Multilingual semantic search (FAISS Vector Index)"
                >
                  <Sparkles size={14} />
                  <span>Semantic</span>
                </div>
                <div 
                  className={`toggle-option ${searchMode === 'hybrid' ? 'active' : ''}`}
                  onClick={() => setSearchMode('hybrid')}
                  title="Combined Keyword + Semantic Search via Reciprocal Rank Fusion (RRF)"
                >
                  <Shuffle size={14} />
                  <span>Hybrid</span>
                </div>
              </div>
            </div>

            {/* AI Ranking Toggle */}
            <div className="toggle-group-container">
              <span className="control-label">Re-ranking:</span>
              <div className="toggle-group">
                <div 
                  className={`toggle-option ${useAi ? 'active' : ''}`}
                  onClick={() => setUseAi(true)}
                  title="Re-order candidates with LambdaMART ML Ranker"
                >
                  <Cpu size={14} />
                  <span>AI Ranker</span>
                </div>
                <div 
                  className={`toggle-option ${!useAi ? 'active' : ''}`}
                  onClick={() => setUseAi(false)}
                  title="No re-ranking, output candidates directly from retrieval engine"
                >
                  <span>No AI</span>
                </div>
              </div>
            </div>

            {/* Results limit slider */}
            <div className="slider-group">
              <span className="control-label">Limit: <b>{topK}</b></span>
              <input
                type="range"
                className="slider-input"
                min="5"
                max="20"
                step="1"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
              />
            </div>
          </div>
          <div className="helper-text-row">
            <span>💡 <b>Mode:</b> {getModeDescription()}</span>
          </div>
        </form>
      </div>

      {/* Error state */}
      {error && (
        <div className="glass-panel" style={{ borderColor: 'rgba(239, 68, 68, 0.2)', background: 'rgba(239, 68, 68, 0.03)' }}>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
            <AlertCircle size={24} style={{ color: 'var(--danger)', flexShrink: 0 }} />
            <div>
              <h4 style={{ color: '#fff', marginBottom: '0.5rem', fontWeight: 600 }}>Backend Connection Error</h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', lineHeight: '1.5', marginBottom: '1rem' }}>
                {error}
              </p>
              <div style={{ background: 'rgba(0,0,0,0.4)', padding: '0.75rem 1rem', borderRadius: '10px', fontFamily: 'monospace', fontSize: '0.8rem', border: '1px solid rgba(255,255,255,0.05)', color: 'var(--primary-light)' }}>
                python run.py
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div className="loader-container">
          <div className="spinner"></div>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Retrieving candidates and applying LambdaMART AI ranking...</span>
        </div>
      )}

      {/* Results Rendering */}
      {!loading && !error && (
        <>
          {results.length > 0 ? (
            <div>
              {/* Stats dashboard */}
              <div className="stats-dashboard">
                <span className="stats-count">
                  Found <b>{total}</b> products for "{query}"
                </span>
                <span className={`stats-mode ${!useAi ? 'bm25' : ''}`}>
                  {useAi ? (
                    <>
                      <Cpu size={14} />
                      <span>LambdaMART ML Re-ranked</span>
                    </>
                  ) : (
                    <>
                      {searchMode === 'keyword' ? <Layers size={14} /> : searchMode === 'semantic' ? <Sparkles size={14} /> : <Shuffle size={14} />}
                      <span style={{ textTransform: 'capitalize' }}>Raw {searchMode} Match</span>
                    </>
                  )}
                </span>
              </div>

              {/* Cards List */}
              <div className="results-list">
                {results.map((item, index) => {
                  const rankNum = index + 1;
                  const isExpanded = expandedCards[index];
                  
                  return (
                    <div className="result-card" key={item.product_index}>
                      <div className="card-header-row">
                        <div className="card-left">
                          {/* Rank Display badge */}
                          <div className={`rank-badge ${!useAi ? 'bm25-rank' : ''}`}>
                            #{rankNum}
                          </div>
                          
                          <div className="product-info">
                            <h3 className="product-title">{item.title_translated}</h3>
                            <div className="product-title-original" title="Original Hindi Title in Dataset">
                              <Info size={12} />
                              <span>{item.title_original}</span>
                            </div>
                          </div>
                        </div>

                        {/* Top-right Score Display */}
                        <div className="card-right-score">
                          <div className={`score-badge ${!useAi ? 'bm25-score' : ''}`}>
                            {item.score.toFixed(4)}
                          </div>
                          <span className="score-label">
                            {useAi ? "AI SCORE" : "RETRIEVAL SCORE"}
                          </span>
                        </div>
                      </div>

                      {/* Product Badges */}
                      <div className="card-tags">
                        <span className="badge badge-price">
                          ₹{item.price.toFixed(2)}
                        </span>
                        
                        <span className="badge badge-rating">
                          <Star size={12} fill="#fbbf24" stroke="none" />
                          <span>{item.rating.toFixed(1)}</span>
                        </span>
                        
                        <span className="badge badge-tag">
                          {item.review_count} reviews
                        </span>

                        <span className="badge badge-tag" style={{ textTransform: 'capitalize' }}>
                          <Tag size={11} />
                          <span>{item.category_translated}</span>
                        </span>

                        {item.bought_last_month > 0 && (
                          <span className="badge badge-tag" style={{ color: 'var(--primary-light)', borderColor: 'rgba(99, 102, 241, 0.2)' }}>
                            <TrendingUp size={11} />
                            <span>{item.bought_last_month}+ bought last mo</span>
                          </span>
                        )}

                        {item.is_best_seller && (
                          <span className="badge badge-best">
                            Best Seller
                          </span>
                        )}

                        {/* Explain Button */}
                        <button 
                          className="btn-explain"
                          onClick={() => toggleExplain(index)}
                          title="Explain ranking features for this candidate"
                        >
                          <FileText size={13} />
                          <span>{isExpanded ? "Hide Explain" : "Explain"}</span>
                          {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                        </button>
                      </div>

                      {/* Expanded Explainer Panel */}
                      {isExpanded && (
                        <div className="explainer-drawer">
                          <div className="explainer-title">
                            <Cpu size={14} style={{ color: 'var(--primary)' }} />
                            <span>AI Ranking Feature Vectors</span>
                          </div>
                          
                          <div className="explainer-grid">
                            {Object.entries(item.features).map(([fKey, fVal]) => {
                              const details = FEATURE_DESCRIPTIONS[fKey] || {
                                label: fKey,
                                desc: "No description available",
                                format: (val) => val
                              };

                              // Calculate progress percentage for visual bar representation
                              let progress = 0;
                              if (fKey === 'query_term_coverage' || fKey === 'exact_match' || fKey === 'title_match') {
                                progress = fVal * 100;
                              } else if (fKey === 'bm25_score') {
                                progress = Math.min((fVal / 15) * 100, 100);
                              } else if (fKey === 'doc_length_norm') {
                                progress = Math.min((fVal / 3) * 100, 100);
                              } else {
                                progress = Math.min((fVal / 30) * 100, 100);
                              }

                              return (
                                <div className="feature-bar-row" key={fKey} title={details.desc}>
                                  <div className="feature-info-row">
                                    <span className="feature-name">{details.label}</span>
                                    <span className="feature-val">{details.format(fVal)}</span>
                                  </div>
                                  <div className="bar-bg">
                                    <div 
                                      className="bar-fill"
                                      style={{ width: `${progress}%` }}
                                    ></div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            query.trim() && (
              <div className="empty-state">
                <AlertCircle size={32} />
                <span className="empty-title">No candidates found</span>
                <span style={{ fontSize: '0.85rem' }}>
                  Please try searching with another keyword or adjust your terms.
                </span>
              </div>
            )
          )}

          {/* Initial landing state */}
          {!query.trim() && (
            <div className="empty-state" style={{ padding: '8rem 2rem' }}>
              <Search size={36} style={{ color: 'var(--text-dark)' }} />
              <span className="empty-title">Enter a query above to start search</span>
              <span style={{ fontSize: '0.85rem', maxWidth: '350px', margin: '0 auto', lineHeight: '1.5' }}>
                Try query samples such as "winter cap", "running shoes", "laptop", or "saree".
              </span>
            </div>
          )}
        </>
      )}

      {/* Fine Print Footer */}
      <footer className="footer-text">
        <span>RankAI System &bull; Keyword / Semantic Search + LambdaMART AI Re-ranking Engine</span>
      </footer>
    </div>
  );
}
