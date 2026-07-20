"""
Backend-agnostic retrieval dispatch.

Both retrievers return the same flattened dict shape (see search.flatten), so the
RAG loops in rag.py stay backend-agnostic. Adding hybrid search in Module 6 means
adding one more branch here, not touching rag.py.

  method="keyword"  Module 1: minsearch TF-IDF index (in-memory).
  method="vector"   Module 2: sqlitesearch HNSW index over local embeddings (on-disk).
"""
from search import build_index, search
from vector_search import load_vector_index, vector_search

# Lazy singletons so each index is built/opened at most once per process.
_keyword_index = None
_vector_index = None


def get_keyword_index():
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = build_index()
    return _keyword_index


def get_vector_index():
    global _vector_index
    if _vector_index is None:
        _vector_index = load_vector_index()
    return _vector_index


def retrieve(
    query: str,
    num_results: int = 5,
    source: str | None = None,
    method: str = "keyword",
) -> list[dict]:
    """Retrieve chunks for a query via the chosen backend."""
    if method == "keyword":
        return search(get_keyword_index(), query, num_results=num_results, source=source)
    if method == "vector":
        return vector_search(get_vector_index(), query, num_results=num_results, source=source)
    raise ValueError(f"Unknown retrieval method: {method!r} (expected 'keyword' or 'vector')")
