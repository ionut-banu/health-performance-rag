"""
Backend-agnostic retrieval dispatch.

Both retrievers return the same flattened dict shape (see search.flatten), so the
RAG loops in rag.py stay backend-agnostic: adding a retrieval strategy means adding a
branch here, not touching rag.py.

  method="keyword"  Module 1: minsearch TF-IDF index (in-memory).
  method="vector"   Module 2: HNSW index over local embeddings.
  method="hybrid"   Module 6: both, fused with Reciprocal Rank Fusion.

The vector backend is chosen at runtime: Postgres + pgvector when PGVECTOR_URL is set
(docker-compose sets it), otherwise the infra-free on-disk sqlitesearch index. Both return
the same flattened dict shape, so nothing downstream — including the evaluation harness —
cares which one is active.
"""
from search import build_index, search
from vector_search import load_vector_index, vector_search
from pgvector_search import pg_vector_search, pgvector_url

# RRF constant. Fusing by rank (not score) sidesteps the fact that TF-IDF scores and
# cosine similarities live on incomparable scales; 60 is the value from the original
# RRF paper and damps the influence of any single retriever's top hit.
RRF_K = 60

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


def warm_indexes() -> None:
    """
    Prime only the indexes this process will actually use.

    Important for the container: with pgvector active, touching get_vector_index() would
    build the unused on-disk sqlitesearch index — several minutes of pointless work.
    """
    get_keyword_index()
    if not pgvector_url():
        get_vector_index()


def reciprocal_rank_fusion(result_lists: list[list[dict]], num_results: int) -> list[dict]:
    """
    Fuse ranked result lists by Reciprocal Rank Fusion: each doc scores sum(1/(RRF_K + rank))
    across the lists it appears in, so documents both retrievers like rise to the top.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in result_lists:
        for rank, doc in enumerate(results, 1):
            key = doc["id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            docs.setdefault(key, doc)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[k] for k in ranked[:num_results]]


def _vector(query: str, num_results: int, source: str | None) -> list[dict]:
    """Vector search via whichever backend is configured (pgvector if PGVECTOR_URL is set)."""
    if pgvector_url():
        return pg_vector_search(query, num_results=num_results, source=source)
    return vector_search(get_vector_index(), query, num_results=num_results, source=source)


def _dispatch(query: str, num_results: int, source: str | None, method: str, candidates: int):
    if method == "keyword":
        return search(get_keyword_index(), query, num_results=num_results, source=source)
    if method == "vector":
        return _vector(query, num_results, source)
    if method == "hybrid":
        return reciprocal_rank_fusion(
            [
                search(get_keyword_index(), query, num_results=candidates, source=source),
                _vector(query, candidates, source),
            ],
            num_results=num_results,
        )
    raise ValueError(
        f"Unknown retrieval method: {method!r} (expected 'keyword', 'vector', or 'hybrid')"
    )


def retrieve(
    query: str,
    num_results: int = 5,
    source: str | None = None,
    # Defaults are the measured winner — see docs/evaluation.md for current numbers.
    # Deliberately no figures inline: they go stale every time the corpus or ground truth
    # is regenerated, and a wrong number in a comment is worse than no number.
    method: str = "hybrid",
    candidates: int = 20,    # per-backend depth fused, and the pool the re-ranker reorders
    rerank: bool = True,     # Module 6: cross-encoder second pass over the candidates
    rewrite: bool = False,   # Module 6: LLM rewrites the query first (off — see evaluation)
) -> list[dict]:
    """Retrieve chunks for a query via the chosen backend, with optional Module 6 stages."""
    if rewrite:
        from query_rewrite import rewrite_query

        query = rewrite_query(query)

    if not rerank:
        return _dispatch(query, num_results, source, method, candidates)

    # Re-ranking only helps if it has a deeper candidate pool to reorder than it returns.
    from rerank import rerank as rerank_chunks

    pool = _dispatch(query, candidates, source, method, candidates)
    return rerank_chunks(query, pool, top_k=num_results)
