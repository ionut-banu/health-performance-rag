"""
Module 6: cross-encoder re-ranking.

The retrievers score a query against a *pre-computed* representation of each chunk
(TF-IDF weights or a single 384-d vector), which is fast but lossy. A cross-encoder
instead reads the query and the chunk text *together* and scores the pair directly —
far more accurate, but too slow to run over the whole corpus. So it's used as a second
pass: retrieve a deep candidate list cheaply, then re-rank those candidates precisely.

Local and free (same sentence-transformers install as embeddings.py — no new dependency
and no API calls).
"""
from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Reuse a single model across calls (first access downloads ~80MB to the HF cache).
_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Re-score (query, chunk) pairs with the cross-encoder and return the best top_k."""
    if not chunks:
        return []
    scores = get_reranker().predict([(query, c["text"]) for c in chunks])
    ordered = sorted(zip(scores, range(len(chunks))), key=lambda p: p[0], reverse=True)
    return [chunks[i] for _, i in ordered[:top_k]]
