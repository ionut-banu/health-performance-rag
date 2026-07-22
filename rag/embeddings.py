"""
Module 2: local text embeddings for vector search.

Uses a small sentence-transformers model (multi-qa-MiniLM-L6-cos-v1, 384-dim) so
embedding the ~5k chunks is free and offline — answer generation stays on OpenAI.
The model is trained for cosine similarity; we L2-normalize so sqlitesearch's cosine
rerank behaves as expected.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
EMBEDDING_DIM = 384

# Reuse a single model across calls (first access downloads ~90MB to the HF cache).
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(
    texts: list[str], batch_size: int = 64, show_progress: bool = True
) -> np.ndarray:
    """
    Embed a list of texts -> (len(texts), EMBEDDING_DIM) float32 array.

    `show_progress` drives tqdm, which is useful in a terminal but renders as unreadable
    carriage-return spam in container logs — callers that print their own progress pass False.
    """
    return get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def embed_query(text: str) -> np.ndarray:
    """Embed a single query -> (EMBEDDING_DIM,) float32 array."""
    return get_model().encode(text, normalize_embeddings=True, convert_to_numpy=True)


def doc_embed_text(flat: dict) -> str:
    """Text fed to the embedder for a chunk: title + chapter carry topical signal."""
    parts = [flat.get("title") or "", flat.get("chapter_title") or "", flat.get("text") or ""]
    return ". ".join(p for p in parts if p)
