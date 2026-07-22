"""
Module 2: semantic (vector) search over the ingested documents using sqlitesearch.

sqlitesearch is the persistent, on-disk sibling of minsearch (Module 1). We embed each
chunk with a local sentence-transformers model and store the vectors in an HNSW index
backed by a single .db file, so the index survives restarts and reopens without
re-embedding. The returned payload matches the flattened dict shape that rag.build_context
expects, so vector retrieval is drop-in for the keyword path.
"""
import os
import sys

from sqlitesearch import VectorSearchIndex

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import load_documents
from search import DOCUMENTS_PATH, KEYWORD_FIELDS, flatten
from embeddings import doc_embed_text, embed_query, embed_texts

VECTOR_DB_PATH = "data/vector_index.db"


def build_vector_index(
    documents_path: str = DOCUMENTS_PATH,
    db_path: str = VECTOR_DB_PATH,
) -> VectorSearchIndex:
    """Load documents.jsonl, embed each chunk, and fit + persist an HNSW vector index."""
    docs = [flatten(d) for d in load_documents(documents_path)]
    vectors = embed_texts([doc_embed_text(d) for d in docs])
    # id_field is omitted: the library appends it to keyword_fields, and "id"
    # collides with the docs table's own primary key. Our doc id round-trips
    # inside the stored payload regardless; upsert-dedup only matters for the
    # shared text+vector file that hybrid search would need.
    index = VectorSearchIndex(
        mode="hnsw",
        keyword_fields=KEYWORD_FIELDS,
        db_path=db_path,
    )
    index.fit(vectors, docs)
    print(f"Indexed {len(docs)} documents into {db_path}")
    return index


def load_vector_index(db_path: str = VECTOR_DB_PATH) -> VectorSearchIndex:
    """Reopen an existing on-disk vector index; build it if the .db is missing."""
    if not os.path.exists(db_path):
        print(f"No vector index at {db_path}; building it now...")
        return build_vector_index(db_path=db_path)
    return VectorSearchIndex(
        mode="hnsw",
        keyword_fields=KEYWORD_FIELDS,
        db_path=db_path,
    )


def vector_search(
    index: VectorSearchIndex,
    query: str,
    num_results: int = 5,
    source: str | None = None,
) -> list[dict]:
    """Semantic search; optionally restrict to a single source (huberman/galpin)."""
    filter_dict = {"source": source} if source else None
    return index.search(
        embed_query(query),
        filter_dict=filter_dict,
        num_results=num_results,
    )
