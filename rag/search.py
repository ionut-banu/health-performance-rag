"""
Module 1: keyword search over the ingested documents using minsearch.

minsearch is a small in-memory TF-IDF index (the course's search library, reused
for vector search in Module 2). Its text_fields/keyword_fields reference top-level
dict keys, so the nested `metadata` from schema.Document is flattened here first.
"""
import os
import sys

from minsearch import Index

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import Document, load_documents

DOCUMENTS_PATH = "data/documents.jsonl"

TEXT_FIELDS = ["text", "title", "chapter_title"]
KEYWORD_FIELDS = ["source", "content_type", "video_id"]
DEFAULT_BOOST = {"text": 1.0, "title": 0.5, "chapter_title": 0.3}


def flatten(doc: Document) -> dict:
    """Promote transcript metadata to top level so minsearch can index it."""
    meta = doc.metadata or {}
    return {
        "id": doc.id,
        "source": doc.source,
        "content_type": doc.content_type,
        "title": doc.title,
        "url": doc.url,
        "text": doc.text,
        "chapter_title": meta.get("chapter_title") or "",
        "video_id": meta.get("video_id") or "",
        "start_timestamp": meta.get("start_timestamp"),
        "end_timestamp": meta.get("end_timestamp"),
        "upload_date": meta.get("upload_date"),
        # The pre-sub-chunking chapter id. Carried through so evaluation collected on
        # chapter-sized chunks still resolves against sub-chunks (see docs/evaluation.md).
        "parent_chunk_id": meta.get("parent_chunk_id") or "",
    }


def build_index(documents_path: str = DOCUMENTS_PATH) -> Index:
    """Load data/documents.jsonl, flatten, and fit a keyword index over it."""
    docs = [flatten(d) for d in load_documents(documents_path)]
    index = Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS)
    index.fit(docs)
    print(f"Indexed {len(docs)} documents from {documents_path}")
    return index


def search(
    index: Index,
    query: str,
    num_results: int = 5,
    source: str | None = None,
) -> list[dict]:
    """Keyword search; optionally restrict to a single source (huberman/galpin)."""
    filter_dict = {"source": source} if source else {}
    return index.search(
        query,
        filter_dict=filter_dict,
        boost_dict=DEFAULT_BOOST,
        num_results=num_results,
    )
