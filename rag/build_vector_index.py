"""
Module 2: one-time build of the on-disk vector index.

Run once (re-run after re-ingesting documents.jsonl):
    uv run rag/build_vector_index.py

First run downloads the sentence-transformers model (~90MB) to the HF cache and
embeds every chunk on CPU — a few minutes. Subsequent RAG queries reopen the .db
without re-embedding.
"""
import time

from vector_search import VECTOR_DB_PATH, build_vector_index


def main():
    start = time.time()
    build_vector_index()
    print(f"Done in {time.time() - start:.1f}s -> {VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()
