"""
Module 7: semantic search backed by Postgres + pgvector.

This is the containerized counterpart to vector_search.py (sqlitesearch). Both return the
same flattened dict shape — see search.flatten — so retrieve.py, rag.py and the evaluation
harness work against either without changes. retrieve.py picks this backend when
PGVECTOR_URL is set (docker-compose sets it; local runs leave it unset and use SQLite).

Chunk payloads are stored as JSONB and returned verbatim, so a retrieved row is
indistinguishable from what the SQLite backend produces.
"""
import json
import os
import sys

import psycopg
from pgvector.psycopg import register_vector

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import load_documents
from search import DOCUMENTS_PATH, flatten
from embeddings import EMBEDDING_DIM, doc_embed_text, embed_query, embed_texts

PGVECTOR_URL_ENV = "PGVECTOR_URL"
TABLE = "chunks"
INSERT_BATCH = 500

SCHEMA_SQL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {TABLE} (
    id        TEXT PRIMARY KEY,
    source    TEXT,
    payload   JSONB NOT NULL,
    embedding vector({EMBEDDING_DIM}) NOT NULL
);

CREATE INDEX IF NOT EXISTS {TABLE}_source_idx ON {TABLE} (source);
"""

# Built after the bulk load: creating an HNSW index up front would make every insert pay
# index-maintenance cost for no benefit.
INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS {TABLE}_embedding_idx
    ON {TABLE} USING hnsw (embedding vector_cosine_ops)
"""


def pgvector_url() -> str | None:
    """The configured Postgres URL, or None when we should fall back to SQLite."""
    return os.environ.get(PGVECTOR_URL_ENV) or None


def connect(url: str | None = None) -> psycopg.Connection:
    """Open a connection with the pgvector type adapters registered."""
    conn = psycopg.connect(url or pgvector_url(), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def count_chunks(conn: psycopg.Connection) -> int:
    """Number of indexed chunks; 0 if the table doesn't exist yet."""
    try:
        return conn.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
    except psycopg.errors.UndefinedTable:
        return 0


def build_pg_index(
    documents_path: str = DOCUMENTS_PATH,
    url: str | None = None,
    force: bool = False,
) -> int:
    """
    Embed every chunk and load it into Postgres. Returns the final row count.

    Resumable and safe to call on every container start: it inserts only the ids that are
    missing, so an interrupted build continues instead of restarting. Note it deliberately
    compares against the *expected* document count rather than "table is non-empty" — a
    partially-loaded table would otherwise look complete and the app would serve queries
    against a silently incomplete index.

    Embedding on CPU is slow (tens of minutes for the full corpus), so work is committed in
    batches: progress is visible in the logs and survives a restart.
    """
    docs = [flatten(d) for d in load_documents(documents_path)]
    expected = len(docs)

    with connect(url) as conn:
        conn.execute(SCHEMA_SQL)
        if force:
            conn.execute(f"TRUNCATE {TABLE}")
            existing_ids: set[str] = set()
        else:
            existing_ids = {r[0] for r in conn.execute(f"SELECT id FROM {TABLE}").fetchall()}

        todo = [d for d in docs if d["id"] not in existing_ids]
        if not todo:
            print(f"pgvector index complete ({len(existing_ids)}/{expected} chunks) — skipping build.")
            conn.execute(INDEX_SQL)   # cheap no-op if it already exists
            return len(existing_ids)

        if existing_ids:
            print(f"Resuming: {len(existing_ids)}/{expected} chunks already indexed.")
        print(f"Embedding + inserting {len(todo)} chunks (CPU-bound; this takes a while)…")

        done = len(existing_ids)
        for start in range(0, len(todo), INSERT_BATCH):
            batch = todo[start:start + INSERT_BATCH]
            vectors = embed_texts(
                [doc_embed_text(d) for d in batch], show_progress=False
            )
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {TABLE} (id, source, payload, embedding)"
                    " VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    [
                        (d["id"], d.get("source"), json.dumps(d, ensure_ascii=False), v)
                        for d, v in zip(batch, vectors)
                    ],
                )
            done += len(batch)
            print(f"  {done}/{expected} chunks indexed", flush=True)

        print("Building HNSW index…", flush=True)
        conn.execute(INDEX_SQL)
        total = count_chunks(conn)
        print(f"Indexed {total} documents into pgvector")
        return total


def pg_vector_search(
    query: str,
    num_results: int = 5,
    source: str | None = None,
    url: str | None = None,
) -> list[dict]:
    """Semantic search; optionally restrict to a single source (huberman/galpin)."""
    embedding = embed_query(query)
    # `<=>` is cosine distance. Embeddings are L2-normalized (see embeddings.py), so this
    # ranks identically to the SQLite backend's cosine similarity.
    if source:
        sql = (
            f"SELECT payload FROM {TABLE} WHERE source = %s"
            " ORDER BY embedding <=> %s LIMIT %s"
        )
        params = (source, embedding, num_results)
    else:
        sql = f"SELECT payload FROM {TABLE} ORDER BY embedding <=> %s LIMIT %s"
        params = (embedding, num_results)

    with connect(url) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [r[0] for r in rows]


if __name__ == "__main__":
    if not pgvector_url():
        raise SystemExit(f"{PGVECTOR_URL_ENV} is not set — nothing to build.")
    build_pg_index()
