"""
Module 5: interaction logging + user feedback.

Every answer the app produces is written here, and a thumbs up/down updates that row.
The dashboard reads the same table — this is the only source of monitoring data, so
nothing is ever seeded synthetically.

SQLite keeps it infra-free (same choice as the sqlitesearch index); the file lives under
data/, which is gitignored.
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

# Mutable state, deliberately kept OUT of data/. data/ holds the shipped knowledge base,
# which is baked into the image; mounting a volume over it would shadow the image's copy
# (Docker only seeds a named volume when it is first created), so a rebuilt image with an
# updated documents.jsonl would never reach a running container. Feedback therefore lives
# on its own path, and only that path is volume-mounted.
FEEDBACK_DB_PATH = os.environ.get("FEEDBACK_DB_PATH", "data/feedback.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id           TEXT PRIMARY KEY,
    ts           TEXT NOT NULL,      -- ISO-8601 UTC
    question     TEXT NOT NULL,
    answer       TEXT NOT NULL,
    retriever    TEXT,               -- keyword | vector | hybrid
    rerank       INTEGER,            -- 0/1
    agentic      INTEGER,            -- 0/1
    num_results  INTEGER,
    latency_ms   REAL,
    sources      TEXT,               -- JSON: [{title, source, url}]
    vote         INTEGER             -- NULL until voted, then 1 / -1
)
"""


def _connect(db_path: str = FEEDBACK_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    return conn


def log_interaction(
    question: str,
    answer: str,
    sources: list[dict],
    retriever: str,
    rerank: bool,
    agentic: bool,
    num_results: int,
    latency_ms: float,
    db_path: str = FEEDBACK_DB_PATH,
) -> str:
    """Record an answered question. Returns the interaction id used to attach a vote."""
    interaction_id = str(uuid.uuid4())
    trimmed = [
        {"title": s.get("title"), "source": s.get("source"), "url": s.get("url")}
        for s in sources
    ]
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO interactions (id, ts, question, answer, retriever, rerank,"
            " agentic, num_results, latency_ms, sources, vote)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (
                interaction_id,
                datetime.now(timezone.utc).isoformat(),
                question,
                answer,
                retriever,
                int(rerank),
                int(agentic),
                num_results,
                latency_ms,
                json.dumps(trimmed, ensure_ascii=False),
            ),
        )
    return interaction_id


def record_vote(interaction_id: str, vote: int, db_path: str = FEEDBACK_DB_PATH) -> None:
    """Attach a thumbs up (1) or down (-1) to an interaction. Re-voting overwrites."""
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE interactions SET vote = ? WHERE id = ?", (vote, interaction_id)
        )


def load_interactions(db_path: str = FEEDBACK_DB_PATH) -> list[dict]:
    """All interactions, newest first, with `sources` decoded back into a list."""
    if not os.path.exists(db_path):
        return []
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM interactions ORDER BY ts DESC").fetchall()
    out = []
    for r in rows:
        row = dict(r)
        try:
            row["sources"] = json.loads(row["sources"] or "[]")
        except json.JSONDecodeError:
            row["sources"] = []
        out.append(row)
    return out
