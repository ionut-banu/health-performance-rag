"""
Unified document schema for all ingested content (transcripts + recipes).
See CLAUDE.md for the full spec.
"""
from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class Document:
    id: str
    source: Literal["huberman", "galpin", "recipe"]
    content_type: Literal["transcript", "recipe"]
    title: str
    url: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def save_documents(docs: list[Document], path: str) -> None:
    """Save a list of Documents as JSONL — one JSON object per line."""
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    print(f"Saved {len(docs)} documents to {path}")


def load_documents(path: str) -> list[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(Document(**json.loads(line)))
    return docs
