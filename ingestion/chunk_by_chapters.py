"""
Step 3: Chunk a fetched transcript (see fetch_transcripts.py output) into
retrievable chunks. Chapters matching sponsor/outro patterns (defined
per-source in sources.yaml) are dropped entirely.

Chapters are the semantic unit, but a whole chapter is far too big to retrieve
well: Module 4's evaluation showed BOTH retrievers lose precision on chunks over
~512 tokens (keyword too, and it isn't truncated at all — so the problem is chunk
size, not embedding truncation). So chapters longer than MAX_CHUNK_TOKENS are
sub-chunked into overlapping token windows, each keeping its parent chapter's
title and its own timestamps.

Falls back to the same token-window logic for the rare episode with no chapters.
"""
import re

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Sized so chunks stay under the embedding model's 512-token input window. We count
# cl100k tokens here while the embedder counts MiniLM wordpieces (which run higher for
# the same text), hence the deliberate headroom — ingestion stays decoupled from the
# retrieval model rather than importing its tokenizer.
MAX_CHUNK_TOKENS = 350
MIN_CHUNK_TOKENS = 100
CHUNK_OVERLAP = 0.15


def _matches_any(title: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    return bool(re.search("|".join(patterns), title, re.IGNORECASE))


def _window_segments(
    segments: list[dict],
    max_tokens: int,
    overlap: float,
    min_tokens: int,
) -> list[list[dict]]:
    """
    Slide a token-window over a segment stream, returning groups of segments.

    Shared by chapter sub-chunking and the no-chapter episode fallback. Each window
    carries the trailing ~overlap fraction of the previous one so an idea split across
    a boundary still appears whole in one chunk.
    """
    overlap_tokens = int(max_tokens * overlap)

    windows: list[list[dict]] = []
    window: list[dict] = []
    window_tokens = 0

    for seg in segments:
        window.append(seg)
        window_tokens += len(_ENCODING.encode(seg["text"]))

        if window_tokens >= max_tokens:
            windows.append(list(window))
            carry: list[dict] = []
            carry_tokens = 0
            for s in reversed(window):
                t = len(_ENCODING.encode(s["text"]))
                if carry_tokens + t > overlap_tokens:
                    break
                carry.insert(0, s)
                carry_tokens += t
            window = carry
            window_tokens = carry_tokens

    # Keep the tail if it's substantial, or if it's all we have.
    if window and (window_tokens >= min_tokens or not windows):
        windows.append(list(window))

    return windows


def _as_chunk(segments: list[dict], chapter_title: str | None) -> dict:
    """Build a chunk dict from a segment group, deriving its timestamp span."""
    return {
        "chapter_title": chapter_title,
        "start_timestamp": segments[0]["start"],
        "end_timestamp": segments[-1]["start"] + segments[-1]["duration"],
        "segments": list(segments),
    }


def chunk_episode(
    transcript: dict,
    skip_chapter_patterns: list[str],
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: float = CHUNK_OVERLAP,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[dict]:
    """
    Group an episode's segments by chapter, sub-chunking any chapter that exceeds
    max_tokens. Returns one dict per chunk:
    {chapter_title, chapter_index, sub_index, start_timestamp, end_timestamp, segments}.

    `chapter_index` counts *surviving* chapters (skipped sponsor chapters don't consume
    an index) and `sub_index` counts windows within a chapter. Together they give each
    chunk a stable id, and let build_documents.py record a `parent_chunk_id` that matches
    the pre-sub-chunking id scheme — which is what keeps the existing evaluation ground
    truth valid across this change.

    Segments are passed through raw (not joined) — normalize.py does the joining +
    filler-stripping on each chunk-scoped slice.
    """
    chapters = transcript["chapters"]
    segments = transcript["segments"]

    chunks = []
    chapter_index = 0
    for chapter in chapters:
        if _matches_any(chapter["title"], skip_chapter_patterns):
            continue

        chapter_segments = [
            seg
            for seg in segments
            if chapter["start_time"] <= seg["start"] < chapter["end_time"]
        ]
        if not chapter_segments:
            continue

        for sub_index, window in enumerate(
            _window_segments(chapter_segments, max_tokens, overlap, min_tokens)
        ):
            chunk = _as_chunk(window, chapter["title"])
            chunk["chapter_index"] = chapter_index
            chunk["sub_index"] = sub_index
            chunks.append(chunk)

        chapter_index += 1

    return chunks


def chunk_by_tokens(
    transcript: dict,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: float = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Fallback for episodes with no chapters: slide a token-window over the whole
    segment stream. Same shape as chunk_episode's output, with chapter_title None
    and each window treated as its own "chapter" for id purposes.
    """
    windows = _window_segments(transcript["segments"], max_tokens, overlap, min_tokens)

    chunks = []
    for chapter_index, window in enumerate(windows):
        chunk = _as_chunk(window, None)
        chunk["chapter_index"] = chapter_index
        chunk["sub_index"] = 0
        chunks.append(chunk)
    return chunks


def chunk_transcript(
    transcript: dict,
    skip_chapter_patterns: list[str],
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: float = CHUNK_OVERLAP,
) -> list[dict]:
    """Dispatch: chapter-based chunking if chapters exist, else token-window fallback."""
    if transcript.get("chapters"):
        return chunk_episode(transcript, skip_chapter_patterns, max_tokens, overlap, min_tokens)
    return chunk_by_tokens(transcript, min_tokens, max_tokens, overlap)
