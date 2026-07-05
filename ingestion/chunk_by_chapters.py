"""
Step 3: Chunk a fetched transcript (see fetch_transcripts.py output) into
one chunk per chapter. Chapters matching sponsor/outro patterns (defined
per-source in sources.yaml) are dropped entirely.

Falls back to token-window chunking for the rare episode with no chapters.
"""
import re

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _matches_any(title: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    return bool(re.search("|".join(patterns), title, re.IGNORECASE))


def chunk_episode(transcript: dict, skip_chapter_patterns: list[str]) -> list[dict]:
    """
    Group an episode's segments by chapter window. Returns one dict per
    surviving chapter: {chapter_title, start_timestamp, end_timestamp, segments}.
    Segments are passed through raw (not joined) — normalize.py does the
    joining + filler-stripping on this chapter-scoped slice.
    """
    chapters = transcript["chapters"]
    segments = transcript["segments"]

    chunks = []
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

        chunks.append(
            {
                "chapter_title": chapter["title"],
                "start_timestamp": chapter["start_time"],
                "end_timestamp": chapter["end_time"],
                "segments": chapter_segments,
            }
        )

    return chunks


def chunk_by_tokens(
    transcript: dict,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: float = 0.15,
) -> list[dict]:
    """
    Fallback for episodes with no chapters: slide a token-window over the
    segment stream. Returns one dict per window, same shape as
    chunk_episode's output (chapter_title is None).
    """
    segments = transcript["segments"]
    overlap_tokens = int(max_tokens * overlap)

    chunks = []
    window: list[dict] = []
    window_tokens = 0

    def flush():
        if not window:
            return
        chunks.append(
            {
                "chapter_title": None,
                "start_timestamp": window[0]["start"],
                "end_timestamp": window[-1]["start"] + window[-1]["duration"],
                "segments": list(window),
            }
        )

    for seg in segments:
        seg_tokens = len(_ENCODING.encode(seg["text"]))
        window.append(seg)
        window_tokens += seg_tokens

        if window_tokens >= max_tokens:
            flush()
            # Carry the trailing ~overlap_tokens worth of segments into the next window.
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

    if window_tokens >= min_tokens or not chunks:
        flush()

    return chunks


def chunk_transcript(
    transcript: dict,
    skip_chapter_patterns: list[str],
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap: float = 0.15,
) -> list[dict]:
    """Dispatch: chapter-based chunking if chapters exist, else token-window fallback."""
    if transcript.get("chapters"):
        return chunk_episode(transcript, skip_chapter_patterns)
    return chunk_by_tokens(transcript, min_tokens, max_tokens, overlap)
