"""
Step 2: Normalize raw transcript segments — strip sponsor reads, intros,
and filler before chunking. Patterns are defined per-source in sources.yaml,
not hardcoded here — edit that file to tune filtering, not this script.

Auto-generated YouTube captions carry no sentence punctuation, so filler
matching can't split on ".!?" — instead it slides a fixed-size window over
consecutive segments and drops windows that match a filler pattern.
"""
import re

_WINDOW_SIZE = 12  # segments per filler-matching window — enough to hold a full sponsor phrase


def join_segments(segments: list[dict]) -> str:
    """Join transcript segments into a single text blob."""
    return " ".join(seg["text"].strip() for seg in segments if seg["text"].strip())


def strip_filler(segments: list[dict], filler_patterns: list[str]) -> list[dict]:
    """Drop segments that fall inside a run matching a known filler pattern."""
    if not filler_patterns:
        return segments
    filler_re = re.compile("|".join(filler_patterns), re.IGNORECASE)

    to_drop = set()
    for start in range(len(segments)):
        window = segments[start : start + _WINDOW_SIZE]
        window_text = join_segments(window)
        if filler_re.search(window_text):
            to_drop.update(range(start, start + len(window)))

    return [seg for i, seg in enumerate(segments) if i not in to_drop]


def normalize(segments: list[dict], filler_patterns: list[str]) -> str:
    kept_segments = strip_filler(segments, filler_patterns)
    return join_segments(kept_segments)
