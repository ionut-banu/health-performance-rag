"""
Module 4: retrieval metrics.

Pure functions over a list of "ranks" — for each evaluation query, the 1-based rank
at which the known-relevant chunk appeared in the retrieved results, or None if it
was not retrieved at all. No I/O, so these are trivially unit-testable.
"""


def hit_rate(ranks: list[int | None], k: int) -> float:
    """Fraction of queries whose relevant chunk appeared in the top-k."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank: average of 1/rank (0 for misses)."""
    if not ranks:
        return 0.0
    return sum((1.0 / r) if r is not None else 0.0 for r in ranks) / len(ranks)
