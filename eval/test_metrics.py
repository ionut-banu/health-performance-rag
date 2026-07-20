"""Unit tests for the pure retrieval metrics.

Run with pytest if installed (`uv run python -m pytest eval/ -q`), or standalone
with no dependencies: `uv run eval/test_metrics.py`.
"""
from metrics import hit_rate, mrr

RANKS = [1, 2, None, 5]  # relevant chunk found at rank 1, 2, missed, and 5


def test_hit_rate_at_k():
    assert hit_rate(RANKS, 1) == 0.25   # only the rank-1 hit
    assert hit_rate(RANKS, 3) == 0.50   # ranks 1 and 2
    assert hit_rate(RANKS, 5) == 0.75   # ranks 1, 2, 5
    assert hit_rate(RANKS, 10) == 0.75  # None never counts


def test_mrr():
    # (1/1 + 1/2 + 0 + 1/5) / 4 = 1.7 / 4
    assert abs(mrr(RANKS) - 0.425) < 1e-9


def test_empty():
    assert hit_rate([], 5) == 0.0
    assert mrr([]) == 0.0


def test_all_missed():
    assert hit_rate([None, None], 10) == 0.0
    assert mrr([None, None]) == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all metrics tests passed")
