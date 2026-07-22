"""
Module 2: qualitative side-by-side of keyword (Module 1) vs vector (Module 2) retrieval.

For each sample query, prints the top-k chunks from each backend so you can eyeball
where semantic search surfaces relevant episodes that lexical search misses (and vice
versa). This is a sanity check, not a metric — the rigorous hit-rate/MRR evaluation lives in
eval/evaluate_retrieval.py.

    uv run rag/compare_retrieval.py
"""
from retrieve import retrieve

# Queries chosen to include lexical/semantic mismatches: wording unlikely to appear
# verbatim in the transcripts, where embeddings should beat keyword matching.
SAMPLE_QUERIES = [
    "how do I fall asleep faster?",
    "what should I eat to build muscle?",
    "ways to boost focus and concentration",
    "is it bad to drink coffee late in the day?",
    "how to get better at endurance",
]

TOP_K = 3


def _fmt(chunk: dict) -> str:
    ts = chunk.get("start_timestamp")
    ts_str = f" @{int(ts)}s" if ts is not None else ""
    chapter = chunk.get("chapter_title") or ""
    tail = f" — {chapter}" if chapter else ""
    return f"    [{chunk['source']}] {chunk['title']}{tail}{ts_str}"


def main():
    for query in SAMPLE_QUERIES:
        print("=" * 100)
        print(f"QUERY: {query}\n")
        for method in ("keyword", "vector"):
            print(f"  {method.upper()}:")
            for chunk in retrieve(query, num_results=TOP_K, method=method):
                print(_fmt(chunk))
            print()


if __name__ == "__main__":
    main()
