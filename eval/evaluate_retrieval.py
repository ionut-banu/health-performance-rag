"""
Evaluate retrieval approaches against the ground-truth set.

For every ground-truth (question -> chunk_id) pair, retrieve the top-10 with each approach
and record the rank of the known-relevant chunk, then report hit-rate@k and MRR. This is the
"multiple approaches evaluated, best one used" evidence for the rubric, and the harness that
proved out each Module 6 change.

    uv run eval/evaluate_retrieval.py                              # free approaches, all 750 pairs
    uv run eval/evaluate_retrieval.py --approaches hybrid,hybrid+rewrite --limit 250

`hybrid+rewrite` costs one LLM call per question, so it's excluded by default — run it
explicitly with --limit to control spend.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))

from schema import load_documents
from retrieve import retrieve
from embeddings import EMBEDDING_MODEL
from metrics import hit_rate, mrr

GROUND_TRUTH_PATH = "eval/ground_truth.jsonl"
DOCUMENTS_PATH = "data/documents.jsonl"
RESULTS_DIR = "eval/results"
K_VALUES = [1, 3, 5, 10]
TOP_K = 10
SEED = 42

# label -> retrieve() kwargs.
# Every flag is pinned explicitly — never rely on retrieve()'s defaults here. Those defaults
# track whichever configuration currently wins, so an approach that omitted `rerank` would
# silently change meaning the moment the production default flipped, making old and new runs
# incomparable while still looking like valid numbers.
ALL_APPROACHES = {
    "keyword": {"method": "keyword", "rerank": False, "rewrite": False},
    "vector": {"method": "vector", "rerank": False, "rewrite": False},
    "hybrid": {"method": "hybrid", "rerank": False, "rewrite": False},
    "hybrid+rerank": {"method": "hybrid", "rerank": True, "rewrite": False},
    "hybrid+rewrite": {"method": "hybrid", "rerank": False, "rewrite": True},
    "hybrid+rerank+rewrite": {"method": "hybrid", "rerank": True, "rewrite": True},
}
# Everything except the one that makes an LLM call per question.
DEFAULT_APPROACHES = ["keyword", "vector", "hybrid", "hybrid+rerank"]

# Since Module 6 every chunk fits the embedding window, so the old truncated/not-truncated
# split is moot. Bucket instead by whether the ground-truth chapter actually needed
# splitting — that's the population sub-chunking targeted, so it shows if the fix landed.
SHORT, LONG = "short (1 sub-chunk)", "long (2+ sub-chunks)"
BUCKETS = (SHORT, LONG)


def rank_of(chunk_id: str, results: list[dict]) -> int | None:
    """
    Rank of the relevant chunk, or None if absent.

    Ground truth was collected on chapter-sized chunks; after Module 6's sub-chunking a
    chapter is split into several documents whose `parent_chunk_id` is the original
    chapter id. Matching on either id keeps the same 750 pairs valid across both corpora,
    so before/after numbers are directly comparable.
    """
    for i, r in enumerate(results, 1):
        if r.get("id") == chunk_id or r.get("parent_chunk_id") == chunk_id:
            return i
    return None


def summarize(ranks: list[int | None]) -> dict:
    row = {f"hr@{k}": round(hit_rate(ranks, k), 4) for k in K_VALUES}
    row["mrr"] = round(mrr(ranks), 4)
    row["n"] = len(ranks)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approaches",
        default=",".join(DEFAULT_APPROACHES),
        help=f"Comma-separated. Available: {','.join(ALL_APPROACHES)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on a random subsample.")
    parser.add_argument("--out", default="retrieval", help="Basename under eval/results/.")
    args = parser.parse_args()

    approaches = [a.strip() for a in args.approaches.split(",") if a.strip()]
    unknown = [a for a in approaches if a not in ALL_APPROACHES]
    if unknown:
        parser.error(f"Unknown approaches: {unknown}. Available: {list(ALL_APPROACHES)}")

    gt = [json.loads(l) for l in open(GROUND_TRUTH_PATH, encoding="utf-8") if l.strip()]
    if args.limit and args.limit < len(gt):
        gt = random.Random(SEED).sample(gt, args.limit)
    print(f"Evaluating {approaches} on {len(gt)} ground-truth pairs")

    # How many sub-chunks each ground-truth chapter was split into (1 = never needed it).
    sub_counts = Counter(
        d.metadata.get("parent_chunk_id") for d in load_documents(DOCUMENTS_PATH)
    )

    ranks = {a: [] for a in approaches}
    bucket_ranks = {a: {b: [] for b in BUCKETS} for a in approaches}

    for i, row in enumerate(gt, 1):
        cid = row["chunk_id"]
        bucket = SHORT if sub_counts.get(cid, 0) <= 1 else LONG
        for a in approaches:
            results = retrieve(row["question"], num_results=TOP_K, **ALL_APPROACHES[a])
            r = rank_of(cid, results)
            ranks[a].append(r)
            bucket_ranks[a][bucket].append(r)
        if i % 50 == 0 or i == len(gt):
            print(f"  evaluated {i}/{len(gt)}")

    overall = {a: summarize(ranks[a]) for a in approaches}
    buckets = {a: {b: summarize(bucket_ranks[a][b]) for b in BUCKETS} for a in approaches}
    winner = max(approaches, key=lambda a: overall[a]["mrr"])

    results = {
        "ground_truth_pairs": len(gt),
        "embedding_model": EMBEDDING_MODEL,
        "approaches": approaches,
        "overall": overall,
        "buckets": buckets,
        "winner_by_mrr": winner,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{args.out}.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_markdown(results, args.out)

    print(f"\nWinner by MRR: {winner}")
    for a in approaches:
        print(f"  {a:16s} {overall[a]}")


def _write_markdown(results: dict, out: str) -> None:
    approaches = results["approaches"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"Ground-truth pairs: **{results['ground_truth_pairs']}** · "
        f"embedding model: `{results['embedding_model']}`",
        "",
        "## Overall",
        "",
        "| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for a in approaches:
        o = results["overall"][a]
        lines.append(f"| {a} | {o['hr@1']} | {o['hr@3']} | {o['hr@5']} | {o['hr@10']} | {o['mrr']} |")
    lines += [
        "",
        f"**Winner by MRR: `{results['winner_by_mrr']}`**",
        "",
        "## Chapter-size buckets",
        "",
        "Split by whether the ground-truth chapter actually needed sub-chunking. The `long`",
        "bucket is the population Module 6 targeted, so that's where the gain should show.",
        "",
        "| Approach | Bucket | n | HR@5 | MRR |",
        "|---|---|---|---|---|",
    ]
    for a in approaches:
        for b in BUCKETS:
            s = results["buckets"][a][b]
            lines.append(f"| {a} | {b} | {s['n']} | {s['hr@5']} | {s['mrr']} |")
    lines.append("")
    with open(os.path.join(RESULTS_DIR, f"{out}.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
