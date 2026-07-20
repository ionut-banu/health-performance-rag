"""
Module 4, step 2: evaluate retrieval — keyword (Module 1) vs vector (Module 2).

For every ground-truth (question -> chunk_id) pair, retrieve the top-10 with each backend
and record the rank of the known-relevant chunk. Reports hit-rate@k and MRR per backend
(the "multiple approaches" comparison the rubric wants), then buckets by the relevant chunk's
embedding-token length to isolate how much the vector index's 512-token truncation costs.

    uv run eval/evaluate_retrieval.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))

from schema import load_documents
from retrieve import retrieve
from embeddings import get_model, EMBEDDING_MODEL
from metrics import hit_rate, mrr

GROUND_TRUTH_PATH = "eval/ground_truth.jsonl"
DOCUMENTS_PATH = "data/documents.jsonl"
RESULTS_DIR = "eval/results"
METHODS = ["keyword", "vector"]
K_VALUES = [1, 3, 5, 10]
TOP_K = 10
TRUNCATION_LIMIT = 512  # vector index (multi-qa-MiniLM) input cap


def rank_of(chunk_id: str, results: list[dict]) -> int | None:
    for i, r in enumerate(results, 1):
        if r.get("id") == chunk_id:
            return i
    return None


def summarize(ranks: list[int | None]) -> dict:
    row = {f"hr@{k}": round(hit_rate(ranks, k), 4) for k in K_VALUES}
    row["mrr"] = round(mrr(ranks), 4)
    row["n"] = len(ranks)
    return row


def main():
    gt = [json.loads(l) for l in open(GROUND_TRUTH_PATH, encoding="utf-8") if l.strip()]
    print(f"Loaded {len(gt)} ground-truth pairs from {GROUND_TRUTH_PATH}")

    # Map chunk_id -> embedding-token length, for the truncation buckets.
    tok = get_model().tokenizer
    docs = {d.id: d for d in load_documents(DOCUMENTS_PATH)}
    chunk_tokens = {
        cid: len(tok.encode(docs[cid].text, add_special_tokens=False))
        for cid in {row["chunk_id"] for row in gt} if cid in docs
    }

    # Collect ranks per method (overall + per truncation bucket).
    ranks = {m: [] for m in METHODS}
    bucket_ranks = {m: {"<=512": [], ">512": []} for m in METHODS}

    for i, row in enumerate(gt, 1):
        cid = row["chunk_id"]
        bucket = "<=512" if chunk_tokens.get(cid, 0) <= TRUNCATION_LIMIT else ">512"
        for m in METHODS:
            r = rank_of(cid, retrieve(row["question"], num_results=TOP_K, method=m))
            ranks[m].append(r)
            bucket_ranks[m][bucket].append(r)
        if i % 100 == 0 or i == len(gt):
            print(f"  evaluated {i}/{len(gt)}")

    overall = {m: summarize(ranks[m]) for m in METHODS}
    buckets = {m: {b: summarize(bucket_ranks[m][b]) for b in ("<=512", ">512")} for m in METHODS}
    winner = max(METHODS, key=lambda m: overall[m]["mrr"])

    results = {
        "ground_truth_pairs": len(gt),
        "embedding_model": EMBEDDING_MODEL,
        "overall": overall,
        "truncation_buckets": buckets,
        "winner_by_mrr": winner,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "retrieval.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_markdown(results)

    print(f"\nWinner by MRR: {winner}")
    for m in METHODS:
        print(f"  {m:8s} {overall[m]}")


def _write_markdown(results: dict) -> None:
    lines = [
        "# Retrieval evaluation — keyword vs vector",
        "",
        f"Ground-truth pairs: **{results['ground_truth_pairs']}** · "
        f"embedding model: `{results['embedding_model']}`",
        "",
        "## Overall",
        "",
        "| Method | HR@1 | HR@3 | HR@5 | HR@10 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        o = results["overall"][m]
        lines.append(f"| {m} | {o['hr@1']} | {o['hr@3']} | {o['hr@5']} | {o['hr@10']} | {o['mrr']} |")
    lines += [
        "",
        f"**Winner by MRR: `{results['winner_by_mrr']}`**",
        "",
        "## Truncation buckets (relevant chunk length in embedding tokens)",
        "",
        "Vector embeds only the first 512 tokens of a chunk. If truncation hurts, vector's",
        "score should drop sharply on the `>512` bucket while keyword (full-text) holds.",
        "",
        "| Method | Bucket | n | HR@5 | MRR |",
        "|---|---|---|---|---|",
    ]
    for m in METHODS:
        for b in ("<=512", ">512"):
            s = results["truncation_buckets"][m][b]
            lines.append(f"| {m} | {b} | {s['n']} | {s['hr@5']} | {s['mrr']} |")
    lines.append("")
    with open(os.path.join(RESULTS_DIR, "retrieval.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
