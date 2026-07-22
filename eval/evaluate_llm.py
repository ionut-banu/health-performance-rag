"""
Module 4, step 3: evaluate the final LLM answer — basic RAG vs agentic RAG.

For a sample of ground-truth questions, generate an answer with each generation approach
(both on the winning retriever from evaluate_retrieval.py) and score it with an LLM-as-judge
against the known-relevant chunk. Reports the label distribution and a mean score per approach
— the "multiple approaches" comparison the rubric wants for LLM evaluation.

    uv run eval/evaluate_llm.py
"""
import argparse
import json
import os
import random
import sys

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))

from schema import load_documents
from rag import rag, agentic_rag
from evaluate_retrieval import ALL_APPROACHES  # single source of truth for approach -> kwargs

load_dotenv()

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GROUND_TRUTH_PATH = "eval/ground_truth.jsonl"
DOCUMENTS_PATH = "data/documents.jsonl"
RESULTS_DIR = "eval/results"
RETRIEVAL_RESULTS = "eval/results/retrieval.json"
SAMPLE_SIZE = 1
SEED = 42
SCORE = {"RELEVANT": 1.0, "PARTLY_RELEVANT": 0.5, "NON_RELEVANT": 0.0}

_client = OpenAI()

JUDGE_PROMPT = """You are grading a health/fitness/nutrition Q&A assistant.

Given a user QUESTION, the REFERENCE excerpt the question was derived from, and the assistant's \
ANSWER, judge whether the answer correctly and relevantly addresses the question and is grounded \
in the reference.

Return JSON exactly as: {{"label": "RELEVANT" | "PARTLY_RELEVANT" | "NON_RELEVANT", "reason": "..."}}
- RELEVANT: directly and correctly answers the question, consistent with the reference.
- PARTLY_RELEVANT: on-topic but incomplete, vague, or partly unsupported.
- NON_RELEVANT: off-topic, wrong, or "I don't know" when the reference clearly answers it.

QUESTION: {question}

REFERENCE:
{reference}

ANSWER:
{answer}"""

# The two generation approaches under test (both use the winning retrieval config).
APPROACHES = {
    "basic_rag": lambda q, kwargs: rag(q, **kwargs),
    "agentic_rag": lambda q, kwargs: agentic_rag(q, **kwargs),
}

FALLBACK_RETRIEVER = "hybrid+rerank"


def pick_retriever() -> tuple[str, dict]:
    """
    Resolve the winning retrieval approach into (label, retrieve() kwargs).

    evaluate_retrieval.py records compound labels like "hybrid+rerank", which are NOT
    valid `method` values — re-ranking and rewriting are separate flags. Translate through
    ALL_APPROACHES so the two scripts can't drift apart again.
    """
    label = FALLBACK_RETRIEVER
    if os.path.exists(RETRIEVAL_RESULTS):
        winner = json.load(open(RETRIEVAL_RESULTS)).get("winner_by_mrr")
        if winner:
            label = winner
    if label not in ALL_APPROACHES:
        raise ValueError(
            f"Winning approach {label!r} from {RETRIEVAL_RESULTS} is not defined in "
            f"evaluate_retrieval.ALL_APPROACHES ({list(ALL_APPROACHES)})."
        )
    return label, dict(ALL_APPROACHES[label])


def judge(question: str, reference: str, answer: str) -> dict:
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference[:4000], answer=answer)}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()

    gt = [json.loads(l) for l in open(GROUND_TRUTH_PATH, encoding="utf-8") if l.strip()]
    rng = random.Random(SEED)
    sample = rng.sample(gt, min(args.sample_size, len(gt)))
    docs = {d.id: d for d in load_documents(DOCUMENTS_PATH)}
    label, retrieve_kwargs = pick_retriever()
    endpoint = os.environ.get("OPENAI_BASE_URL", "api.openai.com")
    # Recorded because a stray OPENAI_BASE_URL/OPENAI_API_KEY in the shell silently
    # redirects these calls (env vars beat .env) — judge scores from a different model
    # aren't comparable, so the results must say which one produced them.
    print(f"Provider: model={MODEL} endpoint={endpoint}")
    print(
        f"Judging {len(sample)} questions · retriever=`{label}` {retrieve_kwargs} · "
        f"approaches={list(APPROACHES)}"
    )

    per_approach = {name: [] for name in APPROACHES}   # list of (label, score)
    errors = {name: 0 for name in APPROACHES}
    for i, row in enumerate(sample, 1):
        ref = docs[row["chunk_id"]].text if row["chunk_id"] in docs else ""
        for name, gen in APPROACHES.items():
            try:
                answer = gen(row["question"], retrieve_kwargs)
                verdict = judge(row["question"], ref, answer)
            except Exception as e:
                # Never score a failed call — an infrastructure error is not evidence that
                # the answer was bad, and silently recording it as NON_RELEVANT would
                # understate the approach while looking like a normal result.
                print(f"  [{i}] {name}: ERROR, excluded from scoring ({e})")
                errors[name] += 1
                continue
            verdict_label = verdict.get("label", "NON_RELEVANT")
            per_approach[name].append((verdict_label, SCORE.get(verdict_label, 0.0)))
        if i % 10 == 0 or i == len(sample):
            print(f"  judged {i}/{len(sample)}")

    summary = {}
    for name, records in per_approach.items():
        n = len(records)
        dist = {lbl: sum(1 for l, _ in records if l == lbl) for lbl in SCORE}
        summary[name] = {
            "n": n,
            "errors": errors[name],
            "mean_score": round(sum(s for _, s in records) / n, 4) if n else None,
            "pct_relevant": round(100 * dist["RELEVANT"] / n, 1) if n else None,
            "labels": dist,
        }

    scored = {k: v for k, v in summary.items() if v["n"]}
    if not scored:
        raise RuntimeError(
            "Every generation call failed — no answers were scored, so these results would "
            "be meaningless. See the ERROR lines above for the cause. Nothing was written."
        )
    winner = max(scored, key=lambda a: scored[a]["mean_score"])
    results = {"judge_model": MODEL, "judge_endpoint": endpoint,
               "retriever": label, "retriever_kwargs": retrieve_kwargs,
               "sample_size": len(sample), "approaches": summary,
               "winner_by_mean_score": winner}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "llm_judge.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_markdown(results)

    print(f"\nWinner by mean score: {winner}")
    for name, s in summary.items():
        print(f"  {name:12s} {s}")


def _write_markdown(results: dict) -> None:
    lines = [
        "# LLM evaluation — basic RAG vs agentic RAG (LLM-as-judge)",
        "",
        f"Sample: **{results['sample_size']}** questions · retriever: `{results['retriever']}` · "
        "judge scores RELEVANT=1.0 / PARTLY=0.5 / NON=0.0",
        "",
        "`n` counts successfully scored answers; `errors` are calls that failed and were "
        "excluded from the score rather than counted as bad answers.",
        "",
        "| Approach | n | errors | Mean score | % RELEVANT | RELEVANT | PARTLY | NON |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, s in results["approaches"].items():
        d = s["labels"]
        lines.append(
            f"| {name} | {s['n']} | {s['errors']} | {s['mean_score']} | {s['pct_relevant']}% | "
            f"{d['RELEVANT']} | {d['PARTLY_RELEVANT']} | {d['NON_RELEVANT']} |"
        )
    lines += ["", f"**Winner by mean score: `{results['winner_by_mean_score']}`**", ""]
    with open(os.path.join(RESULTS_DIR, "llm_judge.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
