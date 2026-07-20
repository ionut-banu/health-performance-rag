"""
Module 4, step 1: generate a ground-truth evaluation set.

For a stratified sample of chunks from data/documents.jsonl, ask the LLM to write
realistic user questions that the chunk answers. Each (question -> chunk_id) pair is a
labeled retrieval example: a retriever "hits" if it returns that chunk for that question.

Questions are phrased as a real user would (not quoting the transcript) so keyword search
can't win trivially on lexical overlap. The full chunk text is sent to the generator so
questions can target content anywhere in a chapter — including the tail that the vector
index truncates at 512 tokens (that gap is what evaluate_retrieval.py measures).

    uv run eval/generate_ground_truth.py --sample-size 150
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import load_documents

load_dotenv()

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
DOCUMENTS_PATH = "data/documents.jsonl"
OUTPUT_PATH = "eval/ground_truth.jsonl"
MIN_CHUNK_TOKENS = 100          # skip trivially short chunks
MAX_PROMPT_TOKENS = 4000        # cap the rare giant chunks sent to the generator
SEED = 42

_client = OpenAI()
_enc = tiktoken.get_encoding("cl100k_base")

PROMPT = """You are building an evaluation set for a health, fitness, and nutrition Q&A \
system grounded in podcast transcripts.

Given the transcript excerpt below, write {n} distinct questions a real user might ask that \
are specifically answered by THIS excerpt. Rules:
- Phrase them naturally, the way a user would type them — do NOT quote or closely paraphrase \
the wording of the excerpt.
- Each question must be answerable from this excerpt alone, and specific enough that this \
excerpt is a strong answer (avoid generic questions any episode could answer).
- Cover different parts of the excerpt, not just the opening.

Return JSON exactly as: {{"questions": ["...", "..."]}}

Episode: {title}
Chapter: {chapter}
Excerpt:
{text}"""


def cap_text(text: str) -> str:
    toks = _enc.encode(text)
    if len(toks) <= MAX_PROMPT_TOKENS:
        return text
    return _enc.decode(toks[:MAX_PROMPT_TOKENS])


def stratified_sample(docs, sample_size: int) -> list:
    """Sample chunks proportionally per source, deterministically."""
    by_source = defaultdict(list)
    for d in docs:
        if len(_enc.encode(d.text)) >= MIN_CHUNK_TOKENS:
            by_source[d.source].append(d)

    rng = random.Random(SEED)
    total = sum(len(v) for v in by_source.values())
    sampled = []
    for source, items in sorted(by_source.items()):
        n = round(sample_size * len(items) / total)
        sampled.extend(rng.sample(items, min(n, len(items))))
    rng.shuffle(sampled)
    return sampled


def load_done_chunk_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["chunk_id"])
    print(f"Resuming — {len(done)} chunks already in {path}, skipping those.")
    return done


def generate_questions(doc, n: int) -> list[str]:
    prompt = PROMPT.format(
        n=n,
        title=doc.title,
        chapter=doc.metadata.get("chapter_title") or "(none)",
        text=cap_text(doc.text),
    )
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    data = json.loads(resp.choices[0].message.content)
    return [q.strip() for q in data.get("questions", []) if q.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--questions-per-chunk", type=int, default=5)
    args = parser.parse_args()

    docs = load_documents(DOCUMENTS_PATH)
    sampled = stratified_sample(docs, args.sample_size)
    done = load_done_chunk_ids(OUTPUT_PATH)
    todo = [d for d in sampled if d.id not in done]
    print(f"Sampled {len(sampled)} chunks; generating for {len(todo)} new ones.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    written = 0
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for i, doc in enumerate(todo, 1):
            try:
                questions = generate_questions(doc, args.questions_per_chunk)
            except Exception as e:
                print(f"  [{i}/{len(todo)}] {doc.id}: SKIPPED ({e})")
                continue
            for q in questions:
                f.write(json.dumps(
                    {"question": q, "chunk_id": doc.id, "source": doc.source},
                    ensure_ascii=False,
                ) + "\n")
            f.flush()
            written += len(questions)
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {written} questions so far")

    print(f"Wrote {written} new question->chunk pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
