# Evaluation

Every retrieval and generation choice in this project is measured against a fixed
ground-truth set, and the winner is wired in as the default. This is the evidence for the
rubric's *"multiple approaches are evaluated, and the best one is used."*

## Summary

| Axis | Approaches compared | Winner | Default? |
|---|---|---|---|
| Retrieval | keyword · vector · hybrid · hybrid+rerank | **hybrid + cross-encoder re-rank** (MRR 0.614 on SQLite, 0.626 on pgvector) | ✅ yes |
| Vector store | sqlitesearch vs Postgres/pgvector | pgvector, marginally (no regression) | ✅ in Docker |
| Query rewriting | with vs without | **without** — rewriting measurably *hurt* | ✅ off |
| Generation | basic `rag()` vs `agentic_rag()` | tie (0.783 vs 0.800) | basic (cost/latency) |

**Headline:** retrieval MRR went **0.413 → 0.614 (+49%)** and HR@1 **0.316 → 0.523 (+65%)**
across Modules 4→6, measured on the *same* 750 pairs.

## Methodology

- **Ground truth** ([eval/ground_truth.jsonl](../eval/ground_truth.jsonl)) — 750
  `question → chunk_id` pairs. For 150 chunks (stratified by source) the LLM wrote ~5 realistic
  user questions each, phrased *not* to quote the transcript so keyword search can't win on
  lexical overlap. Script: [generate_ground_truth.py](../eval/generate_ground_truth.py).
- **Retrieval** — retrieve top-10 per approach; a "hit" means the source chunk appears.
  Report **hit-rate@k** and **MRR** ([metrics.py](../eval/metrics.py), unit-tested).
  Script: [evaluate_retrieval.py](../eval/evaluate_retrieval.py).
- **Comparability across the Module 6 re-chunk** — sub-chunking renumbers documents, which would
  normally invalidate the ground truth. Each sub-chunk therefore records a `parent_chunk_id`
  equal to the id its chapter had *before* splitting, and the hit test accepts either id. The
  same 750 pairs score both corpora, so before/after numbers are directly comparable.
- **Generation** — LLM-as-judge: `gpt-4o-mini` grades each answer against the reference chunk as
  RELEVANT (1.0) / PARTLY (0.5) / NON (0.0), on a 60-question sample.
  Script: [evaluate_llm.py](../eval/evaluate_llm.py).

## Retrieval results

Current corpus: 27,085 sub-chunks (see [Module 6](#module-6-what-actually-moved-the-needle)).

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.308 | 0.444 | 0.499 | 0.601 | 0.394 |
| vector | 0.356 | 0.469 | 0.529 | 0.599 | 0.428 |
| hybrid (RRF) | 0.387 | 0.571 | 0.640 | 0.731 | 0.495 |
| **hybrid + re-rank** | **0.523** | **0.677** | **0.741** | **0.787** | **0.614** |

Baseline for comparison — the Module 4 numbers on the old chapter-sized corpus
([retrieval_baseline_chapters.md](../eval/results/retrieval_baseline_chapters.md)):
keyword MRR 0.377, vector MRR 0.413.

## Module 6: what actually moved the needle

**1. Sub-chunking (modest, and it helped both retrievers).** Module 4 found that chunks over the
embedding window scored badly — but keyword search, which is never truncated, lost just as much.
That ruled out truncation as the cause and pointed at chunk *size*: one representation can't
localize a specific moment in a 15-minute chapter. Splitting chapters into ~350-token overlapping
windows took the corpus from 5,269 → 27,085 chunks and **0% now exceed the embedding window**
(was ~93%). Gains were real but modest — keyword MRR 0.377→0.394, vector 0.413→0.428 — which is
itself the useful finding: chunking was a *precondition*, not the payoff.

**2. Hybrid + re-ranking (the payoff).** Fusing keyword and vector with Reciprocal Rank Fusion
added +0.07 MRR over vector alone, and the cross-encoder re-ranker added another +0.12. The
re-ranker helps most at the sharp end — **HR@1 jumped 0.387 → 0.523** — because it reads the query
and chunk *together* rather than comparing pre-computed representations. Sub-chunking is what made
this affordable: chunks now fit the cross-encoder's window.

**3. Query rewriting made things worse.** Evaluated on a 250-pair subsample
([retrieval_rewrite.md](../eval/results/retrieval_rewrite.md)):

| Approach | HR@1 | HR@5 | MRR |
|---|---|---|---|
| hybrid+rerank | 0.500 | 0.744 | **0.602** |
| hybrid+rerank+rewrite | 0.396 | 0.624 | 0.491 |

Rewriting *lost* 0.11 MRR. The reason is visible in the rewrites themselves: *"is coffee bad for
me at night?"* becomes *"effects of nocturnal caffeine consumption on health"* — fluent, but
**more generic**. It discards the specific details that pinned one chunk, so the query now matches
many chunks weakly. Evaluation questions are already well-formed and specific, which is precisely
the case where rewriting has nothing to add and something to lose. It ships implemented but
**disabled by default** (`rewrite=True` to opt in); it would likely earn its keep on vague or
multi-turn conversational input, which this eval set doesn't contain.

### Chapter-size buckets

Split by whether the ground-truth chapter actually needed sub-chunking:

| Approach | Bucket | n | HR@5 | MRR |
|---|---|---|---|---|
| keyword | long (2+ sub-chunks) | 730 | 0.496 | 0.390 |
| vector | long (2+ sub-chunks) | 730 | 0.540 | 0.437 |
| hybrid | long (2+ sub-chunks) | 730 | 0.643 | 0.498 |
| hybrid+rerank | long (2+ sub-chunks) | 730 | 0.744 | 0.613 |

The `long` bucket is 730 of 750 pairs — the population the fix targeted — and it carries the full
gain. The `short` bucket (n=20, ~4 chapters) is too small to read into; vector scores oddly low
there, but that's a handful of short intro-style chapters, not a trend.

## Module 7: does moving to Postgres + pgvector regress retrieval?

Containerization swapped the vector store from `sqlitesearch` to Postgres + pgvector (HNSW,
cosine). Because that changes the approximate-nearest-neighbour implementation, it can't be
assumed safe — so the **same 750 pairs** were re-run against the containerized index
([retrieval_pgvector.md](../eval/results/retrieval_pgvector.md)):

| Approach | MRR (sqlitesearch) | MRR (pgvector) | Δ |
|---|---|---|---|
| keyword | 0.3936 | 0.3936 | **0.0000** |
| vector | 0.4284 | 0.4914 | +0.0630 |
| hybrid | 0.4948 | 0.5147 | +0.0199 |
| **hybrid+rerank** | 0.6135 | **0.6257** | +0.0122 |

No regression — pgvector is slightly *better*, mostly on pure vector search, and
`hybrid+rerank` remains the winner.

**The keyword row is the control.** Keyword search is `minsearch` and never touches the vector
store, so it must be unchanged — and it is, to four decimals. That's what makes the rest of the
comparison believable: if the harness had drifted, this row would have moved.

> ⚠️ **Methodology bug worth recording.** The first pgvector run reported large gains across the
> board, including keyword "improving" from 0.394 to 0.550 — impossible, since the vector backend
> can't affect keyword-only retrieval. The cause: `ALL_APPROACHES` specified only `method` and let
> `rerank` fall through to `retrieve()`'s default, which had been flipped to `True` when the
> Module 6 winner was wired in. Every approach was silently re-ranked, so "hybrid" and
> "hybrid+rerank" were literally the same run. Evaluation definitions now pin every flag
> explicitly — they must never inherit production defaults, because those defaults intentionally
> track whatever currently wins, which silently makes old and new runs incomparable while still
> producing plausible-looking numbers.

## Generation results (LLM-as-judge)

| Approach | n | Mean score | % RELEVANT | RELEVANT | PARTLY | NON |
|---|---|---|---|---|---|---|
| basic_rag | 60 | 0.783 | 73.3% | 44 | 6 | 10 |
| agentic_rag | 60 | **0.800** | 73.3% | 44 | 8 | 8 |

Both label the same 44/60 answers RELEVANT; agentic's edge is turning 2 complete misses into
partial answers — a 0.017 difference, within noise at n=60. Since agentic makes several API calls
per answer for no reliable gain, **basic RAG stays the default**, with `--agentic` available.
(These numbers predate the Module 6 retrieval improvements, so they understate current quality.)

## Decisions applied

- **hybrid + cross-encoder re-rank** is the default retrieval path
  ([retrieve.py](../rag/retrieve.py), [rag.py](../rag/rag.py), [cli.py](../rag/cli.py)).
- **Query rewriting off** by default — measured as harmful here.
- **Basic RAG** the default generator; `--agentic` opt-in.
- `--retriever keyword|vector|hybrid` and `--no-rerank` remain available to reproduce any row above.

## Reproduce

```bash
uv run eval/generate_ground_truth.py --sample-size 150   # -> eval/ground_truth.jsonl (committed)
uv run eval/evaluate_retrieval.py                        # keyword/vector/hybrid/hybrid+rerank
uv run eval/evaluate_retrieval.py \
  --approaches hybrid+rerank,hybrid+rerank+rewrite --limit 250 --out retrieval_rewrite
uv run eval/evaluate_llm.py                              # LLM-as-judge, basic vs agentic
uv run eval/test_metrics.py                              # metric unit tests (no pytest needed)
```

Cost: ~$1 on `gpt-4o-mini` (ground truth + judging + the rewrite arm). Retrieval, embeddings, and
re-ranking all run locally and free. Ground truth and result files are committed so reviewers can
inspect the numbers without re-running.
