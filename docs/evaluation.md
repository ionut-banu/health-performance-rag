# Evaluation

Every retrieval and generation choice in this project is measured against a fixed ground-truth
set, and the winner is wired in as the default. This is the evidence for the rubric's
*"multiple approaches are evaluated, and the best one is used."*

All numbers below come from the **current** corpus (35,035 chunks, 337 episodes) and the
**current** ground truth. Result files under [eval/results/](../eval/results/) are regenerated
by the commands at the bottom.

## Summary

| Axis | Approaches compared | Winner | Default? |
|---|---|---|---|
| Retrieval | keyword · vector · hybrid · hybrid+rerank | **hybrid + cross-encoder re-rank** | ✅ yes |
| Query rewriting | with vs without | **without** — rewriting measurably hurts | ✅ off |
| Generation | basic `rag()` vs `agentic_rag()` | **basic** — better *and* cheaper | ✅ yes |
| Vector store | sqlitesearch vs Postgres/pgvector | pgvector by a hair; equivalent in practice | pgvector in Docker |

## Methodology

- **Ground truth** ([eval/ground_truth.jsonl](../eval/ground_truth.jsonl)) — 750
  `question → chunk_id` pairs. For 150 chunks (stratified by source) `gpt-4o-mini` wrote 5
  realistic user questions each, phrased *not* to quote the transcript so keyword search can't
  win on lexical overlap. Script: [generate_ground_truth.py](../eval/generate_ground_truth.py).
- **Strict matching.** Each question targets one specific ~350-token passage, and a hit requires
  that exact chunk id. Retrieving a neighbouring passage from the same chapter is a **miss**.
  This is deliberately harsh: it measures whether we find *the answer*, not merely the right
  neighbourhood.
- **Metrics** — hit-rate@k and MRR ([metrics.py](../eval/metrics.py), unit-tested in
  [test_metrics.py](../eval/test_metrics.py)).
- **Generation** — LLM-as-judge: `gpt-4o-mini` grades each answer against the reference passage
  as RELEVANT (1.0) / PARTLY (0.5) / NON (0.0), over 60 questions.
  Script: [evaluate_llm.py](../eval/evaluate_llm.py).

> **Two guardrails**, both added after they caught real errors:
> - Approach definitions pin every flag explicitly rather than inheriting `retrieve()`'s
>   defaults. Those defaults track whichever config currently wins, so an approach that omitted
>   `rerank` silently changed meaning when the production default flipped — producing
>   plausible-looking but incomparable numbers.
> - The LLM scripts print and record the model and endpoint they actually used. A stray
>   `OPENAI_BASE_URL`/`OPENAI_API_KEY` in the shell overrides `.env` and silently redirects
>   judging to a different model.
>
> Failed calls are also excluded from scores rather than counted as bad answers, and reported
> in an `errors` column — an infrastructure failure is not evidence that an answer was poor.

## Retrieval

Local `sqlitesearch` backend, all 750 pairs:

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.172 | 0.317 | 0.384 | 0.465 | 0.261 |
| vector | 0.184 | 0.316 | 0.375 | 0.441 | 0.263 |
| hybrid (RRF) | 0.224 | 0.396 | 0.483 | 0.575 | 0.331 |
| **hybrid + re-rank** | **0.331** | **0.508** | **0.555** | **0.612** | **0.429** |

Three things stand out:

1. **Keyword and vector are near-identical alone** (0.261 vs 0.263). Semantic search is good at
   landing in the right topical neighbourhood, but pinpointing one specific passage is hard for
   both. Under a looser, chapter-level bar vector looked clearly better; strict matching removes
   that advantage.
2. **Hybrid beats both** (+26% MRR over vector) precisely *because* they're comparably strong
   but fail differently — exactly the condition under which rank fusion pays off.
3. **Re-ranking is the single biggest lever** (+30% MRR over hybrid, and HR@1 0.224 → 0.331).
   The cross-encoder reads query and passage *together* instead of comparing pre-computed
   representations. It only reorders the ~20 retrieved candidates, so it cannot rescue a passage
   the first stage missed — which is why the two stages are complementary rather than redundant.

### Vector store: sqlitesearch vs pgvector

| Approach | MRR (sqlitesearch) | MRR (pgvector) |
|---|---|---|
| keyword | 0.2608 | 0.2608 |
| vector | 0.2626 | 0.2873 |
| hybrid | 0.3308 | 0.3471 |
| **hybrid+rerank** | 0.4286 | **0.4393** |

pgvector is marginally ahead; the two paths are equivalent for practical purposes.

**The keyword row is the control.** Keyword search is `minsearch` and never touches the vector
store, so it *must* be unchanged — and it is, to four decimals. If the harness had drifted, that
row would have moved. (It's how an earlier faulty run was caught: keyword "improved" when only
the vector backend had changed, which is impossible.)

### Query rewriting

250-pair subsample ([retrieval_rewrite.md](../eval/results/retrieval_rewrite.md)):

| Approach | HR@1 | HR@5 | MRR |
|---|---|---|---|
| hybrid+rerank | 0.336 | 0.556 | **0.434** |
| hybrid+rerank+rewrite | 0.276 | 0.484 | 0.360 |

Rewriting **costs 0.07 MRR**. The reason is visible in the rewrites themselves: *"is coffee bad
for me at night?"* becomes *"effects of nocturnal caffeine consumption on health"* — fluent, but
**more generic**. It discards the specifics that pin one passage. Evaluation questions are
already well-formed, which is exactly the case where rewriting has nothing to add and something
to lose. It ships implemented but **disabled** (`rewrite=True` to opt in); it would likely earn
its keep on vague or multi-turn conversational input, which this eval set doesn't contain.

## Generation (LLM-as-judge)

60 questions, `gpt-4o-mini` judge, on the winning retriever, 0 errors:

| Approach | n | errors | Mean score | % RELEVANT | RELEVANT | PARTLY | NON |
|---|---|---|---|---|---|---|---|
| **basic_rag** | 60 | 0 | **0.867** | **80.0%** | 48 | 8 | 4 |
| agentic_rag | 60 | 0 | 0.800 | 70.0% | 42 | 12 | 6 |

**Basic RAG wins outright** — it is both better and several times cheaper, since agentic makes
multiple API calls per answer. A plausible reading: once retrieval is strong enough that a
single well-targeted query surfaces the right passage, the agentic loop's reformulations mostly
add marginal chunks that dilute the context. Treat the margin as suggestive rather than decisive
(6 labels out of 60); what it firmly establishes is that agentic is *not* better, which settles
the default given the cost difference.

## Decisions applied

- **hybrid + cross-encoder re-rank** is the default retrieval path
  ([retrieve.py](../rag/retrieve.py), [rag.py](../rag/rag.py), [cli.py](../rag/cli.py)).
- **Query rewriting off** by default — measured as harmful here.
- **Basic RAG** is the default generator; `--agentic` remains available.
- `--retriever keyword|vector|hybrid` and `--no-rerank` reproduce any row above.

## Reproduce

```bash
uv run eval/generate_ground_truth.py --sample-size 150   # -> eval/ground_truth.jsonl (committed)
uv run eval/evaluate_retrieval.py                        # keyword/vector/hybrid/hybrid+rerank
uv run eval/evaluate_retrieval.py \
  --approaches hybrid+rerank,hybrid+rerank+rewrite --limit 250 --out retrieval_rewrite
uv run eval/evaluate_llm.py --sample-size 60             # LLM-as-judge, basic vs agentic
uv run eval/test_metrics.py                              # metric unit tests (no pytest needed)

# Against the containerized pgvector index instead of the local one:
PGVECTOR_URL=postgresql://rag:ragpass@127.0.0.1:5432/rag \
  uv run eval/evaluate_retrieval.py --out retrieval_pgvector
```

Cost: ~$1 on `gpt-4o-mini` (ground truth + judging + the rewrite arm). Retrieval, embeddings and
re-ranking all run locally and free. Ground truth and result files are committed so reviewers can
inspect the numbers without re-running.

> Absolute numbers are only comparable within a single corpus + ground-truth generation. Both
> were regenerated when the transcript set was completed, so figures here supersede any earlier
> ones — do not compare across regenerations, and don't copy figures into other files.
