# Evaluation (Module 4)

Two things are measured and fed back into the system's defaults: **which retriever**
(keyword vs vector) and **which answer generator** (basic vs agentic) works best.

## Summary

| Axis | Approaches compared | Winner | Now the default? |
|---|---|---|---|
| Retrieval | keyword (`minsearch`) vs vector (`sqlitesearch`) | **vector** (MRR 0.413 vs 0.377) | ✅ yes — default retriever |
| Generation | basic `rag()` vs `agentic_rag()` | agentic, but a statistical tie (0.800 vs 0.783) | ❌ no — basic stays default (see below) |

**Headline finding:** oversized chapter-sized chunks depress *both* retrievers — every
question about a `≤512`-token chunk scores far higher than one about a long chunk, even for
keyword search, which isn't truncated. So the highest-leverage next improvement is
**sub-chunking** (deferred to Module 6), not swapping the embedding model.

## Methodology

- **Ground truth** ([eval/ground_truth.jsonl](../eval/ground_truth.jsonl)) — 750 `question → chunk_id`
  pairs. For 150 chunks (stratified by source), the LLM wrote ~5 realistic user questions each,
  phrased *not* to quote the transcript so keyword search can't win on lexical overlap. The full
  chunk text is shown to the generator so questions can target the *tail* of long chapters — the
  part the vector index truncates. Script: [generate_ground_truth.py](../eval/generate_ground_truth.py).
- **Retrieval** — for each pair, retrieve top-10 with each backend; a "hit" means the source chunk
  appears in the results. Report **hit-rate@k** and **MRR** ([metrics.py](../eval/metrics.py),
  unit-tested). Script: [evaluate_retrieval.py](../eval/evaluate_retrieval.py).
- **Generation** — LLM-as-judge: `gpt-4o-mini` grades each answer against the reference chunk as
  RELEVANT (1.0) / PARTLY_RELEVANT (0.5) / NON_RELEVANT (0.0), on a 60-question sample.
  Script: [evaluate_llm.py](../eval/evaluate_llm.py).

## Retrieval results

| Method | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.264 | 0.441 | 0.532 | **0.635** | 0.377 |
| **vector** | **0.316** | **0.485** | **0.541** | 0.620 | **0.413** |

Vector wins where it matters most — top-1/top-3 and MRR — meaning the right chunk lands higher
in the list. Keyword only overtakes at HR@10 (deep in the list). **Vector is used as the default.**

### Truncation / chunk-size buckets

Vector embeds only the first 512 tokens of a chunk. If truncation were the main problem, vector
should collapse on the `>512` bucket while keyword holds. Instead **both** drop sharply:

| Method | Bucket | n | HR@5 | MRR |
|---|---|---|---|---|
| keyword | ≤512 tok | 45 | 0.689 | 0.559 |
| keyword | >512 tok | 705 | 0.522 | 0.365 |
| vector | ≤512 tok | 45 | 0.711 | 0.567 |
| vector | >512 tok | 705 | 0.531 | 0.403 |

Keyword indexes the full text yet still loses ~0.19 MRR on long chunks, so the dominant issue is
that one coarse representation (one BM25 doc / one 384-d vector) can't pinpoint a specific moment
in a 10–20-minute chapter. Vector keeps its lead throughout. → **Sub-chunking helps both backends;
it's the Module 6 priority.** (Only 45/750 pairs fall in the ≤512 bucket, matching the ~93% of the
corpus that exceeds the limit — see the known-limitation note in [CLAUDE.md](../CLAUDE.md#chunking-strategy).)

## Generation results (LLM-as-judge)

| Approach | n | Mean score | % RELEVANT | RELEVANT | PARTLY | NON |
|---|---|---|---|---|---|---|
| basic_rag | 60 | 0.783 | 73.3% | 44 | 6 | 10 |
| agentic_rag | 60 | **0.800** | 73.3% | 44 | 8 | 8 |

Both label the same 44/60 answers RELEVANT. Agentic's edge is turning **2 complete misses into
partial answers** — a 0.017 mean-score difference, well within noise for n=60. Since agentic makes
several API calls per answer (query reformulation + tool loop) for no reliable quality gain,
**basic RAG stays the default**; agentic remains available via `--agentic` for hard queries.

## Decisions applied

- **Vector** is the default retriever ([retrieve.py](../rag/retrieve.py), [rag.py](../rag/rag.py),
  [cli.py](../rag/cli.py)).
- **Basic RAG** stays the default generator (cost/latency); `--agentic` opt-in.
- Sub-chunking is the measured, prioritized fix for Module 6 (best practices).

## Reproduce

```bash
uv run eval/generate_ground_truth.py --sample-size 150   # -> eval/ground_truth.jsonl (committed)
uv run eval/evaluate_retrieval.py                        # -> eval/results/retrieval.{md,json}
uv run eval/evaluate_llm.py                              # -> eval/results/llm_judge.{md,json}
uv run eval/test_metrics.py                              # metric unit tests (no pytest needed)
```

Cost: ~$1 on `gpt-4o-mini` (ground-truth generation + judging). Query embeddings are local/free.
The ground-truth set and result files are committed so reviewers can inspect the numbers without
re-running.
