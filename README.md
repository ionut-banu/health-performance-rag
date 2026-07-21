# Health & Performance RAG Assistant

> Ask science-backed questions about health, performance, and nutrition — answers grounded in Huberman Lab, Andy Galpin's *Perform* podcast, and real recipe data.

## What is this?

This is a Retrieval-Augmented Generation (RAG) application that lets you query evidence-based health and performance content using natural language. Instead of guessing what science says about sleep, recovery, or nutrition, you get answers grounded in actual transcripts from two leading sources — Andrew Huberman's neuroscience-focused podcast and Andy Galpin's performance science podcast — paired with real recipe and nutrition data so recommendations are actionable, not just theoretical.

Built as the final project for [DataTalks.Club's LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

## Why this project?

Most health advice online is either oversimplified clickbait or buried in hours of podcast audio you don't have time to listen to. This project makes that knowledge searchable and queryable — ask a specific question, get an answer with sources, and (where relevant) a recipe that fits.

## Data sources

- **Huberman Lab** podcast transcripts — neuroscience, sleep, hormones, nutrition science
- **Andy Galpin's *Perform*** podcast transcripts — exercise physiology, recovery, performance
- **RecipeNLG** + **USDA FoodData Central** — recipes grounded in real nutritional data

## Tech stack

- **Orchestration:** Apache Airflow
- **Retrieval:** Hybrid search — keyword (`minsearch`) + vector (`sqlitesearch`, local `sentence-transformers` embeddings) fused with RRF, then cross-encoder re-ranking
- **Evaluation:** Hit-rate/MRR against an LLM-generated ground-truth set + LLM-as-a-judge
- **Interface:** *(TBD — Streamlit/FastAPI)*
- **Monitoring:** *(TBD)*

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and an OpenAI API key. Copy `.env.example` to
`.env` and fill in `OPENAI_API_KEY`.

The knowledge base (`data/documents.jsonl`, 27,085 chunks) is **committed to the repo**, so you
can query immediately — no transcript re-fetch needed.

```bash
uv sync                            # install pinned dependencies (uv.lock)
uv run rag/build_vector_index.py   # embed the KB -> data/vector_index.db (first run downloads a ~90MB model)
uv run rag/cli.py "how do I fall asleep faster?"
```

### Rebuild the knowledge base from scratch (optional)

Only needed to refresh transcripts or add sources. Run per source, in order — each step is
resumable and feeds the next; see [docs/pipeline.md](docs/pipeline.md) for what each does and how
they link.

```bash
uv run ingestion/list_all_episodes.py --source huberman   # catalog episodes  -> data/all_huberman_episodes.json
uv run ingestion/fetch_transcripts.py --source huberman   # pull transcripts  -> data/raw_transcripts/huberman/
uv run ingestion/build_documents.py                       # chunk + normalize -> data/documents.jsonl
uv run rag/build_vector_index.py                          # re-embed the rebuilt KB
```

## Retrieval

The knowledge base is chunked into ~350-token overlapping windows within each chapter, and
queried through a pipeline whose every stage was chosen by measurement:

1. **Keyword** — in-memory `minsearch` TF-IDF index.
2. **Vector** — on-disk `sqlitesearch` HNSW index over local `multi-qa-MiniLM-L6-cos-v1`
   embeddings (384-dim, free/offline).
3. **Hybrid** — the two fused with Reciprocal Rank Fusion.
4. **Re-ranking** — a local cross-encoder reorders the fused candidates.

**Hybrid + re-ranking is the default**, and it roughly doubles top-1 accuracy versus the
keyword baseline (HR@1 0.31 → 0.52). Query rewriting is implemented but **off by default** —
it measurably hurt (see [docs/evaluation.md](docs/evaluation.md)).

```bash
uv run rag/cli.py "how do I fall asleep faster?"                      # hybrid + rerank (default)
uv run rag/cli.py "how do I fall asleep faster?" --retriever keyword  # Module 1 baseline
uv run rag/cli.py "how do I fall asleep faster?" --no-rerank          # skip the cross-encoder
uv run rag/cli.py "compare Huberman and Galpin on caffeine timing" --agentic
uv run rag/compare_retrieval.py                                       # backends side by side (no LLM)
```

## Evaluation (Module 4)

Both retrieval and answer generation are evaluated against a 750-pair, LLM-generated
ground-truth set, and the winners are wired in as defaults. Full report + reproduce steps:
[docs/evaluation.md](docs/evaluation.md).

- **Retrieval** (hit-rate / MRR) — measured across four approaches; **hybrid + cross-encoder
  re-ranking wins** and is the default:

  | Approach | HR@1 | HR@5 | MRR |
  |---|---|---|---|
  | keyword | 0.308 | 0.499 | 0.394 |
  | vector | 0.356 | 0.529 | 0.428 |
  | hybrid | 0.387 | 0.640 | 0.495 |
  | **hybrid + re-rank** | **0.523** | **0.741** | **0.614** |

- **Generation** (LLM-as-judge) — basic RAG ≈ agentic (0.783 vs 0.800, a tie); basic stays
  default for cost/latency, `--agentic` available.
- **Query rewriting** — implemented and evaluated, but it *lowered* MRR (0.491 vs 0.602), so it
  ships disabled. The eval questions are already specific, and rewriting made them generic.
- **Progress** — retrieval MRR improved **+49%** (0.413 → 0.614) from Modules 4→6, measured on
  the same 750 pairs.

```bash
uv run eval/generate_ground_truth.py --sample-size 150   # -> eval/ground_truth.jsonl
uv run eval/evaluate_retrieval.py                        # -> eval/results/retrieval.md
uv run eval/evaluate_llm.py                              # -> eval/results/llm_judge.md
```

## Evaluation criteria map

Where each [project rubric](docs/project-guidelines.md#evaluation-criteria) criterion is
addressed (for peer reviewers):

| Criterion | Status | Where |
|---|---|---|
| Problem description | ✅ | [What is this?](#what-is-this) / [Why this project?](#why-this-project) |
| Retrieval flow (knowledge base + LLM) | ✅ | [`rag/`](rag/) — retrieve → prompt → LLM (`rag.py`) |
| Retrieval evaluation (multiple approaches) | ✅ | [docs/evaluation.md](docs/evaluation.md) — 4 approaches compared; best (hybrid + re-rank) is the default |
| LLM evaluation (multiple approaches) | ✅ | [docs/evaluation.md](docs/evaluation.md) — basic vs agentic, LLM-as-judge |
| Ingestion pipeline (automated) | ✅ | [`ingestion/`](ingestion/) Python scripts · [docs/pipeline.md](docs/pipeline.md) |
| Reproducibility (pinned deps, data accessible) | ✅ | KB committed (`data/documents.jsonl`) · `uv.lock` · [Quickstart](#quickstart) |
| Interface | 🚧 1/2 | CLI ([`rag/cli.py`](rag/cli.py)); UI/API planned (Module 7) |
| Monitoring | ⬜ | planned (Module 5) |
| Containerization | ⬜ | planned (Module 7 — docker-compose) |
| Hybrid search (bonus) | ✅ | RRF in [`rag/retrieve.py`](rag/retrieve.py) — the default |
| Document re-ranking (bonus) | ✅ | cross-encoder [`rag/rerank.py`](rag/rerank.py) — biggest single gain |
| User query rewriting (bonus) | ✅ | [`rag/query_rewrite.py`](rag/query_rewrite.py) — evaluated, measured harmful, off by default |

## Status

🚧 Actively in development, built module-by-module alongside LLM Zoomcamp 2026.
