# Health & Performance RAG Assistant

> Ask science-backed questions about health, performance, and nutrition — answers grounded in Huberman Lab and Andy Galpin's *Perform* podcast transcripts, with citations you can jump to.

## What is this?

This is a Retrieval-Augmented Generation (RAG) application that lets you query evidence-based health and performance content using natural language. Instead of guessing what science says about sleep, recovery, or nutrition, you get answers grounded in actual transcripts from two leading sources — Andrew Huberman's neuroscience-focused podcast and Andy Galpin's performance science podcast — each answer citing the exact episode and timestamp it came from, so you can verify it or listen to the full context.

Built as the final project for [DataTalks.Club's LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

## Why this project?

Most health advice online is either oversimplified clickbait or buried in hours of podcast audio you don't have time to listen to. This project makes that knowledge searchable and queryable — ask a specific question, get a grounded answer, and jump straight to the moment in the episode where it's discussed.

## Data sources

- **Huberman Lab** podcast transcripts — neuroscience, sleep, hormones, nutrition science
- **Andy Galpin's *Perform*** podcast transcripts — exercise physiology, recovery, performance

> Recipe/nutrition data (RecipeNLG, USDA FoodData Central) was scoped out — the knowledge base
> is podcast transcripts only. Pairing answers with matching recipes remains possible future work.

## Tech stack

- **Orchestration:** Apache Airflow
- **Retrieval:** Hybrid search — keyword (`minsearch`) + vector (Postgres/**pgvector** in Docker, `sqlitesearch` locally) fused with RRF, then cross-encoder re-ranking
- **Containerization:** docker-compose (app + Postgres/pgvector)
- **Evaluation:** Hit-rate/MRR against an LLM-generated ground-truth set + LLM-as-a-judge
- **Interface:** Streamlit web app (chat + citations) and a CLI
- **Monitoring:** User feedback logged to SQLite + a Streamlit dashboard (7 charts)

## Quickstart

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`. The knowledge base
(`data/documents.jsonl`, 35,035 chunks) is **committed to the repo**, so there's no transcript
re-fetch either way.

### Option A — Docker (everything, no local setup)

```bash
docker compose up --build
# then open http://localhost:8501
```

Brings up Postgres + pgvector and the Streamlit app. The **first** start embeds all 35,035
chunks into the database — CPU-bound, so allow ~15–20 minutes; progress is logged. The database
lives in a named volume, so every later `docker compose up` starts in seconds. The build is
resumable: if it's interrupted, the next start continues from where it stopped.

### Option B — Local, no containers

Requires [`uv`](https://docs.astral.sh/uv/). Uses an on-disk `sqlitesearch` index instead of
Postgres — no infrastructure at all.

```bash
uv sync                            # install pinned dependencies (uv.lock)
uv run rag/build_vector_index.py   # embed the KB -> data/vector_index.db (~3 min)

uv run streamlit run app/app.py    # web app — chat + feedback + dashboard
uv run rag/cli.py "how do I fall asleep faster?"   # or the CLI
```

Both paths run identical retrieval code — `PGVECTOR_URL` (set by compose) is what selects the
Postgres backend. Retrieval quality is equivalent; see [docs/evaluation.md](docs/evaluation.md)
for the head-to-head.

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
2. **Vector** — HNSW index over local `multi-qa-MiniLM-L6-cos-v1` embeddings (384-dim,
   free/offline); pgvector in Docker, `sqlitesearch` locally.
3. **Hybrid** — the two fused with Reciprocal Rank Fusion.
4. **Re-ranking** — a local cross-encoder reorders the fused candidates.

**Hybrid + re-ranking is the default**, and it roughly doubles top-1 accuracy versus the
keyword baseline (HR@1 0.17 → 0.33). Query rewriting is implemented but **off by default** —
it measurably hurt (see [docs/evaluation.md](docs/evaluation.md)).

```bash
uv run rag/cli.py "how do I fall asleep faster?"                      # hybrid + rerank (default)
uv run rag/cli.py "how do I fall asleep faster?" --retriever keyword  # keyword baseline
uv run rag/cli.py "how do I fall asleep faster?" --no-rerank          # skip the cross-encoder
uv run rag/cli.py "compare Huberman and Galpin on caffeine timing" --agentic
uv run rag/compare_retrieval.py                                       # backends side by side (no LLM)
```

## Web app & monitoring

```bash
uv run streamlit run app/app.py
```

- **Chat page** — ask a question, get a grounded answer with **clickable citations** that jump
  to the exact timestamp in the episode. The sidebar exposes every retrieval approach
  (keyword / vector / hybrid, re-ranking, agentic mode); defaults are the measured winners.
- **Feedback** — 👍 / 👎 on each answer. Every interaction (question, answer, sources, config,
  latency, vote) is logged to `data/feedback.db`.
- **📊 Dashboard page** — 7 charts over that log: questions per day, feedback breakdown,
  positive rate over time, answer latency, most-cited episodes, retrieval config mix, and
  citations by podcast source — plus headline metrics.

> The dashboard reflects **real usage only** — no synthetic data is seeded. It starts empty;
> ask a few questions and vote to populate it.

## Evaluation

Both retrieval and answer generation are evaluated against a 750-pair, LLM-generated
ground-truth set, and the winners are wired in as defaults. Full report + reproduce steps:
[docs/evaluation.md](docs/evaluation.md).

- **Retrieval** (hit-rate / MRR) — four approaches over the full 35,035-chunk corpus;
  **hybrid + cross-encoder re-ranking wins** and is the default:

  | Approach | HR@1 | HR@5 | MRR |
  |---|---|---|---|
  | keyword | 0.172 | 0.384 | 0.261 |
  | vector | 0.184 | 0.375 | 0.263 |
  | hybrid | 0.224 | 0.483 | 0.331 |
  | **hybrid + re-rank** | **0.331** | **0.555** | **0.429** |

  Ground truth targets one specific ~350-token passage and matching is exact, so retrieving a
  neighbouring passage from the same chapter counts as a miss — a deliberately strict bar.

- **Generation** (LLM-as-judge) — **basic RAG wins**, 0.867 vs 0.800 for agentic (80% vs 70%
  answers rated relevant). Basic is the default; `--agentic` remains available.
- **Query rewriting** — implemented and evaluated, but it *lowers* accuracy, so it ships
  disabled: the eval questions are already specific and rewriting makes them generic.
- **Vector store** — pgvector (Docker) and sqlitesearch (local) score within noise of each
  other, so both paths are equivalent in quality.

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
| Interface | ✅ | Streamlit web app ([`app/app.py`](app/app.py)) + CLI ([`rag/cli.py`](rag/cli.py)) |
| Monitoring | ✅ | 👍/👎 feedback + 7-chart dashboard ([`app/pages/1_Dashboard.py`](app/pages/1_Dashboard.py)) |
| Containerization | ✅ | [`docker-compose.yml`](docker-compose.yml) — app + Postgres/pgvector, one command |
| Hybrid search (bonus) | ✅ | RRF in [`rag/retrieve.py`](rag/retrieve.py) — the default |
| Document re-ranking (bonus) | ✅ | cross-encoder [`rag/rerank.py`](rag/rerank.py) — biggest single gain |
| User query rewriting (bonus) | ✅ | [`rag/query_rewrite.py`](rag/query_rewrite.py) — evaluated, measured harmful, off by default |

## Status

🚧 Actively in development, built module-by-module alongside LLM Zoomcamp 2026.
