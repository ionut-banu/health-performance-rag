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
- **Retrieval:** Keyword search (`minsearch`) + semantic vector search (`sqlitesearch`, local `sentence-transformers` embeddings); hybrid + PGVector planned
- **Evaluation:** Retrieval metrics + LLM-as-a-judge
- **Interface:** *(TBD — Streamlit/FastAPI)*
- **Monitoring:** *(TBD)*

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/) and an OpenAI API key. Copy `.env.example` to
`.env` and fill in `OPENAI_API_KEY`.

The knowledge base (`data/documents.jsonl`, 5,269 chunks) is **committed to the repo**, so you
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

## Retrieval (Modules 1–2)

Two retrieval backends run over the same `data/documents.jsonl` knowledge base, so
answers can be compared keyword-vs-semantic:

- **Keyword (Module 1):** in-memory `minsearch` TF-IDF index.
- **Vector (Module 2):** on-disk `sqlitesearch` HNSW index over local
  `multi-qa-MiniLM-L6-cos-v1` embeddings (384-dim, free/offline). **Default** — it wins the
  Module 4 retrieval eval.

```bash
uv run rag/cli.py "how do I fall asleep faster?"                      # vector (default)
uv run rag/cli.py "how do I fall asleep faster?" --retriever keyword  # keyword baseline
uv run rag/cli.py "compare Huberman and Galpin on caffeine timing" --agentic
uv run rag/compare_retrieval.py                                       # keyword vs vector, side by side (no LLM)
```

## Evaluation (Module 4)

Both retrieval and answer generation are evaluated against a 750-pair, LLM-generated
ground-truth set, and the winners are wired in as defaults. Full report + reproduce steps:
[docs/evaluation.md](docs/evaluation.md).

- **Retrieval** (hit-rate / MRR) — **vector beats keyword** (MRR 0.413 vs 0.377), so vector is
  the default retriever.
- **Generation** (LLM-as-judge) — basic RAG ≈ agentic (0.783 vs 0.800, a tie); basic stays
  default for cost/latency, `--agentic` available.
- **Finding** — oversized chapter chunks depress *both* retrievers, pointing to sub-chunking as
  the top Module 6 improvement.

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
| Retrieval evaluation (multiple approaches) | ✅ | [docs/evaluation.md](docs/evaluation.md) — keyword vs vector, best (vector) is default |
| LLM evaluation (multiple approaches) | ✅ | [docs/evaluation.md](docs/evaluation.md) — basic vs agentic, LLM-as-judge |
| Ingestion pipeline (automated) | ✅ | [`ingestion/`](ingestion/) Python scripts · [docs/pipeline.md](docs/pipeline.md) |
| Reproducibility (pinned deps, data accessible) | ✅ | KB committed (`data/documents.jsonl`) · `uv.lock` · [Quickstart](#quickstart) |
| Interface | 🚧 1/2 | CLI ([`rag/cli.py`](rag/cli.py)); UI/API planned (Module 7) |
| Monitoring | ⬜ | planned (Module 5) |
| Containerization | ⬜ | planned (Module 7 — docker-compose) |
| Hybrid search · re-ranking · query rewriting (bonus) | ⬜ | planned (Module 6) |

## Status

🚧 Actively in development, built module-by-module alongside LLM Zoomcamp 2026.
