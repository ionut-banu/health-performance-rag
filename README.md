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

## Retrieval (Modules 1–2)

Two retrieval backends run over the same `data/documents.jsonl` knowledge base, so
answers can be compared keyword-vs-semantic:

- **Keyword (Module 1):** in-memory `minsearch` TF-IDF index.
- **Vector (Module 2):** on-disk `sqlitesearch` HNSW index over local
  `multi-qa-MiniLM-L6-cos-v1` embeddings (384-dim, free/offline). **Default** — it wins the
  Module 4 retrieval eval.

```bash
uv sync

# Build the vector index once (embeds all chunks -> data/vector_index.db).
# First run downloads the embedding model (~90MB) to the Hugging Face cache.
uv run rag/build_vector_index.py

# Ask a question (needs OPENAI_API_KEY in .env; see .env.example). Vector is the default.
uv run rag/cli.py "how do I fall asleep faster?"
uv run rag/cli.py "how do I fall asleep faster?" --retriever keyword   # keyword baseline
uv run rag/cli.py "compare Huberman and Galpin on caffeine timing" --agentic

# Eyeball keyword vs vector retrieval side by side (no LLM calls).
uv run rag/compare_retrieval.py
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

## Status

🚧 Actively in development, built module-by-module alongside LLM Zoomcamp 2026.
