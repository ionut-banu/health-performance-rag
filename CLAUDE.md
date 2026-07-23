# CLAUDE.md

Persistent context for Claude Code sessions in this repo. Read this before making changes.

## Project

RAG application answering health/performance/nutrition questions, grounded in podcast transcripts — final project for DataTalks.Club's LLM Zoomcamp. Built incrementally, module-by-module, alongside the course (not end-to-end upfront). Recipe data was **scoped out** (2026-07-22); the knowledge base is transcripts only.

## Dependency management

**`uv`** — not pip. Always use:
```bash
uv venv && uv sync        # setup
uv run <script>           # run any script
```
Dependencies are declared in `pyproject.toml`. Never create a `requirements.txt`.
Commit `uv.lock` — it pins exact transitive versions and counts toward the reproducibility rubric.

## Data sources

All three sources are wired into the pipeline. Channel config (URLs, filter defaults) lives in `ingestion/sources.yaml` — edit that file to change defaults, not the scripts.

| Source | Status | Notes |
|---|---|---|
| Huberman Lab (YouTube) | pipeline ready | `ingestion/sources.yaml` → `huberman` |
| Andy Galpin — Perform (YouTube) | pipeline ready | `ingestion/sources.yaml` → `galpin` |
| RecipeNLG + USDA FoodData Central | **scoped out** | Deliberately dropped — don't add without asking. The `"recipe"` values in `schema.py`'s Literals are harmless leftovers. |

## Ingestion pipeline — step order and status

Run in this order. Each step's output feeds the next.

```
1. list_all_episodes.py --source <source>   # catalog videos with metadata (resumable)
2. fetch_transcripts.py --source <source>   # pull transcripts from YouTube (resumable)
3. build_documents.py                       # chunk + normalize -> data/documents.jsonl
```

| Script | Status | Notes |
|---|---|---|
| `list_all_episodes.py` | ✅ validated on real data | 318 Huberman + 29 Galpin episodes cataloged, chapters confirmed populated |
| `fetch_transcripts.py` | ✅ complete | 308/318 Huberman + 29/29 Galpin. The 10 missing are recorded in `data/fetch_failures_huberman.json` as `NoTranscriptFound` — those videos genuinely have none. Retries/backoff handle YouTube rate limiting; see the module docstring. |
| `normalize.py` | ✅ validated on real data | strips filler by sliding a segment window (not sentence-split — captions have no punctuation); patterns live in `sources.yaml`, not the script |
| `chunk_by_chapters.py` | ✅ built | chapter-based chunking is primary; token-window (`tiktoken`) fallback for chapter-less episodes |
| `build_documents.py` | ✅ built + run end-to-end | produced **35,035** sub-chunks in `data/documents.jsonl` across 337 episodes |

## Next steps

All core rubric criteria are built (see the Build sequence below). What's left is optional:

1. **Screenshots** for the README — the rubric guidelines ask for them, and the monitoring
   dashboard needs real usage before it's worth capturing. Never seed synthetic feedback.
2. **Cloud deployment** (+2 bonus) — the only remaining point-earning work.

Orchestration (Airflow) was considered and **decided against** — see the Orchestration
section for why. It earns 0 rubric points, so it's not a next step.

**After any change to `data/documents.jsonl`, rebuild both indexes:**
`rm data/vector_index.db && uv run rag/build_vector_index.py` for local, and restart the
container for pgvector (the build is resumable and inserts only missing chunks). The keyword
`minsearch` index rebuilds in memory on each run, so it needs nothing. Then re-run
`eval/evaluate_retrieval.py` to confirm no regression.

## Chunking strategy

Chapters are confirmed populated on both Huberman Lab and Galpin episodes.
**Chapter-based chunking** (`ingestion/chunk_by_chapters.py`) is the primary strategy:

- Chapters are the semantic unit, but a chapter is **sub-chunked** into ~350-token
  (`MAX_CHUNK_TOKENS`, cl100k) overlapping windows — a whole chapter is far too coarse to
  retrieve well (see the note below). Each sub-chunk keeps its parent
  `chapter_title` and gets its own `start_timestamp` / `end_timestamp`.
- Every chunk carries **`parent_chunk_id`** = the id of the chapter it came from
  (`{source}_{video_id}_{chapter_index}`). It's provenance only — the evaluation matches chunk
  ids exactly and no longer depends on it. (It was once a compatibility shim so chapter-level
  ground truth survived sub-chunking; that ground truth has since been regenerated.)
- **Skip sponsor/outro chapters** by title pattern — patterns live in `sources.yaml` per
  source (`skip_chapter_patterns`), not hardcoded in the script.
- `chunk_episode` and the no-chapter fallback `chunk_by_tokens` share one windowing helper
  (`_window_segments`), so both paths chunk identically.
- Chunk size counts `tiktoken` cl100k while the embedder counts MiniLM wordpieces; the cap has
  deliberate headroom so ingestion never imports the retrieval model's tokenizer. If you change
  `MAX_CHUNK_TOKENS`, re-verify chunks still land under the embedder's 512-token window.
- Filler/sponsor-read text within kept chapters is stripped by `normalize.py`, matching
  against a sliding window of segments (not sentence-split — auto-captions carry no
  punctuation, so a naive `.!?` split treats a whole chapter as one sentence). Patterns
  also live in `sources.yaml` (`filler_text_patterns`).
- **Recipes**: atomic — one recipe = one chunk, never split.

> ✅ **Resolved in Module 6 — chapter chunks used to be far too large.**
> Chapter-sized chunks had a median of ~1,289 embedding tokens against the model's 512-token
> window, so ~93% were truncated. The Module 4 evaluation showed keyword search — which is
> *never* truncated — lost just as much accuracy on those chunks, which ruled out truncation as
> the root cause and identified **chunk size** instead: one representation can't localize a
> specific moment in a 15-minute chapter.
> Sub-chunking fixed it: every chunk now fits the window, which also unblocked cross-encoder
> re-ranking (chunks fit its input too). Current corpus: **35,035** chunks. Full numbers:
> `docs/evaluation.md`.
> **After any change to chunking:** re-run `build_documents.py`, then
> `rm data/vector_index.db && uv run rag/build_vector_index.py`, then `eval/evaluate_retrieval.py`
> to confirm you haven't regressed.

## Unified document schema

Every ingested doc must conform to this shape (defined in `schema.py`):

```python
{
  "id": str,                          # "huberman_<video_id>_<chapter_index>_<sub_index>"
  "source": "huberman" | "galpin" | "recipe",
  "content_type": "transcript" | "recipe",
  "title": str,
  "url": str,
  "text": str,
  "metadata": {
    # transcript chunks:
    "video_id": str,
    "start_timestamp": float,         # seconds into the episode
    "end_timestamp": float,           # seconds into the episode
    "upload_date": str,               # YYYY-MM-DD
    "chapter_title": str,             # chapter name, if chapter-based chunking used
    "parent_chunk_id": str,           # "<source>_<video_id>_<chapter_index>" — the pre-
                                      # sub-chunking id; keeps eval ground truth valid
    "chapter_index": int,             # position among surviving chapters
    "sub_index": int,                 # window position within the chapter
    # recipe chunks (scoped out — see Data sources):
    # "ingredients": list,
    # "macros": dict,
  }
}
```

## Key config files and known gotchas

- `ingestion/sources.yaml` — channel URLs and per-source filter defaults (min_duration, after_date).
  Always **quote date values** (e.g. `"2024-01-01"`) — unquoted ISO dates are parsed as
  `datetime.date` objects by YAML, not strings, causing a `TypeError` at runtime.

- `youtube-transcript-api` >= 1.2.x — static `get_transcript()` was removed in v1.2.0.
  Correct API: `YouTubeTranscriptApi().fetch(video_id).to_raw_data()`

## Orchestration — decided against (2026-07-23)

**No orchestrator.** The rubric awards full ingestion marks for "an automated Python script
*or* a special tool (e.g., Mage, dlt, Airflow, Prefect)" — the three resumable ingestion
scripts already earn 2/2, so Airflow would add **0 points**. It was ruled out because:

- The project's focus is RAG, not data engineering.
- Ingestion is a one-time build that's already done and committed (`data/documents.jsonl`);
  there's no recurring job for a scheduler to orchestrate.
- Airflow (scheduler + metadata DB + webserver) would bloat `docker-compose.yml` and work
  against the "clone and run" reproducibility story, for a pipeline reviewers are told not to run.

Don't add Airflow (or any orchestrator) without an explicit reason to revisit this.

## Build sequence (matches LLM Zoomcamp modules)

| Module | Status | What it covers in this repo |
|---|---|---|
| 1. Agentic RAG | ✅ built | `rag/` — `minsearch` keyword index + retrieve→prompt→LLM (`rag()`) and agentic function-calling (`agentic_rag()`), OpenAI `gpt-4o-mini` |
| 2. Vector Search | ✅ built | `rag/vector_search.py` — `sqlitesearch` HNSW index over local `multi-qa-MiniLM-L6-cos-v1` embeddings (`rag/embeddings.py`); `rag/retrieve.py` dispatches the backends via `--retriever` (Module 6 added `hybrid` + re-ranking on top); `rag/compare_retrieval.py` shows the side-by-side. PGVector landed in Module 7 as the containerized backend; sqlitesearch remains the infra-free local default. |
| 3. Orchestration | skipped (optional) | Ingestion already scores full marks as automated Python scripts, so an Airflow DAG earns **0 additional rubric points**. Deliberately not built — the project's focus is RAG, not data engineering, and the one-time ingestion is already done and committed. |
| 4. Evaluation | ✅ built | `eval/` — 750-pair LLM-generated ground truth; retrieval hit-rate/MRR (`evaluate_retrieval.py`) and LLM-as-judge (`evaluate_llm.py`). Ground truth is **sub-chunk level** with exact id matching — retrieving a neighbouring passage is a miss. **Don't hard-code result figures anywhere but `docs/evaluation.md`**; they change whenever the corpus or ground truth is regenerated. |
| 5. Monitoring | ✅ built | `app/feedback.py` logs every interaction (question, answer, sources, config, latency, vote) to `data/feedback.db`; `app/pages/1_Dashboard.py` charts it (7 charts). **Never seed synthetic rows** — the dashboard is evidence of real usage. |
| 6. Best Practices | ✅ built | Sub-chunking (`chunk_by_chapters.py`), hybrid RRF (`retrieve.py`), cross-encoder re-ranking (`rag/rerank.py`), query rewriting (`rag/query_rewrite.py`). **Default is hybrid+rerank**, the measured winner. Query rewriting measured as *harmful* → shipped but off by default. See `docs/evaluation.md`. |
| 7. End-to-end | ✅ built | Streamlit app (`app/app.py`) + `docker-compose.yml` (app + Postgres/pgvector, one command). **PGVector migration done** — `rag/pgvector_search.py`; `PGVECTOR_URL` selects it, otherwise `sqlitesearch` (so the eval harness and CLI still run infra-free). Migration verified against the same ground truth with keyword as a control (it must be identical, since it never touches the vector store). |

When asked to build a feature, check this table first — don't jump ahead to a module's
techniques before its row is in progress (e.g. no reranking code before Module 6).

## Conventions

- Python for all pipeline code.
- Keep ingestion, chunking, and retrieval as separate, testable modules — not one script.
- Any new env var goes in `.env.example` (no real secrets committed).
- Functions over classes for pipeline scripts — keep it simple until a class hierarchy is
  genuinely justified.

## Project evaluation rubric

Full guidelines: `docs/project-guidelines.md`. Build decisions should be checked
against this rubric — it defines what scores full points.

| Criterion | Max | What's needed for full marks |
|---|---|---|
| Problem description | 2 | Clear problem statement in README |
| Retrieval flow | 2 | Knowledge base + LLM both used |
| Retrieval evaluation | 2 | Multiple approaches evaluated — ✅ keyword vs vector vs hybrid vs hybrid+rerank (`docs/evaluation.md`) |
| LLM evaluation | 2 | Multiple approaches evaluated — ✅ basic vs agentic (LLM-as-judge) |
| Interface | 2 | UI (Streamlit) or API (FastAPI) — ✅ `app/app.py` |
| Ingestion pipeline | 2 | Automated ingestion — ✅ 3 resumable Python scripts (guidelines award full marks for a script *or* a tool like Airflow; a script suffices) |
| Monitoring | 2 | User feedback + dashboard (5+ charts) — ✅ 👍/👎 + 7 charts |
| Containerization | 2 | Full docker-compose — ✅ app + Postgres/pgvector |
| Reproducibility | 2 | Clear instructions, pinned deps, data accessible |
| Hybrid search | 1 | Vector + keyword combined (bonus) — ✅ RRF, and it's the default |
| Document re-ranking | 1 | Bonus — ✅ cross-encoder, biggest single gain (HR@1 +0.14) |
| User query rewriting | 1 | Bonus — ✅ implemented + evaluated (measured harmful → off by default) |
| Cloud deployment | 2 | Bonus |