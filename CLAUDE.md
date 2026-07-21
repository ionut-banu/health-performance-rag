# CLAUDE.md

Persistent context for Claude Code sessions in this repo. Read this before making changes.

## Project

RAG application answering health/performance/nutrition questions, grounded in podcast transcripts and recipe data — final project for DataTalks.Club's LLM Zoomcamp. Built incrementally, module-by-module, alongside the course (not end-to-end upfront).

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
| RecipeNLG + USDA FoodData Central | not started | add after transcript pipeline is validated end-to-end |

## Ingestion pipeline — step order and status

Run in this order. Each step's output feeds the next.

```
1. list_all_episodes.py --source <source>   # catalog videos with metadata (resumable)
2. fetch_transcripts.py --source <source>   # pull transcripts from YouTube (resumable)
3. build_documents.py                       # chunk + normalize -> data/documents.jsonl
```

| Script | Status | Notes |
|---|---|---|
| `list_all_episodes.py` | ✅ validated on real data | 315 Huberman episodes, chapters confirmed populated |
| `fetch_transcripts.py` | ✅ fixed + resume support | 200/315 Huberman + 28/28 Galpin fetched so far |
| `normalize.py` | ✅ validated on real data | strips filler by sliding a segment window (not sentence-split — captions have no punctuation); patterns live in `sources.yaml`, not the script |
| `chunk_by_chapters.py` | ✅ built | chapter-based chunking is primary; token-window (`tiktoken`) fallback for chapter-less episodes |
| `build_documents.py` | ✅ built + run end-to-end | produced 27,085 sub-chunks in `data/documents.jsonl` from currently-fetched transcripts (was 5,269 before Module 6 sub-chunking) |

## Next steps (in order)

1. Run `fetch_transcripts.py --source huberman` to completion (200/315 fetched so far),
   then re-run `build_documents.py` to pick up the rest. After any change to
   `data/documents.jsonl`, rebuild the vector index too: `rm data/vector_index.db &&
   uv run rag/build_vector_index.py` (the keyword `minsearch` index rebuilds in-memory
   on each run, but the vector `.db` is persisted and won't refresh on its own).
2. Modules 1 (Agentic RAG) and 2 (Vector Search) are ✅ built. Next unbuilt module in the
   Build sequence below is Module 3 (Airflow orchestration) or Module 4 (Evaluation —
   now has two retrieval approaches, keyword + vector, to score against a ground-truth set).

## Chunking strategy

Chapters are confirmed populated on both Huberman Lab and Galpin episodes.
**Chapter-based chunking** (`ingestion/chunk_by_chapters.py`) is the primary strategy:

- Chapters are the semantic unit, but a chapter is **sub-chunked** into ~350-token
  (`MAX_CHUNK_TOKENS`, cl100k) overlapping windows — a whole chapter is far too coarse to
  retrieve well (see the Module 6 note below). Each sub-chunk keeps its parent
  `chapter_title` and gets its own `start_timestamp` / `end_timestamp`.
- Every chunk carries **`parent_chunk_id`** = the id its chapter would have had before
  sub-chunking (`{source}_{video_id}_{chapter_index}`). This is load-bearing: it's what keeps
  the Module 4 evaluation ground truth valid across the re-chunk. Don't break this scheme
  without regenerating `eval/ground_truth.jsonl`.
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
> Sub-chunking took the corpus from 5,269 → **27,085** chunks with **0% over the window**, and
> unblocked cross-encoder re-ranking (chunks now fit its input). Full numbers:
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
    # recipe chunks (not yet implemented):
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

## Orchestration

**Apache Airflow** — chosen over Kestra (which the course teaches). Since it's not covered in the course, document any non-trivial DAG setup in the README for reviewers.

- Don't introduce Airflow until the ingestion scripts work standalone in plain Python.
- Pin the Airflow version explicitly in `pyproject.toml` (current stable line: 3.2.x).
- Airflow is a Module 3 concern — don't add it earlier.

## Build sequence (matches LLM Zoomcamp modules)

| Module | Status | What it covers in this repo |
|---|---|---|
| 1. Agentic RAG | ✅ built | `rag/` — `minsearch` keyword index + retrieve→prompt→LLM (`rag()`) and agentic function-calling (`agentic_rag()`), OpenAI `gpt-4o-mini` |
| 2. Vector Search | ✅ built | `rag/vector_search.py` — `sqlitesearch` HNSW index over local `multi-qa-MiniLM-L6-cos-v1` embeddings (`rag/embeddings.py`); `rag/retrieve.py` dispatches the backends via `--retriever` (Module 6 added `hybrid` + re-ranking on top); `rag/compare_retrieval.py` shows the side-by-side. **PGVector deferred to Module 7** (kept infra-free to respect containerization gating). |
| 3. Orchestration | not started | Wrap ingestion in an Airflow DAG |
| 4. Evaluation | ✅ built | `eval/` — 750-pair LLM-generated ground truth; retrieval hit-rate/MRR (`evaluate_retrieval.py`) and LLM-as-judge (`evaluate_llm.py`). **Vector beat keyword** (MRR 0.413 vs 0.377) → made vector the default at the time, since superseded by Module 6's hybrid+rerank; basic≈agentic generation (tie) → basic is still the default generator. Report + reproduce steps in `docs/evaluation.md`. |
| 5. Monitoring | ✅ built | `app/feedback.py` logs every interaction (question, answer, sources, config, latency, vote) to `data/feedback.db`; `app/pages/1_Dashboard.py` charts it (7 charts). **Never seed synthetic rows** — the dashboard is evidence of real usage. |
| 6. Best Practices | ✅ built | Sub-chunking (`chunk_by_chapters.py`), hybrid RRF (`retrieve.py`), cross-encoder re-ranking (`rag/rerank.py`), query rewriting (`rag/query_rewrite.py`). **Default is now hybrid+rerank** — MRR 0.614 vs 0.413 (+49%). Query rewriting measured as *harmful* here → shipped but off. See `docs/evaluation.md`. |
| 7. End-to-end | partially built | **Interface ✅** — Streamlit app (`app/app.py`), sidebar exposes every retrieval approach with the eval winners as defaults. **Docker-compose still to do.** |

When asked to build a feature, check this table first — don't jump ahead to a module's
techniques before its row is in progress (e.g. no reranking code before Module 6).

## Conventions

- Python for all pipeline code.
- Keep ingestion, chunking, and retrieval as separate, testable modules — not one script.
- Any new env var goes in `.env.example` (no real secrets committed).
- Functions over classes for pipeline scripts — keep it simple until a class hierarchy is
  genuinely justified (revisit when wiring into Airflow DAGs at Module 3).

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
| Ingestion pipeline | 2 | Automated (Airflow DAG) |
| Monitoring | 2 | User feedback + dashboard (5+ charts) — ✅ 👍/👎 + 7 charts |
| Containerization | 2 | Full docker-compose |
| Reproducibility | 2 | Clear instructions, pinned deps, data accessible |
| Hybrid search | 1 | Vector + keyword combined (bonus) — ✅ RRF, and it's the default |
| Document re-ranking | 1 | Bonus — ✅ cross-encoder, biggest single gain (HR@1 +0.14) |
| User query rewriting | 1 | Bonus — ✅ implemented + evaluated (measured harmful → off by default) |
| Cloud deployment | 2 | Bonus |