# Concepts explained simply

A plain-language guide to the LLM / RAG ideas this project uses. Each term has a simple
definition, an analogy, and **how this project uses it** so it stays concrete. No prior
course knowledge assumed.

The whole system in one sentence:

> When you ask a question, the app **finds** the most relevant passages from ~35,000 podcast
> transcript chunks, **hands them to an LLM**, and the LLM writes an answer using only those
> passages — citing where each fact came from.

That pattern is **RAG**. Everything below is a piece of it.

---

## The big idea

### RAG (Retrieval-Augmented Generation)
Instead of asking an LLM to answer from memory (where it can make things up), you first
**retrieve** relevant text from a trusted source, then ask the LLM to answer **using that text**.

- *Analogy:* open-book exam. The model doesn't recite from memory — it looks up the
  passage first, then answers.
- *In this project:* the "book" is podcast transcripts. The LLM never answers from its own
  knowledge; if the transcripts don't cover it, the app says "I don't know."

### Knowledge base (corpus)
The collection of documents the system can search — here, `data/documents.jsonl`.

- *In this project:* 35,035 text chunks from 337 Huberman Lab and Andy Galpin episodes.

### Grounding & citations
Forcing the answer to be based on ("grounded in") the retrieved text, and showing which
source each claim came from.

- *In this project:* every answer links to the exact episode **and timestamp**, so you can
  verify it or jump to that moment.

---

## Building the knowledge base

### Chunking
Splitting long documents into small pieces so search can return a *specific passage* rather
than a whole 2-hour episode.

- *Analogy:* index cards instead of whole books. You want the paragraph that answers the
  question, not the entire volume.
- *In this project:* each episode is split by chapter, and long chapters are further split
  into **~350-token overlapping windows** ("sub-chunks"). Overlap means a sentence split across
  a boundary still appears whole in one chunk. Too-big chunks were measurably worse — one piece
  of text can't pinpoint a specific moment in a 15-minute chapter.

### Tokens
The units an LLM reads text in — roughly ¾ of a word each. Models have a maximum number of
tokens they can take at once.

- *In this project:* the embedding model only reads the first **512 tokens** of any text, which
  is *why* chunks are kept to ~350 tokens — so nothing important gets cut off.

### Embeddings (embedding model)
A way to turn a piece of text into a list of numbers (a **vector**) that captures its *meaning*.
Texts with similar meaning get similar numbers.

- *Analogy:* a GPS coordinate for meaning. "How do I sleep better?" and "tips for falling
  asleep" land close together, even with no words in common.
- *In this project:* a small local model (`multi-qa-MiniLM-L6`) turns each chunk into 384
  numbers. It runs on your machine, free, no API needed.

---

## Finding the relevant text (retrieval)

This is the "R" in RAG. The project has three search strategies and combines them.

### Keyword search (TF-IDF / lexical search)
Matches the actual **words** in your question against the documents, ranked by how important
those words are.

- *Analogy:* a smart `Ctrl+F` — finds exact word matches and ranks them.
- *Weakness:* misses synonyms. Searching "sleep" won't find a passage that only says "slumber."
- *In this project:* the `minsearch` index.

### Vector search (semantic search)
Turns your question into a vector (see *embeddings*) and finds the chunks whose vectors are
**closest** — i.e. closest in *meaning*, not wording.

- *Analogy:* finding the nearest GPS points to where your question lands on the "map of meaning."
- *Strength:* catches synonyms and paraphrases keyword search misses.
- *In this project:* Postgres **pgvector** in Docker, or file-based `sqlitesearch` locally —
  identical results, just different storage.

### Cosine similarity
The specific measure of "how close" two vectors are. Closer = more similar in meaning.

### ANN / HNSW index
Comparing your question to all 35,000 vectors one-by-one would be slow. An **Approximate
Nearest Neighbour** index (the project uses **HNSW**) is a clever shortcut that finds the
closest ones almost instantly, trading a tiny bit of accuracy for a huge speed-up.

- *Analogy:* a library's catalogue system — you don't scan every shelf, you jump near the
  right one.

### Hybrid search
Running **both** keyword and vector search and combining their results — because they fail in
different ways, together they're stronger than either alone.

- *In this project:* the default. It measurably beat either method by itself.

### RRF (Reciprocal Rank Fusion)
The simple rule used to *merge* the two ranked lists into one. Each result scores points based
on **how high it ranks** in each list (1/rank), and the points are added up.

- *Why not just add the raw scores?* Keyword scores and vector similarities are on totally
  different scales — adding them is apples and oranges. Using **rank position** instead sidesteps
  that. A chunk that both methods rank highly rises to the top.

### Re-ranking (cross-encoder)
A slow-but-accurate **second pass**: take the ~20 candidates the fast search found, and score
each one by reading the question and the passage **together**, then reorder.

- *Analogy:* résumé screening. You keyword-filter 1,000 applicants down to 20 (fast, rough),
  then actually interview those 20 (slow, accurate). You can't interview all 1,000.
- **Bi-encoder vs cross-encoder** — the key distinction:
  - The search step uses a **bi-encoder**: it turns the question and each passage into vectors
    *separately, in advance*, then compares them. Fast, because passages are pre-computed, but
    it never sees the two together.
  - The re-ranker is a **cross-encoder**: it reads the question and one passage *at the same
    time* and judges the match directly. Far more accurate, but too slow to run over 35,000
    chunks — so it only re-scores the top candidates.
- *In this project:* this was the single **biggest** accuracy improvement (it roughly doubled
  top-1 accuracy vs plain search). It only works because chunks are small enough to fit the
  cross-encoder's input. It can only reorder what search already found — so it can't rescue a
  passage the first stage missed, which is why the two steps are complementary.

### Query rewriting
Using an LLM to reword your question into a "better" search query before searching.

- *In this project:* implemented, **evaluated, and measured to make things worse**, so it ships
  **off by default**. Rewrites like "is coffee bad at night?" → "effects of nocturnal caffeine
  consumption" are fluent but *more generic*, which discards the specifics that pinpoint one
  passage. (Kept as an option — it would likely help on vague or conversational questions.)
  A good example of letting measurement, not intuition, decide.

---

## Generating the answer

### Prompt / context stuffing
The retrieved chunks are pasted into the LLM's instructions ("here is the context; answer using
only this"), and the LLM writes the reply.

- *In this project:* the system prompt forbids outside knowledge and requires citations.

### Basic RAG vs Agentic RAG (function calling / tool use)
- **Basic RAG:** retrieve once → answer. One search, one LLM call.
- **Agentic RAG:** the LLM is given search as a **tool** and decides *itself* when and what to
  search, possibly several times, reformulating as it goes ("function calling").
- *In this project:* both are built and compared. **Basic won** — it was both better *and*
  cheaper. Once retrieval is strong, the agentic loop's extra searches mostly added noise.
  Agentic stays available via a flag.

---

## Measuring quality (evaluation)

You can't improve what you don't measure. Every design choice here was picked by evaluation,
not guesswork.

### Ground truth
A labelled test set of `question → correct chunk` pairs, used to check retrieval automatically.

- *In this project:* 750 pairs. An LLM read 150 chunks and wrote realistic questions each chunk
  answers; that chunk is the "correct answer" for those questions. Questions are phrased *not*
  to copy the chunk's wording, so keyword search can't cheat.

### Hit Rate — HR@k (a.k.a. Recall@k)
Of all test questions, what fraction had the correct chunk **somewhere in the top k results**?
Pure pass/fail per question.

- *Example:* HR@5 = 0.55 means for 55% of questions, the right chunk was in the top 5.
- *Reading it:* HR@1 is the strictest (right answer must be #1); HR@10 is the most lenient.

### MRR (Mean Reciprocal Rank)
Like hit rate, but it **rewards ranking the right answer higher**. Each question scores
`1 / (position of the correct chunk)`; those are averaged.

- *Example:* correct chunk at rank 1 → 1.0; rank 2 → 0.5; rank 4 → 0.25; not found → 0.
- *Reading it:* MRR of 0.43 means, loosely, the right answer tends to sit around position 2–3.
  Higher is better; 1.0 would mean "always ranked first."
- *Why both HR and MRR?* HR asks "did we find it at all?", MRR asks "did we rank it near the
  top?" A good retriever needs both.

> **Why the project's numbers look modest (~0.43 MRR):** the test is deliberately strict — it
> demands the one *exact* ~350-token passage a question was written from. Retrieving a
> neighbouring passage that also answers the question counts as a **miss**. It measures whether
> we find *the* passage, not merely a good one. The number that reflects the actual user
> experience is the LLM-judge score below (80% of answers rated relevant).

### LLM-as-a-judge
Using a separate LLM call to **grade** the final answers — reading the question, the answer, and
the reference passage, and labelling the answer RELEVANT / PARTLY RELEVANT / NON-RELEVANT.

- *Why:* answer *quality* is subjective and can't be checked by exact string match. An LLM judge
  approximates a human rater, cheaply and at scale.
- *In this project:* how basic vs agentic generation was compared (basic scored 80% relevant).

---

## How it all fits together

```
Your question
   │
   ├── keyword search ─┐
   │                   ├── RRF fusion ── cross-encoder re-rank ── top passages
   └── vector search ──┘
                                                  │
                          passages pasted into the LLM prompt
                                                  │
                        grounded answer + episode/timestamp citations
```

Retrieval quality is checked with **HR / MRR** against the 750-pair ground truth; answer quality
is checked with the **LLM-as-a-judge**. See [evaluation.md](evaluation.md) for the actual numbers
and [pipeline.md](pipeline.md) for the end-to-end data flow.
