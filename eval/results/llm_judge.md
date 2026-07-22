# LLM evaluation — basic RAG vs agentic RAG (LLM-as-judge)

Sample: **60** questions · retriever: `hybrid+rerank` · judge scores RELEVANT=1.0 / PARTLY=0.5 / NON=0.0

`n` counts successfully scored answers; `errors` are calls that failed and were excluded from the score rather than counted as bad answers.

| Approach | n | errors | Mean score | % RELEVANT | RELEVANT | PARTLY | NON |
|---|---|---|---|---|---|---|---|
| basic_rag | 60 | 0 | 0.8667 | 80.0% | 48 | 8 | 4 |
| agentic_rag | 60 | 0 | 0.8 | 70.0% | 42 | 12 | 6 |

**Winner by mean score: `basic_rag`**
