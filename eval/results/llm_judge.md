# LLM evaluation — basic RAG vs agentic RAG (LLM-as-judge)

Sample: **60** questions · retriever: `vector` · judge scores RELEVANT=1.0 / PARTLY=0.5 / NON=0.0

| Approach | n | Mean score | % RELEVANT | RELEVANT | PARTLY | NON |
|---|---|---|---|---|---|---|
| basic_rag | 60 | 0.7833 | 73.3% | 44 | 6 | 10 |
| agentic_rag | 60 | 0.8 | 73.3% | 44 | 8 | 8 |

**Winner by mean score: `agentic_rag`**
