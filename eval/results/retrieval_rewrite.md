# Retrieval evaluation

Ground-truth pairs: **250** · embedding model: `multi-qa-MiniLM-L6-cos-v1`

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| hybrid+rerank | 0.336 | 0.508 | 0.556 | 0.616 | 0.4341 |
| hybrid+rerank+rewrite | 0.276 | 0.424 | 0.484 | 0.528 | 0.3598 |

**Winner by MRR: `hybrid+rerank`**
