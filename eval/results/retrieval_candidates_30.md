# Retrieval evaluation

Ground-truth pairs: **100** · embedding model: `multi-qa-MiniLM-L6-cos-v1`

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.14 | 0.26 | 0.35 | 0.4 | 0.2129 |
| vector | 0.17 | 0.31 | 0.41 | 0.49 | 0.2578 |
| hybrid | 0.2 | 0.31 | 0.43 | 0.57 | 0.2935 |
| hybrid+rerank | 0.3 | 0.5 | 0.55 | 0.6 | 0.4084 |

**Winner by MRR: `hybrid+rerank`**
