# Retrieval evaluation

Ground-truth pairs: **750** · embedding model: `multi-qa-MiniLM-L6-cos-v1`

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.172 | 0.3173 | 0.384 | 0.4653 | 0.2608 |
| vector | 0.184 | 0.316 | 0.3747 | 0.4413 | 0.2626 |
| hybrid | 0.2227 | 0.3893 | 0.48 | 0.576 | 0.3292 |
| hybrid+rerank | 0.316 | 0.496 | 0.556 | 0.628 | 0.4199 |

**Winner by MRR: `hybrid+rerank`**
