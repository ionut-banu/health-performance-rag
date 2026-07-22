# Retrieval evaluation

Ground-truth pairs: **750** · embedding model: `multi-qa-MiniLM-L6-cos-v1`

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.172 | 0.3173 | 0.384 | 0.4653 | 0.2608 |
| vector | 0.184 | 0.316 | 0.3747 | 0.4413 | 0.2626 |
| hybrid | 0.224 | 0.396 | 0.4827 | 0.5747 | 0.3308 |
| hybrid+rerank | 0.3307 | 0.508 | 0.5547 | 0.612 | 0.4286 |

**Winner by MRR: `hybrid+rerank`**
