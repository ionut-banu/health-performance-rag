# Retrieval evaluation

Ground-truth pairs: **250** · embedding model: `multi-qa-MiniLM-L6-cos-v1`

## Overall

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| hybrid+rerank | 0.5 | 0.672 | 0.744 | 0.788 | 0.6019 |
| hybrid+rerank+rewrite | 0.396 | 0.556 | 0.624 | 0.676 | 0.4912 |

**Winner by MRR: `hybrid+rerank`**

## Chapter-size buckets

Split by whether the ground-truth chapter actually needed sub-chunking. The `long`
bucket is the population Module 6 targeted, so that's where the gain should show.

| Approach | Bucket | n | HR@5 | MRR |
|---|---|---|---|---|
| hybrid+rerank | short (1 sub-chunk) | 5 | 0.6 | 0.6 |
| hybrid+rerank | long (2+ sub-chunks) | 245 | 0.7469 | 0.6019 |
| hybrid+rerank+rewrite | short (1 sub-chunk) | 5 | 0.4 | 0.4 |
| hybrid+rerank+rewrite | long (2+ sub-chunks) | 245 | 0.6286 | 0.4931 |
