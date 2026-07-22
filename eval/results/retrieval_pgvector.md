# Retrieval evaluation

Ground-truth pairs: **750** · embedding model: `multi-qa-MiniLM-L6-cos-v1`

## Overall

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.172 | 0.3173 | 0.384 | 0.4653 | 0.2608 |
| vector | 0.2 | 0.3453 | 0.4133 | 0.4853 | 0.2873 |
| hybrid | 0.2413 | 0.416 | 0.488 | 0.5827 | 0.3471 |
| hybrid+rerank | 0.3387 | 0.52 | 0.568 | 0.628 | 0.4393 |

**Winner by MRR: `hybrid+rerank`**

## Chapter-size buckets

Split by whether the ground-truth chapter actually needed sub-chunking. The `long`
bucket is the population Module 6 targeted, so that's where the gain should show.

| Approach | Bucket | n | HR@5 | MRR |
|---|---|---|---|---|
| keyword | short (1 sub-chunk) | 750 | 0.384 | 0.2608 |
| keyword | long (2+ sub-chunks) | 0 | 0.0 | 0.0 |
| vector | short (1 sub-chunk) | 750 | 0.4133 | 0.2873 |
| vector | long (2+ sub-chunks) | 0 | 0.0 | 0.0 |
| hybrid | short (1 sub-chunk) | 750 | 0.488 | 0.3471 |
| hybrid | long (2+ sub-chunks) | 0 | 0.0 | 0.0 |
| hybrid+rerank | short (1 sub-chunk) | 750 | 0.568 | 0.4393 |
| hybrid+rerank | long (2+ sub-chunks) | 0 | 0.0 | 0.0 |
