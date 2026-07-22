# Retrieval evaluation

Ground-truth pairs: **750** · embedding model: `multi-qa-MiniLM-L6-cos-v1`

## Overall

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.308 | 0.444 | 0.4987 | 0.6013 | 0.3936 |
| vector | 0.412 | 0.536 | 0.5987 | 0.6787 | 0.4914 |
| hybrid | 0.408 | 0.588 | 0.6587 | 0.7427 | 0.5147 |
| hybrid+rerank | 0.536 | 0.6867 | 0.7493 | 0.7987 | 0.6257 |

**Winner by MRR: `hybrid+rerank`**

## Chapter-size buckets

Split by whether the ground-truth chapter actually needed sub-chunking. The `long`
bucket is the population Module 6 targeted, so that's where the gain should show.

| Approach | Bucket | n | HR@5 | MRR |
|---|---|---|---|---|
| keyword | short (1 sub-chunk) | 20 | 0.6 | 0.5417 |
| keyword | long (2+ sub-chunks) | 730 | 0.4959 | 0.3896 |
| vector | short (1 sub-chunk) | 20 | 0.2 | 0.1625 |
| vector | long (2+ sub-chunks) | 730 | 0.6096 | 0.5004 |
| hybrid | short (1 sub-chunk) | 20 | 0.55 | 0.3825 |
| hybrid | long (2+ sub-chunks) | 730 | 0.6616 | 0.5183 |
| hybrid+rerank | short (1 sub-chunk) | 20 | 0.65 | 0.625 |
| hybrid+rerank | long (2+ sub-chunks) | 730 | 0.7521 | 0.6257 |
