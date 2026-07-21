# Retrieval evaluation

Ground-truth pairs: **750** · embedding model: `multi-qa-MiniLM-L6-cos-v1`

## Overall

| Approach | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.308 | 0.444 | 0.4987 | 0.6013 | 0.3936 |
| vector | 0.356 | 0.4693 | 0.5293 | 0.5987 | 0.4284 |
| hybrid | 0.3867 | 0.5707 | 0.64 | 0.7307 | 0.4948 |
| hybrid+rerank | 0.5227 | 0.6773 | 0.7413 | 0.7867 | 0.6135 |

**Winner by MRR: `hybrid+rerank`**

## Chapter-size buckets

Split by whether the ground-truth chapter actually needed sub-chunking. The `long`
bucket is the population Module 6 targeted, so that's where the gain should show.

| Approach | Bucket | n | HR@5 | MRR |
|---|---|---|---|---|
| keyword | short (1 sub-chunk) | 20 | 0.6 | 0.5417 |
| keyword | long (2+ sub-chunks) | 730 | 0.4959 | 0.3896 |
| vector | short (1 sub-chunk) | 20 | 0.15 | 0.1187 |
| vector | long (2+ sub-chunks) | 730 | 0.5397 | 0.4369 |
| hybrid | short (1 sub-chunk) | 20 | 0.55 | 0.3834 |
| hybrid | long (2+ sub-chunks) | 730 | 0.6425 | 0.4979 |
| hybrid+rerank | short (1 sub-chunk) | 20 | 0.65 | 0.625 |
| hybrid+rerank | long (2+ sub-chunks) | 730 | 0.7438 | 0.6132 |
