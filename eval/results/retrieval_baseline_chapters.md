# Retrieval evaluation — keyword vs vector

Ground-truth pairs: **750** · embedding model: `multi-qa-MiniLM-L6-cos-v1`

## Overall

| Method | HR@1 | HR@3 | HR@5 | HR@10 | MRR |
|---|---|---|---|---|---|
| keyword | 0.264 | 0.4413 | 0.532 | 0.6347 | 0.3765 |
| vector | 0.316 | 0.4853 | 0.5413 | 0.62 | 0.4129 |

**Winner by MRR: `vector`**

## Truncation buckets (relevant chunk length in embedding tokens)

Vector embeds only the first 512 tokens of a chunk. If truncation hurts, vector's
score should drop sharply on the `>512` bucket while keyword (full-text) holds.

| Method | Bucket | n | HR@5 | MRR |
|---|---|---|---|---|
| keyword | <=512 | 45 | 0.6889 | 0.5593 |
| keyword | >512 | 705 | 0.522 | 0.3649 |
| vector | <=512 | 45 | 0.7111 | 0.5673 |
| vector | >512 | 705 | 0.5305 | 0.403 |
