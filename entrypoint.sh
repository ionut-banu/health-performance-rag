#!/usr/bin/env bash
# Container entrypoint: make sure the vector index exists, then serve the app.
#
# The index isn't baked into the image, so the first start embeds all chunks into Postgres
# (a few minutes, logged below). build_pg_index() is idempotent, so every later start —
# including after `docker compose down` — finds the populated table and skips straight to
# serving, because the database lives in a named volume.
set -euo pipefail

echo "==> Ensuring pgvector index is populated (idempotent; first run takes a few minutes)…"
python rag/pgvector_search.py

echo "==> Starting Streamlit on :8501"
exec streamlit run app/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false
