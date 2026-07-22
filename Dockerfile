# Health & Performance RAG — Streamlit app + local embedding/re-ranking models.
#
# The embedding and cross-encoder models (~350MB) are baked in at build time so the first
# container start doesn't have to download them. The vector index is NOT baked: it's built
# into Postgres on first start (see entrypoint.sh), which keeps the image smaller and lets
# the index persist in a volume across restarts.
FROM python:3.13-slim

# uv for reproducible installs straight from uv.lock (same tool used locally).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Dependencies first, in their own layer, so code changes don't reinstall torch.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Bake the models into the image so the first start doesn't download ~350MB.
# Only the two modules that name the models are copied first: baking is slow, and putting it
# after the full source copy would re-download the models on every code change.
COPY rag/embeddings.py rag/rerank.py ./rag/
RUN python -c "\
import sys; sys.path.insert(0, 'rag'); \
from embeddings import get_model; \
from rerank import get_reranker; \
get_model(); get_reranker(); \
print('models cached to', __import__('os').environ['HF_HOME'])"

# Application code and the knowledge base (documents.jsonl is re-included in .dockerignore).
COPY schema.py ./
COPY rag/ ./rag/
COPY app/ ./app/
COPY ingestion/ ./ingestion/
COPY data/documents.jsonl ./data/documents.jsonl

# Mutable state (feedback.db) lives here and is the only volume-mounted path — keeping it
# out of /app/data so the shipped knowledge base in the image is never shadowed by a volume.
RUN mkdir -p /app/state

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 8501

# start-period is generous because the first start embeds the whole corpus on CPU, which
# can take tens of minutes. A shorter grace period marks the container "unhealthy" while
# it's doing exactly what it should.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45m --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["./entrypoint.sh"]
