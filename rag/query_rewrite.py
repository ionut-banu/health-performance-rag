"""
Module 6: LLM query rewriting.

Users ask conversationally ("is coffee bad for me at night?") while the corpus speaks
clinically ("caffeine's quarter-life and adenosine clearance before sleep"). This rewrites
the question into a retrieval-friendly query — domain vocabulary, no filler — before it
reaches the index.

Uses the same OpenAI model as answer generation (OPENAI_MODEL, default gpt-4o-mini).
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

_client = OpenAI()

REWRITE_PROMPT = """Rewrite the user's question as a search query for a database of health, \
fitness, and nutrition podcast transcripts.

- Keep the original meaning and all specifics; do not answer the question.
- Prefer the technical/physiological terms an expert would say out loud.
- Drop conversational filler ("can you tell me", "I was wondering").
- Return ONLY the rewritten query, no quotes or preamble.

Question: {query}"""


def rewrite_query(query: str) -> str:
    """Return a retrieval-optimized version of the query (falls back to the original)."""
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
            temperature=0.0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or query
    except Exception:
        # Retrieval must not hard-fail because the rewrite call did.
        return query
