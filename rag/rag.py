"""
Module 1: Agentic RAG over the podcast-transcript knowledge base.

Two entry points:
  - rag()          basic retrieve -> prompt -> LLM loop.
  - agentic_rag()  the LLM decides when/what to search via function calling.

Provider is OpenAI (see .env.example: OPENAI_API_KEY, OPENAI_MODEL).
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from retrieve import retrieve

load_dotenv()

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_AGENT_ITERATIONS = 5

SYSTEM_PROMPT = """You are a health, fitness, and nutrition assistant. Answer the \
user's question using ONLY the context provided from podcast transcripts (Huberman \
Lab and Andy Galpin's Perform). Follow these rules:

- Ground every claim in the context. Do not use outside knowledge.
- Cite the episode title and the timestamp link for the sources you use.
- If the context does not contain the answer, say "I don't know based on the \
available episodes." Do not guess.
- Be concise and practical."""

# Reuse a single client across calls (indexes are cached in retrieve.py).
_client = OpenAI()


def _timestamp_link(chunk: dict) -> str:
    ts = chunk.get("start_timestamp")
    if ts is None:
        return chunk["url"]
    return f"{chunk['url']}&t={int(ts)}s"


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a citable, timestamped context block."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        chapter = c.get("chapter_title") or ""
        header = f"[{i}] {c['title']}"
        if chapter:
            header += f" — {chapter}"
        blocks.append(f"{header}\nsource: {_timestamp_link(c)}\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def rag_with_sources(
    query: str,
    num_results: int = 5,
    source: str | None = None,
    method: str = "hybrid",  # Module 6 eval winner (see docs/evaluation.md)
    rerank: bool = True,
    rewrite: bool = False,
) -> tuple[str, list[dict]]:
    """
    Basic RAG, returning both the answer and the chunks it was grounded in.

    The UI renders those chunks as clickable citations and logs them for the monitoring
    dashboard; `rag()` below wraps this for callers that only want the answer text.
    """
    chunks = retrieve(
        query,
        num_results=num_results,
        source=source,
        method=method,
        rerank=rerank,
        rewrite=rewrite,
    )
    context = build_context(chunks)
    user_prompt = f"Question: {query}\n\nContext:\n{context}"

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content, chunks


def rag(
    query: str,
    num_results: int = 5,
    source: str | None = None,
    method: str = "hybrid",
    rerank: bool = True,
    rewrite: bool = False,
) -> str:
    """Basic RAG: retrieve, stuff context into the prompt, answer in one call."""
    answer, _ = rag_with_sources(
        query,
        num_results=num_results,
        source=source,
        method=method,
        rerank=rerank,
        rewrite=rewrite,
    )
    return answer


# --- Agentic RAG: the LLM calls `search` as a tool -------------------------

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the podcast-transcript knowledge base for relevant chunks. "
            "Call this one or more times (reformulating the query as needed) "
            "before answering. Returns chunks with title, timestamp link, and text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "source": {
                    "type": "string",
                    "enum": ["huberman", "galpin"],
                    "description": "Optional: restrict to a single podcast source.",
                },
            },
            "required": ["query"],
        },
    },
}


def _run_search_tool(
    args: dict, method: str = "hybrid", rerank: bool = True, rewrite: bool = False
) -> tuple[list[dict], list[dict]]:
    """Returns (payload for the model, raw chunks) — the raw chunks feed UI citations."""
    chunks = retrieve(
        args["query"],
        num_results=args.get("num_results", 5),
        source=args.get("source"),
        method=method,
        rerank=rerank,
        rewrite=rewrite,
    )
    # Send the model only the fields it needs to answer + cite.
    payload = [
        {
            "title": c["title"],
            "chapter_title": c.get("chapter_title") or "",
            "source_link": _timestamp_link(c),
            "text": c["text"],
        }
        for c in chunks
    ]
    return payload, chunks


def agentic_rag_with_sources(
    query: str,
    verbose: bool = False,
    method: str = "hybrid",
    rerank: bool = True,
    rewrite: bool = False,
) -> tuple[str, list[dict]]:
    """
    Agentic RAG returning the answer plus every chunk retrieved across all tool calls.

    The model may search several times with reformulated queries, so sources accumulate
    across iterations; duplicates (the same chunk found by two searches) are dropped.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    collected: dict[str, dict] = {}   # chunk id -> chunk, preserving first-seen order

    for _ in range(MAX_AGENT_ITERATIONS):
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[SEARCH_TOOL],
        )
        message = resp.choices[0].message

        if not message.tool_calls:
            return message.content, list(collected.values())

        # Append the assistant turn (with its tool calls) before the results.
        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            if verbose:
                print(f"  [tool call] search({args})")
            payload, chunks = _run_search_tool(
                args, method=method, rerank=rerank, rewrite=rewrite
            )
            for c in chunks:
                collected.setdefault(c["id"], c)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )

    # Hit the iteration cap — make one final call without tools to force an answer.
    resp = _client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content, list(collected.values())


def agentic_rag(
    query: str,
    verbose: bool = False,
    method: str = "hybrid",
    rerank: bool = True,
    rewrite: bool = False,
) -> str:
    """Agentic RAG: the model decides when/what to search via function calling."""
    answer, _ = agentic_rag_with_sources(
        query, verbose=verbose, method=method, rerank=rerank, rewrite=rewrite
    )
    return answer
