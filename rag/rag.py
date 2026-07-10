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

from search import build_index, search

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

# Reuse a single client and index across calls.
_client = OpenAI()
_index = None


def get_index():
    global _index
    if _index is None:
        _index = build_index()
    return _index


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


def rag(query: str, num_results: int = 5, source: str | None = None) -> str:
    """Basic RAG: retrieve, stuff context into the prompt, answer in one call."""
    chunks = search(get_index(), query, num_results=num_results, source=source)
    context = build_context(chunks)
    user_prompt = f"Question: {query}\n\nContext:\n{context}"

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


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


def _run_search_tool(args: dict) -> list[dict]:
    chunks = search(
        get_index(),
        args["query"],
        num_results=args.get("num_results", 5),
        source=args.get("source"),
    )
    # Return only the fields the model needs to answer + cite.
    return [
        {
            "title": c["title"],
            "chapter_title": c.get("chapter_title") or "",
            "source_link": _timestamp_link(c),
            "text": c["text"],
        }
        for c in chunks
    ]


def agentic_rag(query: str, verbose: bool = False) -> str:
    """Agentic RAG: the model decides when/what to search via function calling."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for _ in range(MAX_AGENT_ITERATIONS):
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=[SEARCH_TOOL],
        )
        message = resp.choices[0].message

        if not message.tool_calls:
            return message.content

        # Append the assistant turn (with its tool calls) before the results.
        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments)
            if verbose:
                print(f"  [tool call] search({args})")
            results = _run_search_tool(args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(results, ensure_ascii=False),
                }
            )

    # Hit the iteration cap — make one final call without tools to force an answer.
    resp = _client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content
