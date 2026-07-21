"""
CLI for Module 1 RAG.

Usage:
    uv run rag/cli.py "how much protein should I eat to build muscle?"
    uv run rag/cli.py "how do I fall asleep faster?" --retriever vector
    uv run rag/cli.py "compare Huberman and Galpin on caffeine timing" --agentic
    uv run rag/cli.py "best exercises for VO2 max?" --source galpin --num-results 8
"""
import argparse

from rag import agentic_rag, rag


def main():
    parser = argparse.ArgumentParser(description="Ask the health/performance RAG.")
    parser.add_argument("query", help="The question to ask.")
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Use the agentic (function-calling) loop instead of basic RAG.",
    )
    parser.add_argument(
        "--source",
        choices=["huberman", "galpin"],
        default=None,
        help="Restrict retrieval to a single source (basic RAG only).",
    )
    parser.add_argument(
        "--retriever",
        choices=["keyword", "vector", "hybrid"],
        default="hybrid",
        help="Retrieval backend. hybrid (default) fuses keyword + vector and won the "
        "Module 6 evaluation; keyword and vector are the Module 1/2 baselines.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the cross-encoder re-ranking pass (faster, but measurably worse).",
    )
    parser.add_argument("--num-results", type=int, default=5)
    args = parser.parse_args()

    if args.agentic:
        answer = agentic_rag(
            args.query, verbose=True, method=args.retriever, rerank=not args.no_rerank
        )
    else:
        answer = rag(
            args.query,
            num_results=args.num_results,
            source=args.source,
            method=args.retriever,
            rerank=not args.no_rerank,
        )

    print("\n" + answer)


if __name__ == "__main__":
    main()
