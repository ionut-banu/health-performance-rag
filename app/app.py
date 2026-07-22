"""
Streamlit chat interface for the health & performance RAG.

    uv run streamlit run app/app.py

Defaults match the Module 6 evaluation winner (hybrid retrieval + cross-encoder
re-ranking), so the app ships the configuration the measurements actually chose. The
sidebar exposes the other approaches so the comparison is reproducible in the UI.

Every answer is logged to data/feedback.db and can be voted on; the Dashboard page reads
that log. See docs/evaluation.md for why these defaults were picked.
"""
import logging
import os
import sys
import time

import streamlit as st

# Streamlit's file watcher walks every module in sys.modules looking for local sources to
# hot-reload, calling hasattr(m, "__path__") on each. `transformers` registers hundreds of
# lazy submodule placeholders whose __getattr__ triggers a real import on that check — and
# the vision ones (yolos, zoedepth, ...) import torchvision, which we don't install since
# nothing here processes images. The result is a wall of harmless ModuleNotFoundError
# tracebacks in the terminal on every rerun.
#
# Silencing just this logger keeps hot-reload working; the usual alternative
# (server.fileWatcherType = "none") would disable auto-reload entirely.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_APP_DIR)
for path in (_ROOT, os.path.join(_ROOT, "rag"), _APP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from rag import rag_with_sources, agentic_rag_with_sources  # noqa: E402
from feedback import log_interaction, record_vote  # noqa: E402

st.set_page_config(page_title="Health & Performance RAG", page_icon="🏋️", layout="centered")


@st.cache_resource(show_spinner="Loading indexes and models (first run only)…")
def warm_up():
    """
    Build/open the indexes and load the models exactly once per server process.

    Streamlit re-runs this script top-to-bottom on every widget interaction, so without
    this cache each click would rebuild the 27k-document keyword index and reload both
    transformer models. `cache_resource` is the right decorator here (not `cache_data`)
    because these are unserializable, long-lived objects.
    """
    from retrieve import get_keyword_index, get_vector_index
    from rerank import get_reranker

    get_keyword_index()
    get_vector_index()
    get_reranker()
    return True


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            ts = s.get("start_timestamp")
            url = s.get("url") or ""
            link = f"{url}&t={int(ts)}s" if ts is not None and url else url
            chapter = s.get("chapter_title") or ""
            stamp = f"{int(ts // 60)}:{int(ts % 60):02d}" if ts is not None else ""
            label = f"**{s.get('title', 'Untitled')}**"
            if chapter:
                label += f" — {chapter}"
            st.markdown(f"{label}  \n[{s.get('source', '')} · {stamp}]({link})")


def main():
    st.title("🏋️ Health & Performance RAG")
    st.caption(
        "Answers grounded in Huberman Lab and Andy Galpin's *Perform* transcripts — "
        "with citations you can jump to."
    )

    with st.sidebar:
        st.header("Retrieval settings")
        retriever = st.selectbox(
            "Retriever", ["hybrid", "vector", "keyword"], index=0,
            help="hybrid fuses keyword + vector and won the evaluation (MRR 0.614).",
        )
        rerank = st.checkbox(
            "Cross-encoder re-ranking", value=True,
            help="Biggest single accuracy gain: HR@1 0.387 → 0.523. Adds some latency.",
        )
        agentic = st.checkbox(
            "Agentic mode", value=False,
            help="The model searches repeatedly, reformulating. Slower; measured ≈ basic.",
        )
        num_results = st.slider("Sources to retrieve", 3, 10, 5)
        source = st.selectbox("Restrict to source", ["all", "huberman", "galpin"], index=0)
        st.divider()
        st.caption("Defaults are the measured winners — see `docs/evaluation.md`.")

    warm_up()

    question = st.text_input(
        "Ask a question",
        placeholder="e.g. how do I fall asleep faster?",
    )
    ask = st.button("Ask", type="primary")

    if ask and question.strip():
        source_filter = None if source == "all" else source
        started = time.perf_counter()
        try:
            with st.spinner("Retrieving and answering…"):
                if agentic:
                    answer, sources = agentic_rag_with_sources(
                        question, method=retriever, rerank=rerank
                    )
                else:
                    answer, sources = rag_with_sources(
                        question,
                        num_results=num_results,
                        source=source_filter,
                        method=retriever,
                        rerank=rerank,
                    )
        except Exception as e:
            st.error(f"Something went wrong answering that: {e}")
            st.stop()
        latency_ms = (time.perf_counter() - started) * 1000

        interaction_id = log_interaction(
            question=question,
            answer=answer,
            sources=sources,
            retriever=retriever,
            rerank=rerank,
            agentic=agentic,
            num_results=num_results,
            latency_ms=latency_ms,
        )
        # Survive the rerun that a vote button click triggers.
        st.session_state["last"] = {
            "id": interaction_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
        }
        st.session_state.pop("voted", None)

    last = st.session_state.get("last")
    if last:
        st.markdown("### Answer")
        st.markdown(last["answer"])
        render_sources(last["sources"])
        st.caption(f"Answered in {last['latency_ms'] / 1000:.1f}s")

        st.divider()
        if st.session_state.get("voted"):
            st.success("Thanks — your feedback was recorded.")
        else:
            st.write("Was this answer helpful?")
            up, down, _ = st.columns([1, 1, 6])
            if up.button("👍 Yes"):
                record_vote(last["id"], 1)
                st.session_state["voted"] = True
                st.rerun()
            if down.button("👎 No"):
                record_vote(last["id"], -1)
                st.session_state["voted"] = True
                st.rerun()


if __name__ == "__main__":
    main()
