"""
Module 5: monitoring dashboard.

Reads the interaction log written by the chat page (data/feedback.db) and charts real
usage — volume, user satisfaction, latency, which episodes get cited, and which retrieval
configurations are actually used.

Everything here reflects genuine interactions; nothing is seeded. An empty dashboard means
the app hasn't been used yet, not that something is broken.
"""
import os
import sys
from collections import Counter

import pandas as pd
import streamlit as st

_PAGES_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_PAGES_DIR)
for path in (os.path.dirname(_APP_DIR), _APP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from feedback import load_interactions  # noqa: E402

st.set_page_config(page_title="Monitoring · Health RAG", page_icon="📊", layout="wide")


def load_frame() -> pd.DataFrame:
    rows = load_interactions()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], format="mixed", utc=True)
    df["day"] = df["ts"].dt.date
    df["latency_s"] = df["latency_ms"] / 1000.0
    df["vote_label"] = df["vote"].map({1: "👍 positive", -1: "👎 negative"}).fillna("no vote")
    df["config"] = df.apply(
        lambda r: f"{r['retriever']}{'+rerank' if r['rerank'] else ''}"
                  f"{' (agentic)' if r['agentic'] else ''}",
        axis=1,
    )
    return df


def main():
    st.title("📊 Monitoring")
    st.caption("Live usage and feedback from the chat page — no synthetic data.")

    df = load_frame()
    if df.empty:
        st.info(
            "No interactions logged yet. Ask a few questions on the main page "
            "(and vote on the answers), then come back — this dashboard reflects real usage only."
        )
        return

    voted = df[df["vote"].notna()]
    positive_rate = (voted["vote"] == 1).mean() * 100 if not voted.empty else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total questions", len(df))
    c2.metric("Feedback received", len(voted))
    c3.metric("Positive rate", f"{positive_rate:.0f}%" if positive_rate is not None else "—")
    c4.metric("Median latency", f"{df['latency_s'].median():.1f}s")

    st.divider()

    # 1 + 2 — volume over time, and how feedback splits.
    left, right = st.columns(2)
    with left:
        st.subheader("1 · Questions per day")
        st.bar_chart(df.groupby("day").size().rename("questions"))
    with right:
        st.subheader("2 · Feedback breakdown")
        st.bar_chart(df["vote_label"].value_counts().rename("count"))

    # 3 + 4 — satisfaction trend and latency distribution.
    left, right = st.columns(2)
    with left:
        st.subheader("3 · Positive rate over time")
        if voted.empty:
            st.caption("No votes yet — vote on a few answers to populate this.")
        else:
            daily = voted.groupby("day")["vote"].apply(lambda v: (v == 1).mean() * 100)
            st.line_chart(daily.rename("% positive"))
    with right:
        st.subheader("4 · Answer latency")
        st.bar_chart(
            df["latency_s"].round().value_counts().sort_index().rename("answers")
        )

    # 5 + 6 — what content gets used, and which configurations.
    left, right = st.columns(2)
    with left:
        st.subheader("5 · Most cited episodes")
        titles = Counter(
            s.get("title") for row in df["sources"] for s in row if s.get("title")
        )
        if titles:
            top = pd.Series(dict(titles.most_common(10))).rename("citations")
            st.bar_chart(top)
        else:
            st.caption("No sources recorded yet.")
    with right:
        st.subheader("6 · Retrieval configuration used")
        st.bar_chart(df["config"].value_counts().rename("questions"))

    # 7 — corpus coverage.
    st.subheader("7 · Citations by podcast source")
    srcs = Counter(
        s.get("source") for row in df["sources"] for s in row if s.get("source")
    )
    if srcs:
        st.bar_chart(pd.Series(dict(srcs)).rename("citations"))
    else:
        st.caption("No sources recorded yet.")

    with st.expander("Recent questions"):
        st.dataframe(
            df[["ts", "question", "config", "latency_s", "vote_label"]].head(25),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
