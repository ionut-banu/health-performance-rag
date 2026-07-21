"""
Step 4: Build the unified document set from fetched transcripts.
Reads data/raw_transcripts/{source}/*.json (see fetch_transcripts.py),
chunks each episode by chapter (see chunk_by_chapters.py), normalizes the
text of each chunk (see normalize.py), and writes data/documents.jsonl in
the schema defined in schema.py.

Usage:
    uv run ingestion/build_documents.py
    uv run ingestion/build_documents.py --source huberman
"""
import argparse
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunk_by_chapters import chunk_transcript
from normalize import normalize
from schema import Document, save_documents

SOURCES_CONFIG = os.path.join(os.path.dirname(__file__), "sources.yaml")
RAW_TRANSCRIPTS_DIR_TEMPLATE = "data/raw_transcripts/{source}"
OUTPUT_PATH = "data/documents.jsonl"


def load_sources() -> dict:
    with open(SOURCES_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


def build_documents_for_source(source: str, source_config: dict) -> list[Document]:
    skip_chapter_patterns = source_config.get("skip_chapter_patterns", [])
    filler_text_patterns = source_config.get("filler_text_patterns", [])

    transcript_dir = RAW_TRANSCRIPTS_DIR_TEMPLATE.format(source=source)
    transcript_paths = sorted(glob.glob(os.path.join(transcript_dir, "*.json")))

    documents = []
    for path in transcript_paths:
        with open(path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        chunks = chunk_transcript(transcript, skip_chapter_patterns)

        for chunk in chunks:
            text = normalize(chunk["segments"], filler_text_patterns)
            if not text:
                continue

            # parent_chunk_id is the id this chunk's chapter would have had before
            # sub-chunking, so evaluation ground truth collected on chapter-sized
            # chunks still resolves after the split (see docs/evaluation.md).
            parent_chunk_id = f"{source}_{transcript['video_id']}_{chunk['chapter_index']}"

            documents.append(
                Document(
                    id=f"{parent_chunk_id}_{chunk['sub_index']}",
                    source=source,
                    content_type="transcript",
                    title=transcript["title"],
                    url=transcript["url"],
                    text=text,
                    metadata={
                        "video_id": transcript["video_id"],
                        "start_timestamp": chunk["start_timestamp"],
                        "end_timestamp": chunk["end_timestamp"],
                        "upload_date": transcript.get("upload_date"),
                        "chapter_title": chunk["chapter_title"],
                        "parent_chunk_id": parent_chunk_id,
                        "chapter_index": chunk["chapter_index"],
                        "sub_index": chunk["sub_index"],
                    },
                )
            )

        print(f"  {transcript['video_id']}: {len(chunks)} chunks ({transcript['title']})")

    return documents


def main():
    sources = load_sources()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=list(sources.keys()),
        default=None,
        help="Build documents for a single source only. Default: all sources in sources.yaml.",
    )
    args = parser.parse_args()

    selected_sources = [args.source] if args.source else list(sources.keys())

    all_documents = []
    for source in selected_sources:
        print(f"Building documents for source: {source}")
        all_documents.extend(build_documents_for_source(source, sources[source]))

    save_documents(all_documents, OUTPUT_PATH)


if __name__ == "__main__":
    main()
