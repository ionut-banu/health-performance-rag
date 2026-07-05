"""
Step 1: Fetch raw transcripts for a given source's episodes (huberman or
galpin). Loads the episode catalog from list_all_episodes.py output if
present (preferred — has duration/date/chapters already), falling back
to discover_videos.py's output, then to a manual dict.

Usage:
    uv run ingestion/fetch_transcripts.py --source huberman
    uv run ingestion/fetch_transcripts.py --source galpin

Uses youtube-transcript-api >= 1.2.x instance-based API:
    YouTubeTranscriptApi().fetch(video_id).to_raw_data()
The old static get_transcript() was removed in v1.2.0.
"""
import argparse
import json
import os
from youtube_transcript_api import YouTubeTranscriptApi

# Fallback if you'd rather hand-pick instead of running a discovery script.
MANUAL_VIDEO_IDS = {
    "VIDEO_ID_1": "Episode title 1",
    "VIDEO_ID_2": "Episode title 2",
}

CATALOG_PATH_TEMPLATE = "data/all_{source}_episodes.json"  # from list_all_episodes.py
OUTPUT_DIR_TEMPLATE = "data/raw_transcripts/{source}"


def load_episodes(source: str) -> dict:
    """
    Returns {video_id: {title, ...}} from the catalog produced by
    list_all_episodes.py. Falls back to MANUAL_VIDEO_IDS if no catalog
    exists yet (run list_all_episodes.py --source <source> first).
    """
    catalog_path = CATALOG_PATH_TEMPLATE.format(source=source)
    if os.path.exists(catalog_path):
        with open(catalog_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        print(f"Loaded {len(catalog)} episodes from {catalog_path}")
        return catalog

    print(f"No catalog found at {catalog_path} — using MANUAL_VIDEO_IDS.")
    print(f"Run: uv run ingestion/list_all_episodes.py --source {source}")
    return {vid: {"title": title} for vid, title in MANUAL_VIDEO_IDS.items()}


# Module-level instance — reused across all fetches in a run.
_ytt_api = YouTubeTranscriptApi()


def fetch_transcript(video_id: str) -> list[dict]:
    """Returns a list of {text, start, duration} segments."""
    return _ytt_api.fetch(video_id).to_raw_data()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["huberman", "galpin"], default="huberman")
    args = parser.parse_args()

    output_dir = OUTPUT_DIR_TEMPLATE.format(source=args.source)
    os.makedirs(output_dir, exist_ok=True)

    episodes = load_episodes(args.source)
    total = len(episodes)
    for i, (video_id, meta) in enumerate(episodes.items(), 1):
        title = meta.get("title", "")
        out_path = os.path.join(output_dir, f"{video_id}.json")

        if os.path.exists(out_path):
            print(f"[{i}/{total}] Already fetched, skipping: {title}")
            continue

        print(f"[{i}/{total}] Fetching: {title}")
        try:
            segments = fetch_transcript(video_id)
        except Exception as e:
            print(f"[{i}/{total}] Failed {video_id} ({title}): {e}")
            continue

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "video_id": video_id,
                    "title": title,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "upload_date": meta.get("upload_date"),
                    "description": meta.get("description"),
                    "tags": meta.get("tags", []),
                    "chapters": meta.get("chapters", []),
                    "segments": segments,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[{i}/{total}] Saved: {title} -> {out_path}")


if __name__ == "__main__":
    main()