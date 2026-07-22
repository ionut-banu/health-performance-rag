"""
Step 2: Fetch raw transcripts for a given source's episodes (huberman or
galpin). Loads the episode catalog from list_all_episodes.py output.

Usage:
    uv run ingestion/fetch_transcripts.py --source huberman
    uv run ingestion/fetch_transcripts.py --source huberman --delay 2.0   # gentler on YouTube
    uv run ingestion/fetch_transcripts.py --source huberman --retry-failed

Uses youtube-transcript-api >= 1.2.x instance-based API:
    YouTubeTranscriptApi().fetch(video_id).to_raw_data()
The old static get_transcript() was removed in v1.2.0.

RATE LIMITING
YouTube throttles bulk transcript requests — a long run typically succeeds for a few
hundred videos and then starts raising RequestBlocked. This script therefore:
  - sleeps `--delay` seconds (with jitter) between requests so it's less likely to trip,
  - retries *only* blocking errors with exponential backoff,
  - never retries per-video failures like "transcripts disabled" (those aren't transient),
  - stops the run entirely after `--block-limit` consecutive blocks, rather than burning
    through the remaining videos and marking them all failed (which deepens the block).
Fetching is resumable — already-downloaded transcripts are skipped — so when a run stops
early, just wait for the block to expire and run it again to continue.

If your IP stays blocked (common on cloud providers), the library supports routing through
a proxy; see "Working around IP bans" in the youtube-transcript-api README.
"""
import argparse
import json
import os
import random
import time

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    RequestBlocked,
    YouTubeRequestFailed,
)

# Fallback if you'd rather hand-pick instead of running a discovery script.
MANUAL_VIDEO_IDS = {
    "VIDEO_ID_1": "Episode title 1",
    "VIDEO_ID_2": "Episode title 2",
}

CATALOG_PATH_TEMPLATE = "data/all_{source}_episodes.json"  # from list_all_episodes.py
OUTPUT_DIR_TEMPLATE = "data/raw_transcripts/{source}"
# Kept OUTSIDE the transcripts dir: build_documents.py globs that dir for *.json and would
# try to parse a failures file as an episode.
FAILURES_PATH_TEMPLATE = "data/fetch_failures_{source}.json"

# IpBlocked subclasses RequestBlocked, so this covers both.
RETRYABLE_ERRORS = (RequestBlocked, YouTubeRequestFailed)

DEFAULT_DELAY = 1.0        # seconds between requests
DEFAULT_MAX_RETRIES = 4    # per video, on blocking errors only
DEFAULT_BACKOFF = 30.0     # first backoff; doubles each attempt
DEFAULT_BLOCK_LIMIT = 3    # consecutive blocked videos before giving up the run


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


def load_failures(source: str) -> dict:
    path = FAILURES_PATH_TEMPLATE.format(source=source)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_failures(source: str, failures: dict) -> None:
    path = FAILURES_PATH_TEMPLATE.format(source=source)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)


# Module-level instance — reused across all fetches in a run.
_ytt_api = YouTubeTranscriptApi()


def fetch_transcript(video_id: str) -> list[dict]:
    """Returns a list of {text, start, duration} segments."""
    return _ytt_api.fetch(video_id).to_raw_data()


def fetch_with_retry(
    video_id: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> list[dict]:
    """
    Fetch one transcript, retrying blocking errors with exponential backoff.

    Only RETRYABLE_ERRORS are retried — a video whose transcripts are disabled will fail
    the same way no matter how long we wait, so those raise immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return fetch_transcript(video_id)
        except RETRYABLE_ERRORS:
            if attempt == max_retries:
                raise
            wait = backoff * (2**attempt)
            wait += random.uniform(0, wait * 0.1)  # jitter, so retries don't sync up
            print(
                f"    rate-limited (attempt {attempt + 1}/{max_retries + 1}) — "
                f"waiting {wait:.0f}s before retrying…"
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["huberman", "galpin"], default="huberman")
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help="Seconds to wait between requests (jittered). Raise if you keep getting blocked.",
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--backoff", type=float, default=DEFAULT_BACKOFF,
        help="First backoff in seconds; doubles each retry.",
    )
    parser.add_argument(
        "--block-limit", type=int, default=DEFAULT_BLOCK_LIMIT,
        help="Stop the run after this many consecutive blocked videos.",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="Also re-attempt videos previously recorded as permanently unavailable.",
    )
    args = parser.parse_args()

    output_dir = OUTPUT_DIR_TEMPLATE.format(source=args.source)
    os.makedirs(output_dir, exist_ok=True)

    episodes = load_episodes(args.source)
    failures = {} if args.retry_failed else load_failures(args.source)
    total = len(episodes)

    saved = skipped = permanent = 0
    consecutive_blocks = 0
    stopped_early = False

    for i, (video_id, meta) in enumerate(episodes.items(), 1):
        title = meta.get("title", "")
        out_path = os.path.join(output_dir, f"{video_id}.json")

        if os.path.exists(out_path):
            skipped += 1
            continue
        if video_id in failures:
            print(f"[{i}/{total}] Known unavailable, skipping: {title}")
            skipped += 1
            continue

        print(f"[{i}/{total}] Fetching: {title}")
        try:
            segments = fetch_with_retry(video_id, args.max_retries, args.backoff)
        except RETRYABLE_ERRORS:
            consecutive_blocks += 1
            print(
                f"[{i}/{total}] Still blocked after {args.max_retries} retries: {title}"
            )
            if consecutive_blocks >= args.block_limit:
                print(
                    f"\nStopping: {consecutive_blocks} videos in a row were blocked. Your IP is "
                    "rate-limited, and continuing would only prolong it.\nWait a while (often "
                    "30-60+ min), then re-run the same command — completed transcripts are "
                    "skipped, so it picks up where it left off. Consider a larger --delay."
                )
                stopped_early = True
                break
            continue
        except CouldNotRetrieveTranscript as e:
            # Not transient: transcripts disabled, no transcript, video unavailable, etc.
            consecutive_blocks = 0
            permanent += 1
            failures[video_id] = {"title": title, "error": type(e).__name__}
            print(f"[{i}/{total}] Unavailable ({type(e).__name__}), recorded: {title}")
            continue

        consecutive_blocks = 0
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
        saved += 1
        print(f"[{i}/{total}] Saved: {title} -> {out_path}")

        # Throttle only after a real request, and only if more work remains.
        if i < total:
            time.sleep(args.delay + random.uniform(0, args.delay * 0.25))

    if failures:
        save_failures(args.source, failures)

    on_disk = len([f for f in os.listdir(output_dir) if f.endswith(".json")])
    print(
        f"\nDone. saved={saved} skipped={skipped} unavailable={permanent} "
        f"| {on_disk}/{total} transcripts on disk"
    )
    if stopped_early or on_disk < total:
        print("Re-run the same command to continue fetching the rest.")


if __name__ == "__main__":
    main()
