"""
List videos from a YouTube channel with air date and duration, filtering
out short clips by duration and (optionally) episodes before a given
date. Works for any source defined in sources.yaml.

Runs in two passes:
1. Flat-list the channel to get every video ID + title (fast).
2. Full lookup per video to get accurate duration + upload date
   (slower — one request per video, but only needs to run once).

Resumable: if the output file already exists, videos already cataloged
are skipped — so interrupted runs continue where they left off.

Filter defaults are defined per-source in sources.yaml. CLI flags
always take precedence when provided.

Usage:
    uv run ingestion/list_all_episodes.py --source huberman
    uv run ingestion/list_all_episodes.py --source galpin
    uv run ingestion/list_all_episodes.py --source huberman --after-date 2024-01-01
    uv run ingestion/list_all_episodes.py --source huberman --limit 10

To add a new source, add it to ingestion/sources.yaml — no code changes needed.

Output:
    data/all_{source}_episodes.json
    {video_id: {title, duration_seconds, upload_date, description, tags,
    chapters}}, upload_date as YYYY-MM-DD. chapters is a list of
    {title, start_time, end_time} if the creator added them, else [].
    Note: chapters are NOT the transcript — they're separate creator
    markers. Joining them with transcript timestamps (from
    fetch_transcripts.py) is a separate step, not done here.
"""
import argparse
import json
import os
from datetime import datetime

import yaml
from yt_dlp import YoutubeDL

SOURCES_CONFIG = os.path.join(os.path.dirname(__file__), "sources.yaml")


def load_sources() -> dict:
    with open(SOURCES_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["sources"]


def load_existing_catalog(output_path: str) -> dict:
    """Load already-fetched entries so we can skip them on resume."""
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            catalog = json.load(f)
        print(f"Resuming — {len(catalog)} episodes already in {output_path}, skipping those.")
        return catalog
    return {}


def save_catalog(catalog: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)


def list_video_ids(channel_url: str) -> list[dict]:
    """Fast flat listing: just id + title for every video on the channel."""
    ydl_opts = {"extract_flat": True, "quiet": True, "skip_download": True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    return info.get("entries", [])


def get_video_details(video_id: str) -> dict | None:
    """Full lookup for one video: accurate duration + upload date."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {"quiet": True, "skip_download": True}
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"  Skipping {video_id}: {e}")
        return None

    upload_date_raw = info.get("upload_date")  # e.g. "20240115"
    upload_date = None
    if upload_date_raw:
        upload_date = datetime.strptime(upload_date_raw, "%Y%m%d").strftime("%Y-%m-%d")

    return {
        "title": info.get("title"),
        "duration_seconds": info.get("duration"),
        "upload_date": upload_date,
        "description": info.get("description"),
        "tags": info.get("tags") or [],
        "chapters": info.get("chapters") or [],  # {title, start_time, end_time} or []
    }


def main():
    sources = load_sources()

    parser = argparse.ArgumentParser(
        description="List YouTube channel episodes with filtering. CLI flags override sources.yaml defaults."
    )
    parser.add_argument(
        "--source",
        choices=list(sources.keys()),
        default="huberman",
        help=f"Which source to list. Defined in sources.yaml. Options: {list(sources.keys())}",
    )
    parser.add_argument(
        "--channel-url",
        default=None,
        help="Override the channel URL from sources.yaml.",
    )
    parser.add_argument(
        "--min-duration-minutes",
        type=int,
        default=None,  # None = fall back to sources.yaml default
        help="Drop episodes shorter than this (default per source in sources.yaml).",
    )
    parser.add_argument(
        "--after-date",
        default=None,  # None = fall back to sources.yaml default
        help="Only keep episodes on or after this date, YYYY-MM-DD (default per source in sources.yaml).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of videos to look up (useful for a quick test run).",
    )
    args = parser.parse_args()

    source_config = sources[args.source]
    source_defaults = source_config.get("defaults", {})

    channel_url = args.channel_url or source_config["channel_url"]

    # CLI flag wins; fall back to sources.yaml default; then hard fallback.
    min_duration = (
        args.min_duration_minutes
        if args.min_duration_minutes is not None
        else source_defaults.get("min_duration_minutes", 15)
    )
    min_seconds = min_duration * 60

    _raw_after = args.after_date or source_defaults.get("after_date")
    # YAML parses unquoted dates (e.g. 2024-01-01) as datetime.date objects,
    # not strings. Coerce to string so strptime always gets what it expects.
    # The real fix is quoting dates in sources.yaml, but this guards against it.
    after_date_str = str(_raw_after) if _raw_after is not None else None
    after_date = (
        datetime.strptime(after_date_str, "%Y-%m-%d") if after_date_str else None
    )

    output_path = f"data/all_{args.source}_episodes.json"

    print(f"Source:  {source_config['label']} ({channel_url})")
    print(f"Filters: min_duration={min_duration}min, after_date={after_date_str or 'none'}")

    catalog = load_existing_catalog(output_path)

    print(f"Listing videos from channel ...")
    entries = list_video_ids(channel_url)
    print(f"Found {len(entries)} total videos.")

    if args.limit:
        entries = entries[: args.limit]
        print(f"Limiting detail lookup to first {len(entries)} videos.")

    new_count = 0
    skipped_count = 0
    for i, entry in enumerate(entries, 1):
        video_id = entry["id"]

        if video_id in catalog:
            skipped_count += 1
            continue  # already cataloged — resumable run

        print(f"[{i}/{len(entries)}] Looking up {video_id} ...")
        details = get_video_details(video_id)
        if details is None:
            continue

        duration = details["duration_seconds"]
        if duration is not None and duration < min_seconds:
            continue  # likely a clip/short

        if after_date and details["upload_date"]:
            episode_date = datetime.strptime(details["upload_date"], "%Y-%m-%d")
            if episode_date < after_date:
                continue

        catalog[video_id] = details
        new_count += 1

        # Save after each new entry so progress survives interruptions.
        save_catalog(catalog, output_path)

    print(f"\n{len(catalog)} total episodes in catalog ({new_count} new, {skipped_count} already cached).")
    print(f"Output -> {output_path}")


if __name__ == "__main__":
    main()