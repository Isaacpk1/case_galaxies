"""Collect public YouTube metadata and produce a Data Discovery profile.

This collector is resumable at the channel level: raw files are saved after
each successful channel so a temporary network failure does not discard the run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_KEY = dotenv_values(PROJECT_ROOT / ".env").get("YOUTUBE_API_KEY")
BASE_URL = "https://www.googleapis.com/youtube/v3"
MAX_UPLOADS_PER_CHANNEL = 300
MAX_SHORT_DURATION_SECONDS = 180
MIN_POST_AGE_DAYS = 30
HISTORY_MONTHS = 18
MIN_SHORTS_PER_CHANNEL = 30
PREFERRED_SHORTS_PER_CHANNEL = 50


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, default=str, indent=2), encoding="utf-8")


def youtube_get(session: requests.Session, resource: str, **params: object) -> dict:
    for attempt in range(3):
        try:
            response = session.get(
                f"{BASE_URL}/{resource}",
                params={**params, "key": API_KEY},
                timeout=30,
            )
            if not response.ok:
                detail = response.json().get("error", {}).get("message", "unknown API error")
                raise RuntimeError(f"YouTube API {response.status_code}: {detail}")
            return response.json()
        except requests.RequestException:
            if attempt == 2:
                raise RuntimeError("Network error after three attempts.") from None
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def duration_seconds(value: str) -> int:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def collect_channel(session: requests.Session, candidate: dict[str, str]) -> tuple[dict, list[dict]]:
    channel_id = candidate["channel_id"]
    channel = youtube_get(session, "channels", part="snippet,statistics,contentDetails", id=channel_id)["items"][0]
    uploads_playlist_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    video_ids: list[str] = []
    page_token: str | None = None
    while len(video_ids) < MAX_UPLOADS_PER_CHANNEL:
        payload = youtube_get(
            session,
            "playlistItems",
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=min(50, MAX_UPLOADS_PER_CHANNEL - len(video_ids)),
            pageToken=page_token,
        )
        video_ids.extend(item["contentDetails"]["videoId"] for item in payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    video_items: list[dict] = []
    for start in range(0, len(video_ids), 50):
        payload = youtube_get(
            session,
            "videos",
            part="snippet,contentDetails,statistics,status",
            id=",".join(video_ids[start : start + 50]),
        )
        video_items.extend(payload.get("items", []))

    collected_at = datetime.now(UTC).isoformat()
    channel_record = {
        **candidate,
        "subscriber_count_at_collection": channel.get("statistics", {}).get("subscriberCount"),
        "video_count_at_collection": channel.get("statistics", {}).get("videoCount"),
        "country": channel.get("snippet", {}).get("country"),
        "uploads_playlist_id": uploads_playlist_id,
        "collected_at": collected_at,
    }
    records: list[dict] = []
    for video in video_items:
        snippet = video.get("snippet", {})
        details = video.get("contentDetails", {})
        statistics = video.get("statistics", {})
        status = video.get("status", {})
        thumbnails = snippet.get("thumbnails", {})
        thumbnail = thumbnails.get("maxres") or thumbnails.get("high") or thumbnails.get("medium") or thumbnails.get("default") or {}
        records.append(
            {
                "video_id": video["id"],
                "channel_id": channel_id,
                "channel_name": candidate["channel_name"],
                "sector": candidate["sector"],
                "market": candidate["market"],
                "channel_type": candidate.get("channel_type", "unknown"),
                "candidate_notes": candidate["notes"],
                "published_at": snippet.get("publishedAt"),
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "tags": json.dumps(snippet.get("tags", []), ensure_ascii=False),
                "duration_seconds": duration_seconds(details.get("duration", "")),
                "privacy_status": status.get("privacyStatus"),
                "view_count": statistics.get("viewCount"),
                "like_count": statistics.get("likeCount"),
                "comment_count": statistics.get("commentCount"),
                "thumbnail_url": thumbnail.get("url"),
                "collected_at": collected_at,
            }
        )
    return channel_record, records


def profile(channel_records: list[dict], video_records: list[dict], silver_dir: Path) -> None:
    channels = pd.DataFrame(channel_records)
    videos = pd.DataFrame(video_records)
    videos["published_at"] = pd.to_datetime(videos["published_at"], utc=True, errors="coerce")
    for column in ("like_count", "view_count", "comment_count"):
        videos[column] = pd.to_numeric(videos[column], errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=MIN_POST_AGE_DAYS)
    history_start = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=HISTORY_MONTHS)
    eligible = videos.loc[
        videos["privacy_status"].eq("public")
        & videos["published_at"].between(history_start, cutoff)
        & videos["duration_seconds"].between(1, MAX_SHORT_DURATION_SECONDS)
        & videos["like_count"].notna()
    ].copy()
    grouped = (
        eligible.groupby(["channel_id", "channel_name", "sector", "market", "channel_type"], dropna=False)
        .agg(
            eligible_shorts=("video_id", "nunique"),
            median_likes=("like_count", "median"),
            mean_likes=("like_count", "mean"),
            first_short=("published_at", "min"),
            last_short=("published_at", "max"),
        )
        .reset_index()
    )
    profiles = channels.merge(
        grouped,
        on=["channel_id", "channel_name", "sector", "market", "channel_type"],
        how="left",
    )
    profiles["eligible_shorts"] = profiles["eligible_shorts"].fillna(0).astype(int)
    profiles["meets_minimum_history"] = profiles["eligible_shorts"].ge(MIN_SHORTS_PER_CHANNEL)
    profiles["meets_preferred_history"] = profiles["eligible_shorts"].ge(PREFERRED_SHORTS_PER_CHANNEL)
    eligible.to_csv(silver_dir / "eligible_shorts.csv", index=False)
    profiles.to_csv(silver_dir / "channel_profile.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sectors", nargs="+", default=["moda", "automotivo"])
    args = parser.parse_args()
    if not API_KEY:
        raise RuntimeError("Set YOUTUBE_API_KEY in .env before running the collector.")

    candidates = pd.read_csv(PROJECT_ROOT / "data" / "candidates" / "channels.csv", dtype="string").fillna("")
    candidates = candidates[candidates["sector"].isin(args.sectors)].to_dict("records")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bronze_dir = PROJECT_ROOT / "data" / "bronze" / "youtube" / run_id
    silver_dir = PROJECT_ROOT / "data" / "silver" / "data_discovery" / run_id
    bronze_dir.mkdir(parents=True)
    silver_dir.mkdir(parents=True)
    channel_records: list[dict] = []
    video_records: list[dict] = []
    errors: list[dict] = []
    session = requests.Session()

    for candidate in candidates:
        print(f"Collecting {candidate['channel_name']}...")
        try:
            channel_record, records = collect_channel(session, candidate)
            channel_records.append(channel_record)
            video_records.extend(records)
        except (RuntimeError, IndexError, KeyError) as error:
            errors.append({"channel_id": candidate["channel_id"], "channel_name": candidate["channel_name"], "error": str(error)})
            print(f"  skipped: {error}")
        write_json(bronze_dir / "channels.json", channel_records)
        write_json(bronze_dir / "videos.json", video_records)
        write_json(bronze_dir / "errors.json", errors)

    if channel_records and video_records:
        profile(channel_records, video_records, silver_dir)
    print(f"Run saved at {run_id}. Channels: {len(channel_records)}. Videos: {len(video_records)}. Errors: {len(errors)}.")


if __name__ == "__main__":
    main()
