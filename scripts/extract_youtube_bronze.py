"""Extract immutable raw YouTube API responses into the Bronze layer.

This script deliberately performs *only* extraction.  It does not infer
Shorts, compute metrics, or create Silver tables; those are responsibilities
of a later transform step.  Every API response is retained so a transform can
be rerun without spending YouTube API quota again.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://www.googleapis.com/youtube/v3"
DEFAULT_MAX_UPLOADS_PER_CHANNEL = 300
REQUEST_RETRIES = 3


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_id_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_json_new(path: Path, value: object) -> None:
    """Write a new Bronze artefact, refusing accidental overwrites."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite Bronze artefact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_json_replace(path: Path, value: object) -> None:
    """Update the run manifest; manifests are operational metadata, not raw data."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unknown"


class YouTubeClient:
    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()

    def get(self, resource: str, **params: object) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(REQUEST_RETRIES):
            try:
                response = self.session.get(
                    f"{BASE_URL}/{resource}",
                    params={**params, "key": self.api_key},
                    timeout=30,
                )
                if response.ok:
                    return response.json()
                try:
                    message = response.json().get("error", {}).get("message", response.text)
                except ValueError:
                    message = response.text
                raise RuntimeError(f"YouTube API {response.status_code}: {message[:500]}")
            except (requests.RequestException, RuntimeError) as error:
                last_error = error
                if attempt + 1 < REQUEST_RETRIES:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Request failed after {REQUEST_RETRIES} attempts: {last_error}") from last_error


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def extract_channel(
    client: YouTubeClient,
    candidate: dict[str, str],
    attempt_dir: Path,
    max_uploads: int,
) -> dict[str, int]:
    """Save raw responses for one channel and return its extraction counts."""
    channel_id = candidate["channel_id"]
    channel_payload = client.get("channels", part="snippet,statistics,contentDetails", id=channel_id)
    write_json_new(attempt_dir / "channel.json", channel_payload)

    items = channel_payload.get("items", [])
    if not items:
        raise RuntimeError(f"Channel was not returned by the API: {channel_id}")
    uploads_playlist_id = items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    if not uploads_playlist_id:
        raise RuntimeError(f"Channel has no uploads playlist: {channel_id}")

    video_ids: list[str] = []
    page_token: str | None = None
    page_number = 1
    while len(video_ids) < max_uploads:
        payload = client.get(
            "playlistItems",
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=min(50, max_uploads - len(video_ids)),
            pageToken=page_token,
        )
        write_json_new(attempt_dir / "playlist_items" / f"page_{page_number:04d}.json", payload)
        video_ids.extend(
            item["contentDetails"]["videoId"]
            for item in payload.get("items", [])
            if item.get("contentDetails", {}).get("videoId")
        )
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
        page_number += 1

    video_batches = chunked(video_ids, 50)
    for batch_number, video_ids_batch in enumerate(video_batches, start=1):
        payload = client.get(
            "videos",
            part="snippet,contentDetails,statistics,status",
            id=",".join(video_ids_batch),
        )
        write_json_new(attempt_dir / "videos" / f"batch_{batch_number:04d}.json", payload)

    return {
        "playlist_pages": page_number,
        "requested_video_ids": len(video_ids),
        "video_batches": len(video_batches),
    }


def load_candidates(sectors: list[str]) -> list[dict[str, str]]:
    candidates = pd.read_csv(PROJECT_ROOT / "data" / "candidates" / "channels.csv", dtype="string").fillna("")
    selected = candidates.loc[candidates["sector"].isin(sectors)].copy()
    if selected.empty:
        raise ValueError(f"No candidates found for sectors: {', '.join(sectors)}")
    return selected.to_dict("records")


def load_or_create_manifest(run_dir: Path, args: argparse.Namespace, candidates: list[dict[str, str]]) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"Run already exists: {run_dir}. Use --resume to continue it.")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.resume:
        raise FileNotFoundError(f"Cannot resume a run that does not exist: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json_new(run_dir / "input" / "candidates_snapshot.json", candidates)
    manifest = {
        "run_id": run_dir.name,
        "pipeline": "youtube_bronze_extraction",
        "status": "running",
        "started_at": utc_now(),
        "parameters": {"sectors": args.sectors, "max_uploads_per_channel": args.max_uploads_per_channel},
        "candidate_count": len(candidates),
        "channels": {},
        "errors": [],
    }
    write_json_replace(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract raw YouTube Data API responses into Bronze.")
    parser.add_argument("--sectors", nargs="+", default=["moda", "automotivo"])
    parser.add_argument("--max-uploads-per-channel", type=int, default=DEFAULT_MAX_UPLOADS_PER_CHANNEL)
    parser.add_argument("--run-id", help="Existing run ID is accepted only with --resume.")
    parser.add_argument("--resume", action="store_true", help="Skip channels already completed in this run.")
    args = parser.parse_args()
    if args.max_uploads_per_channel < 1:
        raise ValueError("--max-uploads-per-channel must be at least 1")

    api_key = dotenv_values(PROJECT_ROOT / ".env").get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("Set YOUTUBE_API_KEY in .env before running the extractor.")

    candidates = load_candidates(args.sectors)
    run_id = args.run_id or run_id_now()
    run_dir = PROJECT_ROOT / "data" / "bronze" / "youtube" / run_id
    manifest = load_or_create_manifest(run_dir, args, candidates)
    client = YouTubeClient(api_key)

    for candidate in candidates:
        channel_id = candidate["channel_id"]
        previous = manifest["channels"].get(channel_id, {})
        if previous.get("status") == "completed":
            print(f"Skipping completed channel: {candidate['channel_name']}")
            continue
        print(f"Extracting {candidate['channel_name']}...")
        try:
            # A retry receives a fresh attempt directory.  Thus, a partial
            # attempt remains immutable and a resumed run never overwrites it.
            attempt_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            attempt_dir = run_dir / "extractions" / safe_component(channel_id) / attempt_id
            counts = extract_channel(client, candidate, attempt_dir, args.max_uploads_per_channel)
            manifest["channels"][channel_id] = {
                "status": "completed",
                "channel_name": candidate["channel_name"],
                "completed_at": utc_now(),
                "attempt_path": str(attempt_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                **counts,
            }
        except (RuntimeError, FileExistsError) as error:
            record = {"channel_id": channel_id, "channel_name": candidate["channel_name"], "occurred_at": utc_now(), "error": str(error)}
            manifest["channels"][channel_id] = {"status": "failed", **record}
            manifest["errors"].append(record)
            print(f"  failed: {error}")
        write_json_replace(run_dir / "run_manifest.json", manifest)

    completed = sum(item.get("status") == "completed" for item in manifest["channels"].values())
    failed = sum(item.get("status") == "failed" for item in manifest["channels"].values())
    manifest["status"] = "completed" if failed == 0 and completed == len(candidates) else "completed_with_errors"
    manifest["finished_at"] = utc_now()
    write_json_replace(run_dir / "run_manifest.json", manifest)
    print(f"Run {run_id}: {completed} channels completed, {failed} failed. Bronze only; no Silver was created.")


if __name__ == "__main__":
    main()
