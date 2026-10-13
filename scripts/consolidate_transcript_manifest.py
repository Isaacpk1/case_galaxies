"""Build one Silver transcript manifest from all Bronze caption runs.

One row is emitted for every eligible Short. Successful transcripts take
precedence over later failed attempts; unavailable modalities remain explicit
coverage flags rather than removing a Short from the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_attempts() -> list[dict]:
    attempts: list[dict] = []
    for summary in (ROOT / "data" / "bronze" / "transcripts" / "youtube_caption_pilot").glob("*/run_summary.json"):
        payload = json.loads(summary.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            attempts.append({**record, "bronze_run_id": summary.parent.name})
    return attempts


def priority(record: dict) -> tuple[int, str]:
    # A valid download is more useful than a later failed retry for the same ID.
    return (1 if record.get("transcript_status") == "downloaded" else 0, str(record.get("collected_at") or ""))


def validate_download(record: dict) -> dict:
    files = record.get("subtitle_files") or []
    if not files:
        return {**record, "transcript_status": "failed", "error_code": "missing_transcript_file", "validation_status": "invalid"}
    path = ROOT / files[0]["local_path"]
    if not path.is_file():
        return {**record, "transcript_status": "failed", "error_code": "local_file_missing", "validation_status": "invalid"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        snippets = payload.get("snippets", [])
        if not isinstance(snippets, list) or not snippets:
            raise ValueError("snippets missing or empty")
        if not all({"text", "start", "duration"}.issubset(item) for item in snippets):
            raise ValueError("snippet fields missing")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {**record, "transcript_status": "failed", "error_code": "invalid_transcript_json", "validation_status": "invalid", "diagnostic": str(error)[-500:]}
    return {
        **record,
        "sha256": sha256(path),
        "snippet_count": len(snippets),
        "has_timestamps": True,
        "language": record.get("language") or payload.get("language"),
        "language_code": record.get("language_code") or payload.get("language_code"),
        "validation_status": "valid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate Bronze caption attempts into a coverage-aware Silver manifest.")
    parser.add_argument("--youtube-run-id", required=True)
    parser.add_argument("--silver-run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()

    eligible_path = ROOT / "data" / "silver" / "eligible_shorts" / args.youtube_run_id / "eligible_shorts.csv"
    if not eligible_path.exists():
        raise FileNotFoundError(f"Eligible Shorts not found: {eligible_path}")
    eligible = pd.read_csv(eligible_path, dtype="string")[["video_id"]].drop_duplicates()
    by_video: dict[str, dict] = {}
    for record in load_attempts():
        video_id = record.get("video_id")
        if video_id and (video_id not in by_video or priority(record) > priority(by_video[video_id])):
            by_video[video_id] = record

    rows: list[dict] = []
    for video_id in eligible["video_id"]:
        record = by_video.get(video_id)
        if record is None:
            rows.append({"video_id": video_id, "transcript_status": "not_attempted", "has_transcript": False, "validation_status": "not_available"})
            continue
        if record.get("transcript_status") == "downloaded":
            record = validate_download(record)
        else:
            record = {**record, "validation_status": "not_available"}
        rows.append({**record, "has_transcript": record.get("transcript_status") == "downloaded" and record.get("validation_status") == "valid"})

    result = pd.DataFrame(rows)
    output = ROOT / "data" / "silver" / "transcript_manifest" / args.silver_run_id / "transcript_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Silver transcript run {args.silver_run_id}: {int(result['has_transcript'].sum())}/{len(result)} valid transcripts.")


if __name__ == "__main__":
    main()
