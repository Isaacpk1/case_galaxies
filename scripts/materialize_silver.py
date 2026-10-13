"""Validate Bronze artefacts and materialize the three Silver contracts.

The command performs no network I/O and never deletes a Short because an
optional modality is unavailable.  It is deliberately separate from the
extractors so Silver can be recreated from an immutable Bronze run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

UTC = timezone.utc  # datetime.UTC only exists on Python >= 3.11
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MAX_SHORT_DURATION_SECONDS = 180
MIN_POST_AGE_DAYS = 30
HISTORY_MONTHS = 18


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_id_now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def eligible_shorts(youtube_run_id: str, min_history: int, min_post_age_days: int, history_months: int) -> pd.DataFrame:
    source = ROOT / "data" / "bronze" / "youtube" / youtube_run_id / "videos.json"
    if not source.exists():
        raise FileNotFoundError(f"Legacy Bronze videos file not found: {source}")
    data = pd.DataFrame(json.loads(source.read_text(encoding="utf-8")))
    data["published_at"] = pd.to_datetime(data["published_at"], utc=True, errors="coerce")
    for column in ("duration_seconds", "like_count", "view_count", "comment_count"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # Reuse the collection timestamp as the cut-off.  Using the current clock
    # would silently change a historical Silver run as videos age past 30 days.
    collected_at = pd.to_datetime(data.get("collected_at"), utc=True, errors="coerce")
    now = collected_at.max() if collected_at.notna().any() else pd.Timestamp.now(tz="UTC")
    eligible = data.loc[
        data["privacy_status"].eq("public")
        & data["duration_seconds"].between(1, MAX_SHORT_DURATION_SECONDS)
        & data["like_count"].notna()
        & data["published_at"].notna()
        & data["published_at"].between(now - pd.DateOffset(months=history_months), now - pd.Timedelta(days=min_post_age_days))
    ].copy()
    eligible["channel_eligible_shorts"] = eligible.groupby("channel_id")["video_id"].transform("nunique")
    eligible["meets_minimum_history"] = eligible["channel_eligible_shorts"].ge(min_history)
    eligible["source_run_id"] = youtube_run_id
    eligible["eligibility_status"] = "eligible"
    return eligible.sort_values(["channel_id", "published_at", "video_id"])


def eligible_shorts_many(youtube_run_ids: list[str], min_history: int, min_post_age_days: int, history_months: int) -> pd.DataFrame:
    """Combine several YouTube runs (e.g. one per sector); newest collection wins per video_id.

    Each run keeps its own collection cut-off, so a sector collected on another
    day is not re-filtered against a different clock.
    """
    frames = [eligible_shorts(run_id, min_history, min_post_age_days, history_months) for run_id in youtube_run_ids]
    combined = pd.concat(frames, ignore_index=True)
    combined["_collected"] = pd.to_datetime(combined.get("collected_at"), utc=True, errors="coerce")
    combined = combined.sort_values("_collected", ascending=False).drop_duplicates("video_id").drop(columns="_collected")
    combined["channel_eligible_shorts"] = combined.groupby("channel_id")["video_id"].transform("nunique")
    combined["meets_minimum_history"] = combined["channel_eligible_shorts"].ge(min_history)
    return combined.sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)


def thumbnail_manifest_many(thumbnail_run_ids: list[str]) -> pd.DataFrame:
    frames = [thumbnail_manifest(run_id) for run_id in thumbnail_run_ids]
    combined = pd.concat(frames, ignore_index=True)
    combined["_valid"] = (combined["validation_status"] != "valid").astype(int)
    return combined.sort_values(["video_id", "_valid", "collected_at"], ascending=[True, True, False]).drop_duplicates("video_id").drop(columns="_valid").reset_index(drop=True)


def thumbnail_manifest(thumbnail_run_id: str) -> pd.DataFrame:
    source = ROOT / "data" / "bronze" / "thumbnails" / "youtube" / thumbnail_run_id / "run_summary.json"
    if not source.exists():
        raise FileNotFoundError(f"Bronze thumbnail summary not found: {source}")
    rows = json.loads(source.read_text(encoding="utf-8"))["records"]
    validated: list[dict] = []
    for row in rows:
        record = dict(row)
        path = ROOT / str(record.get("local_path") or "")
        if record.get("status") == "downloaded" and path.is_file():
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    record.update({"width": image.width, "height": image.height, "image_format": image.format})
                record["sha256"] = sha256(path)
                record["validation_status"] = "valid"
            except OSError as error:
                record.update({"status": "failed", "error_code": "invalid_image", "validation_status": "invalid", "diagnostic": str(error)[-500:]})
        elif record.get("status") == "downloaded":
            record.update({"status": "failed", "error_code": "local_file_missing", "validation_status": "invalid"})
        else:
            record["validation_status"] = "not_available"
        validated.append(record)
    return pd.DataFrame(validated)


def count_snippets(path: Path) -> int:
    """JSON files (transcript-api / apify) carry `snippets`; yt-dlp writes WebVTT cues."""
    if path.suffix == ".vtt":
        return sum("-->" in line for line in path.read_text(encoding="utf-8").splitlines())
    return len(json.loads(path.read_text(encoding="utf-8")).get("snippets", []))


def is_latin_transcript(path: Path) -> bool:
    """Auto-dubbed tracks (Arabic, Hindi, ...) are not the Portuguese original and must not become features."""
    if path.suffix != ".json":
        return True
    snippets = json.loads(path.read_text(encoding="utf-8")).get("snippets", [])
    letters = [c for c in " ".join(str(x.get("text", "")) for x in snippets[:50]) if c.isalpha()]
    return not letters or sum(c.isascii() or "\u00c0" <= c <= "\u024f" for c in letters) / len(letters) >= 0.7


STATUS_PRIORITY = {"downloaded": 0, "not_available": 1, "rate_limited": 2, "blocked": 3, "failed": 4}


def transcript_manifest_many(transcript_run_ids: list[str]) -> pd.DataFrame:
    """Combine several transcript runs, keeping the best outcome per video_id (valid download first, then newest)."""
    frames = [transcript_manifest(run_id) for run_id in transcript_run_ids]
    combined = pd.concat(frames, ignore_index=True)
    combined["_valid"] = (combined["validation_status"] != "valid").astype(int)
    combined["_priority"] = combined["transcript_status"].map(STATUS_PRIORITY).fillna(9)
    combined = combined.sort_values(["video_id", "_valid", "_priority", "collected_at"], ascending=[True, True, True, False])
    return combined.drop_duplicates("video_id").drop(columns=["_valid", "_priority"]).reset_index(drop=True)


def transcript_manifest(transcript_run_id: str) -> pd.DataFrame:
    source = ROOT / "data" / "bronze" / "transcripts" / "youtube_caption_pilot" / transcript_run_id / "run_summary.json"
    if not source.exists():
        raise FileNotFoundError(f"Bronze transcript summary not found: {source}")
    rows = json.loads(source.read_text(encoding="utf-8"))["records"]
    validated: list[dict] = []
    for row in rows:
        record = dict(row)
        files = record.get("subtitle_files", [])
        if record.get("transcript_status") == "downloaded" and files:
            path = ROOT / files[0]["local_path"]
            if path.is_file():
                record["snippet_count"] = count_snippets(path)
                record["sha256"] = sha256(path)
                record["validation_status"] = "valid" if is_latin_transcript(path) else "wrong_language"
            else:
                record.update({"transcript_status": "failed", "error_code": "local_file_missing", "validation_status": "invalid"})
        else:
            record["validation_status"] = "not_available"
        validated.append(record)
    return pd.DataFrame(validated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize validated Silver contracts from Bronze runs.")
    parser.add_argument("--youtube-run-id", nargs="+", required=True,
                        help="One or more Bronze YouTube runs (e.g. one per sector); newest collection wins per video_id.")
    parser.add_argument("--thumbnail-run-id", nargs="+", required=True)
    parser.add_argument("--transcript-run-id", nargs="+",
                        help="Optional Bronze transcript run(s); with several runs the best outcome per video is kept.")
    parser.add_argument("--min-shorts-per-channel", type=int, default=30)
    parser.add_argument("--min-post-age-days", type=int, default=MIN_POST_AGE_DAYS)
    parser.add_argument("--history-months", type=int, default=HISTORY_MONTHS)
    parser.add_argument("--silver-run-id", default=run_id_now())
    args = parser.parse_args()
    if args.min_shorts_per_channel < 1 or args.min_post_age_days < 0 or args.history_months < 1:
        raise ValueError("History thresholds must be positive; minimum post age may be zero")

    eligible = eligible_shorts_many(args.youtube_run_id, args.min_shorts_per_channel, args.min_post_age_days, args.history_months)
    thumbnails = thumbnail_manifest_many(args.thumbnail_run_id)
    transcripts = transcript_manifest_many(args.transcript_run_id) if args.transcript_run_id else None
    contracts = [("eligible_shorts", eligible), ("thumbnail_manifest", thumbnails)]
    if transcripts is not None:
        contracts.append(("transcript_manifest", transcripts))
    for name, frame in contracts:
        output = ROOT / "data" / "silver" / name / args.silver_run_id / f"{name}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    write_json(
        ROOT / "data" / "silver" / "run_manifest" / args.silver_run_id / "run_manifest.json",
        {"run_id": args.silver_run_id, "created_at": datetime.now(UTC).isoformat(), "youtube_run_id": args.youtube_run_id, "thumbnail_run_id": args.thumbnail_run_id, "transcript_run_id": args.transcript_run_id, "eligible_shorts": len(eligible), "eligible_by_sector": eligible["sector"].value_counts().to_dict() if "sector" in eligible else {}, "thumbnail_records": len(thumbnails), "transcript_records": len(transcripts) if transcripts is not None else 0},
    )
    print(f"Silver run {args.silver_run_id}: {len(eligible)} eligible Shorts, {len(thumbnails)} thumbnails, {len(transcripts) if transcripts is not None else 0} transcripts.")


if __name__ == "__main__":
    main()
