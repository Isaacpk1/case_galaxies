"""Download YouTube thumbnails referenced by eligible Shorts and create a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "galaxies-elt-pilot/0.1 (thumbnail feature extraction)"
MAX_BYTES = 10 * 1024 * 1024


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def extension(url: str, content_type: str) -> str:
    url_suffix = Path(urlparse(url).path).suffix.lower()
    if url_suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return url_suffix
    return {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(content_type.lower(), ".img")


def download_thumbnail(session: requests.Session, row: dict, files_dir: Path, run_id: str, source_run_id: str) -> dict:
    record = {
        "run_id": run_id,
        "source_run_id": source_run_id,
        "video_id": row["video_id"],
        "thumbnail_url": row.get("thumbnail_url"),
        "collected_at": datetime.now(UTC).isoformat(),
        "status": "failed",
    }
    url = row.get("thumbnail_url")
    if not url:
        return {**record, "status": "missing_url", "error_code": "thumbnail_url_missing"}
    try:
        response = session.get(url, timeout=45)
        response.raise_for_status()
        content = response.content
        if len(content) > MAX_BYTES:
            return {**record, "status": "failed", "error_code": "thumbnail_exceeds_size_limit"}
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
        image = Image.open(BytesIO(content))
        image.verify()
        image = Image.open(BytesIO(content))
        suffix = extension(url, content_type)
        destination = files_dir / f"{row['video_id']}{suffix}"
        destination.write_bytes(content)
        return {
            **record,
            "status": "downloaded",
            "error_code": None,
            "content_type": content_type,
            "local_path": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "bytes": len(content),
            "sha256": sha256(content),
            "width": image.width,
            "height": image.height,
            "image_format": image.format,
        }
    except (requests.RequestException, OSError) as error:
        return {**record, "status": "failed", "error_code": type(error).__name__, "diagnostic": str(error)[-500:]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download thumbnails for eligible YouTube Shorts and create a manifest.")
    parser.add_argument("--source-run-id", required=True, help="Run ID containing silver/data_discovery/<run_id>/eligible_shorts.csv")
    parser.add_argument("--sector", choices=["moda", "automotivo"], help="Optional sector filter")
    parser.add_argument("--video-id", nargs="*", help="Optional explicit video IDs")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum thumbnails to attempt; use 0 to collect every eligible, unique video in the source run.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be zero (all videos) or a positive integer")

    source = PROJECT_ROOT / "data" / "silver" / "data_discovery" / args.source_run_id / "eligible_shorts.csv"
    if not source.exists():
        raise FileNotFoundError(f"Eligible Shorts not found: {source}")
    videos = pd.read_csv(source, dtype="string").fillna("")
    if args.sector:
        videos = videos[videos["sector"].eq(args.sector)]
    if args.video_id:
        videos = videos[videos["video_id"].isin(args.video_id)]
    videos = videos.drop_duplicates("video_id")
    if args.limit:
        videos = videos.head(args.limit)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bronze_dir = PROJECT_ROOT / "data" / "bronze" / "thumbnails" / "youtube" / run_id
    files_dir = bronze_dir / "files"
    silver_dir = PROJECT_ROOT / "data" / "silver" / "thumbnail_manifest" / run_id
    files_dir.mkdir(parents=True)
    silver_dir.mkdir(parents=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    records = []
    for row in videos.to_dict("records"):
        print(f"Downloading thumbnail for {row['video_id']}...")
        record = download_thumbnail(session, row, files_dir, run_id, args.source_run_id)
        records.append(record)
        print(f"  {record['status']}")
    write_json(bronze_dir / "run_summary.json", {"run_id": run_id, "source_run_id": args.source_run_id, "records": records})
    write_json(silver_dir / "thumbnail_manifest.json", records)
    downloaded = sum(record["status"] == "downloaded" for record in records)
    print(f"Run saved at {run_id}. Downloaded thumbnails: {downloaded}/{len(records)}.")


if __name__ == "__main__":
    main()
