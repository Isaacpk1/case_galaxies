"""Download Short thumbnails using only the Python standard library.

Why this exists: the Inteli Windows Application Control policy blocks the
native DLLs of pandas/numpy/Pillow, so the original collector cannot even start
there.  This version needs no third-party package.  It writes the same Bronze
layout and run_summary.json contract as the former requests-based collector; image
validation (Pillow) happens later in materialize_silver.py, on a machine where
Pillow loads.

Usage (PowerShell, inside case_galaxies):
  python scripts/coleta/collect_youtube_thumbnails_stdlib.py --source-run-id 20260928T213206Z --sector moda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "galaxies-elt-pilot/0.1 (thumbnail feature extraction)"
MAX_BYTES = 10 * 1024 * 1024
EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        content = response.read(MAX_BYTES + 1)
        return content, response.headers.get("Content-Type", "").split(";", 1)[0]


def download(row: dict, files_dir: Path, run_id: str, source_run_id: str) -> dict:
    video_id = row["video_id"]
    url = row.get("thumbnail_url") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    record = {"run_id": run_id, "source_run_id": source_run_id, "video_id": video_id,
              "thumbnail_url": url, "collected_at": now(), "status": "failed"}
    try:
        content, content_type = fetch(url)
        if len(content) > MAX_BYTES:
            return {**record, "error_code": "thumbnail_exceeds_size_limit"}
        if not content_type.startswith("image/"):
            return {**record, "error_code": "not_an_image", "content_type": content_type}
        destination = files_dir / f"{video_id}{EXTENSIONS.get(content_type, '.jpg')}"
        destination.write_bytes(content)
        return {**record, "status": "downloaded", "error_code": None, "content_type": content_type,
                "local_path": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    except urllib.error.HTTPError as error:
        return {**record, "error_code": f"HTTP{error.code}"}
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        return {**record, "error_code": type(error).__name__, "diagnostic": str(error)[-500:]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download thumbnails (stdlib only).")
    parser.add_argument("--source-run-id", required=True,
                        help="Run with data/silver/data_discovery/<id>/eligible_shorts.csv or data/silver/eligible_shorts/<id>/eligible_shorts.csv")
    parser.add_argument("--sector", choices=["moda", "automotivo"])
    parser.add_argument("--limit", type=int, default=0, help="0 = all videos")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Pause between downloads")
    args = parser.parse_args()

    candidates = [PROJECT_ROOT / "data" / "silver" / folder / args.source_run_id / "eligible_shorts.csv"
                  for folder in ("data_discovery", "eligible_shorts")]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError(f"eligible_shorts.csv not found for {args.source_run_id}")
    with source.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if not args.sector or r.get("sector") == args.sector]
    seen, videos = set(), []
    for row in rows:
        if row["video_id"] not in seen:
            seen.add(row["video_id"])
            videos.append(row)
    if args.limit:
        videos = videos[: args.limit]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bronze_dir = PROJECT_ROOT / "data" / "bronze" / "thumbnails" / "youtube" / run_id
    files_dir = bronze_dir / "files"
    files_dir.mkdir(parents=True)
    print(f"{len(videos)} thumbnails to download -> {bronze_dir.relative_to(PROJECT_ROOT)}", flush=True)

    records = []
    for index, row in enumerate(videos, 1):
        records.append(download(row, files_dir, run_id, args.source_run_id))
        if index % 100 == 0 or index == len(videos):
            ok = sum(r["status"] == "downloaded" for r in records)
            print(f"  {index}/{len(videos)}  downloaded={ok}", flush=True)
            # Save progress so an interrupted run still leaves a usable summary.
            (bronze_dir / "run_summary.json").write_text(json.dumps(
                {"run_id": run_id, "source_run_id": args.source_run_id, "collector": "stdlib", "records": records},
                ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(args.sleep_seconds)

    downloaded = sum(r["status"] == "downloaded" for r in records)
    print(f"Run {run_id}: {downloaded}/{len(records)} thumbnails downloaded.")


if __name__ == "__main__":
    main()
