"""Ingest a small, license-verified automotive video lot from Wikimedia Commons.

This is intentionally separate from YouTube performance collection: it tests the
licensed-media branch of the ELT architecture. Only CC0 or CC BY (without SA,
NC, or ND restrictions) assets are eligible for download.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "galaxies-elt-pilot/0.1 (licensed-media architecture validation)"
ACCEPTED_LICENSE = re.compile(r"^(cc0|cc by(?: [0-9.]+)?)$", re.IGNORECASE)


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_html(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(value or ""))).strip()


def field(metadata: dict, name: str) -> str:
    value = metadata.get(name, {})
    return clean_html(value.get("value") if isinstance(value, dict) else "")


def is_eligible_license(label: str) -> bool:
    normalized = re.sub(r"\s+", " ", label.strip().lower())
    return bool(ACCEPTED_LICENSE.fullmatch(normalized))


def commons_get(session: requests.Session, **params: object) -> dict:
    response = session.get(API_URL, params={"format": "json", **params}, timeout=45)
    response.raise_for_status()
    return response.json()


def find_candidates(session: requests.Session, query: str, limit: int) -> list[dict]:
    payload = commons_get(
        session,
        action="query",
        generator="search",
        gsrnamespace=6,
        gsrsearch=query,
        gsrlimit=min(limit * 5, 50),
        prop="imageinfo",
        iiprop="url|size|mime|sha1|extmetadata",
    )
    return list(payload.get("query", {}).get("pages", {}).values())


def as_manifest_record(page: dict, run_id: str, max_bytes: int) -> dict | None:
    info = (page.get("imageinfo") or [{}])[0]
    metadata = info.get("extmetadata") or {}
    mime = info.get("mime", "")
    license_name = field(metadata, "LicenseShortName")
    source_bytes = int(info.get("size") or 0)
    eligible = mime.startswith("video/") and source_bytes <= max_bytes and is_eligible_license(license_name)
    if not eligible:
        return None
    title = page["title"]
    return {
        "asset_id": f"commons:{page['pageid']}",
        "run_id": run_id,
        "sector": "automotivo",
        "source_platform": "Wikimedia Commons",
        "source_page_url": "https://commons.wikimedia.org/wiki/" + quote(title.replace(" ", "_")),
        "media_url": info["url"],
        "original_title": title,
        "mime": mime,
        "source_bytes": source_bytes,
        "source_sha1": info.get("sha1"),
        "license_short_name": license_name,
        "license_url": field(metadata, "LicenseUrl"),
        "author": field(metadata, "Artist"),
        "attribution": field(metadata, "Credit") or field(metadata, "Attribution"),
        "description": field(metadata, "ImageDescription"),
        "license_verified_at": now(),
        "license_policy": "accept_cc0_or_cc_by_only",
        "verification_status": "eligible_for_download",
        "is_native_short": False,
        "linkage_status": "no_platform_linkage",
    }


def screening_record(page: dict, max_bytes: int) -> dict:
    info = (page.get("imageinfo") or [{}])[0]
    metadata = info.get("extmetadata") or {}
    mime = info.get("mime", "")
    license_name = field(metadata, "LicenseShortName")
    source_bytes = int(info.get("size") or 0)
    reasons = []
    if not mime.startswith("video/"):
        reasons.append("not_video")
    if source_bytes > max_bytes:
        reasons.append("exceeds_size_limit")
    if not is_eligible_license(license_name):
        reasons.append("license_outside_policy")
    return {
        "asset_id": f"commons:{page.get('pageid')}",
        "original_title": page.get("title"),
        "mime": mime,
        "source_bytes": source_bytes,
        "license_short_name": license_name,
        "screening_status": "eligible" if not reasons else "rejected",
        "rejection_reasons": reasons,
        "screened_at": now(),
    }


def safe_filename(title: str, media_url: str) -> str:
    suffix = Path(media_url.split("?", 1)[0]).suffix.lower() or ".bin"
    raw_stem = title.removeprefix("File:")
    if raw_stem.lower().endswith(suffix):
        raw_stem = raw_stem[: -len(suffix)]
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw_stem).strip("._")[:100]
    return f"{stem}{suffix}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_media(path: Path) -> dict:
    try:
        output = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height", "-of", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(output.stdout)
        video = next((item for item in payload.get("streams", []) if item.get("width")), {})
        return {
            "duration_seconds": float(payload.get("format", {}).get("duration", 0) or 0),
            "video_codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "media_validation_status": "valid",
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as error:
        return {"media_validation_status": "unprobed", "media_validation_error": str(error)}


def download(session: requests.Session, record: dict, bronze_dir: Path) -> dict:
    filename = safe_filename(record["original_title"], record["media_url"])
    destination = bronze_dir / "files" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(record["media_url"], stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    record.update(
        {
            "local_path": str(destination.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "downloaded_at": now(),
            "local_bytes": destination.stat().st_size,
            "local_sha256": sha256(destination),
            "verification_status": "downloaded_and_hashed",
        }
    )
    record.update(probe_media(destination))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="automobile filetype:video", help="Wikimedia Commons search query")
    parser.add_argument("--limit", type=int, default=3, help="Maximum eligible files to download")
    parser.add_argument("--max-mb", type=int, default=40, help="Maximum source file size in MiB")
    parser.add_argument("--download", action="store_true", help="Download eligible files; otherwise only create a candidate manifest")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    bronze_dir = PROJECT_ROOT / "data" / "bronze" / "media" / "wikimedia_commons" / run_id
    silver_dir = PROJECT_ROOT / "data" / "silver" / "media_manifest" / run_id
    bronze_dir.mkdir(parents=True)
    silver_dir.mkdir(parents=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    max_bytes = args.max_mb * 1024 * 1024
    pages = find_candidates(session, args.query, args.limit)
    screening = [screening_record(page, max_bytes) for page in pages]
    candidates = [record for page in pages if (record := as_manifest_record(page, run_id, max_bytes))]
    candidates = candidates[: args.limit]
    write_json(bronze_dir / "commons_api_response.json", pages)
    write_json(bronze_dir / "license_screening.json", screening)
    records = [download(session, record, bronze_dir) for record in candidates] if args.download else candidates
    for record in records:
        record.setdefault("local_path", None)
        record.setdefault("local_sha256", None)
        record.setdefault("media_validation_status", "not_downloaded")
    write_json(silver_dir / "media_manifest.json", records)
    print(f"Run saved at {run_id}. Eligible assets: {len(records)}. Downloaded: {sum(record['local_path'] is not None for record in records)}.")
    for record in records:
        print(f"- {record['original_title']} | {record['license_short_name']} | {record['source_bytes']} bytes")


if __name__ == "__main__":
    main()
