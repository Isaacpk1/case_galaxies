"""Create an append-only, cross-source Bronze collection manifest."""

from __future__ import annotations

import argparse
import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype="string").fillna("")


def as_record(**values: object) -> dict[str, object]:
    return values


def first_local_path(value: str) -> str | None:
    if not value:
        return None
    try:
        files = ast.literal_eval(value)
        return files[0].get("local_path") if files else None
    except (SyntaxError, ValueError, AttributeError, IndexError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate YouTube, thumbnail and transcript collection manifests.")
    parser.add_argument("--eligible-silver-run-id", required=True)
    parser.add_argument("--thumbnail-silver-run-id", required=True)
    parser.add_argument("--transcript-silver-run-id", required=True)
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()

    eligible = read_csv(ROOT / "data" / "silver" / "eligible_shorts" / args.eligible_silver_run_id / "eligible_shorts.csv")
    thumbnails = read_csv(ROOT / "data" / "silver" / "thumbnail_manifest" / args.thumbnail_silver_run_id / "thumbnail_manifest.csv")
    transcripts = read_csv(ROOT / "data" / "silver" / "transcript_manifest" / args.transcript_silver_run_id / "transcript_manifest.csv")

    records: list[dict[str, object]] = []
    for row in eligible.to_dict("records"):
        video_id = row["video_id"]
        records.append(as_record(
            origin="youtube_data_api",
            run_id=row.get("source_run_id"),
            video_id=video_id,
            collected_at=row.get("collected_at"),
            source_url=f"https://www.youtube.com/watch?v={video_id}",
            local_path=f"data/bronze/youtube/{row.get('source_run_id')}/videos.json",
            sha256=None,
            status="collected",
            error_code=None,
        ))
    for row in thumbnails.to_dict("records"):
        records.append(as_record(
            origin="youtube_thumbnail",
            run_id=row.get("run_id"),
            source_run_id=row.get("source_run_id"),
            video_id=row["video_id"],
            collected_at=row.get("collected_at"),
            source_url=row.get("thumbnail_url"),
            local_path=row.get("local_path"),
            sha256=row.get("sha256"),
            status=row.get("status"),
            error_code=row.get("error_code") or None,
        ))
    for row in transcripts.to_dict("records"):
        local_path = first_local_path(row.get("subtitle_files", ""))
        records.append(as_record(
            origin="youtube_transcript",
            run_id=row.get("bronze_run_id") or row.get("run_id"),
            video_id=row["video_id"],
            collected_at=row.get("collected_at") or None,
            source_url=row.get("youtube_url") or f"https://www.youtube.com/watch?v={row['video_id']}",
            local_path=local_path,
            sha256=row.get("sha256") or None,
            status=row.get("transcript_status"),
            error_code=row.get("error_code") or None,
        ))

    output_dir = ROOT / "data" / "bronze" / "collection_manifest" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "collection_manifest.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "eligible_silver_run_id": args.eligible_silver_run_id,
        "thumbnail_silver_run_id": args.thumbnail_silver_run_id,
        "transcript_silver_run_id": args.transcript_silver_run_id,
        "records": len(records),
        "origins": pd.Series([record["origin"] for record in records]).value_counts().to_dict(),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Collection manifest {args.run_id}: {len(records)} records.")


if __name__ == "__main__":
    main()
