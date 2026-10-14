"""OCR enrichment for thumbnails -> data/silver/features/thumbnail_ocr_v1/<run_id>/.

Engine: EasyOCR (latin_g2 recognizer: pt + en, CRAFT detector).  PaddleOCR was
the first choice but its native DLLs are blocked on the Inteli Windows machine
and it needs Python <=3.12; EasyOCR runs on CPU with plain pip wheels.

OCR runs on the content crop (centre 9:16 panel for pillarboxed Shorts), so
text is not read twice from the blurred sides; bboxes are in crop coordinates.

Resumable: raw detections are appended to ocr_raw.jsonl; rerun with the same
--run-id to continue.  Features are aggregated at the end.

Usage:
  python scripts/pipeline/extract_thumbnail_ocr.py                      # latest manifest
  python scripts/pipeline/extract_thumbnail_ocr.py --image-root <dir>   # images elsewhere
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_thumbnail_features import content_crop  # noqa: E402  same pillarbox rule as thumbnail_v1

ROOT = Path(__file__).resolve().parents[2]
FEATURE_VERSION = "thumbnail_ocr_v1"
OCR_ENGINE = "easyocr-latin_g2"
MIN_CONFIDENCE = 0.40  # boxes below this are kept in raw but excluded from features
CANVAS_SIZE = 1280


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def latest_run(base: Path) -> str:
    return sorted(p.name for p in base.iterdir() if p.is_dir())[-1]


def aggregate(video_id: str, width: int, height: int, boxes: list[dict]) -> dict:
    kept = [b for b in boxes if b["confidence"] >= MIN_CONFIDENCE]
    text = " ".join(b["text"].strip() for b in kept).strip()
    letters = [c for c in text if c.isalpha()]
    area = 0.0
    for b in kept:
        xs = [p[0] for p in b["bbox"]]
        ys = [p[1] for p in b["bbox"]]
        area += (max(xs) - min(xs)) * (max(ys) - min(ys))
    return {
        "video_id": video_id,
        "ocr_engine": OCR_ENGINE,
        "ocr_boxes_raw": len(boxes),
        "ocr_boxes": len(kept),
        "has_text": int(len(text) >= 3),
        "ocr_char_count": len(text.replace(" ", "")),
        "ocr_word_count": len(re.findall(r"\w+", text)),
        "ocr_text_area_pct": round(min(area / float(width * height), 1.0), 4) if width and height else None,
        "ocr_mean_confidence": round(sum(b["confidence"] for b in kept) / len(kept), 4) if kept else None,
        "ocr_uppercase_ratio": round(sum(c.isupper() for c in letters) / len(letters), 4) if letters else None,
        "ocr_has_digits": int(bool(re.search(r"\d", text))),
        "ocr_has_question": int("?" in text),
        "ocr_has_exclamation": int("!" in text),
        "ocr_text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest-run-id")
    parser.add_argument("--run-id", help="resume an existing OCR run")
    parser.add_argument("--image-root", type=Path, help="override folder holding <video_id>.jpg")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    import easyocr  # imported lazily so --help works without the dependency

    manifest_base = ROOT / "data" / "silver" / "thumbnail_manifest"
    manifest_run = args.manifest_run_id or latest_run(manifest_base)
    manifest = pd.read_csv(manifest_base / manifest_run / "thumbnail_manifest.csv")
    manifest = manifest.loc[manifest["validation_status"].eq("valid")]
    if args.limit:
        manifest = manifest.head(args.limit)

    run_id = args.run_id or run_id_now()
    out_dir = ROOT / "data" / "silver" / "features" / FEATURE_VERSION / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "ocr_raw.jsonl"
    done = set()
    if raw_path.exists():
        done = {json.loads(l)["video_id"] for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    reader = easyocr.Reader(["pt", "en"], gpu=False, verbose=False)
    with raw_path.open("a", encoding="utf-8") as raw:
        for i, rec in enumerate(manifest.itertuples(index=False), 1):
            if rec.video_id in done:
                continue
            path = (args.image_root / f"{rec.video_id}.jpg") if args.image_root else ROOT / str(rec.local_path)
            try:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                box, _ = content_crop(rgb)
                crop = rgb.crop(box)
                result = reader.readtext(np.asarray(crop), canvas_size=CANVAS_SIZE)
                boxes = [{"bbox": [[float(x), float(y)] for x, y in b], "text": t, "confidence": round(float(c), 4)}
                         for b, t, c in result]
                row = {"video_id": rec.video_id, "width": crop.width, "height": crop.height, "crop_box": list(box), "status": "ok", "boxes": boxes}
            except Exception as exc:
                row = {"video_id": rec.video_id, "width": int(rec.width), "height": int(rec.height), "status": "error", "error": repr(exc), "boxes": []}
            raw.write(json.dumps(row, ensure_ascii=False) + "\n")
            raw.flush()
            if i % 50 == 0:
                print(f"{i}/{len(manifest)}", flush=True)

    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    table = pd.DataFrame([aggregate(r["video_id"], r["width"], r["height"], r["boxes"]) for r in rows if r["status"] == "ok"])
    table.insert(0, "feature_version", FEATURE_VERSION)
    table.insert(0, "manifest_run_id", manifest_run)
    table.insert(0, "run_id", run_id)
    table.to_csv(out_dir / "thumbnail_ocr_features.csv", index=False)
    summary = {
        "run_id": run_id,
        "feature_version": FEATURE_VERSION,
        "ocr_engine": OCR_ENGINE,
        "min_confidence": MIN_CONFIDENCE,
        "canvas_size": CANVAS_SIZE,
        "manifest_run_id": manifest_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_valid_thumbnails": int(len(manifest)),
        "processed": len(rows),
        "errors": [r for r in rows if r["status"] != "ok"],
        "has_text_rate": round(float(table["has_text"].mean()), 4) if len(table) else None,
        "median_words_when_text": float(table.loc[table["has_text"].eq(1), "ocr_word_count"].median()) if len(table) else None,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(table)} rows -> {out_dir}")


if __name__ == "__main__":
    main()
