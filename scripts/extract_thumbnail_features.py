"""Extract visual features (no OCR) from validated thumbnails into Silver.

Reads a Silver ``thumbnail_manifest`` run and writes
``data/silver/features/thumbnail_v1/<run_id>/thumbnail_features.csv`` plus a
``run_summary.json``.  No network I/O; the Bronze images are never modified.

Features (one row per video_id):
- width, height, aspect_ratio, is_low_resolution
- letterbox_fraction: share of image area in near-black bars (hqdefault 4:3)
- is_vertical_pillarbox: the 16:9 thumbnail is a 9:16 frame in the centre with
  blurred/dimmed copies on the sides (~75% of the pilot Shorts).  When true,
  every pixel feature below is computed on the centre panel only, otherwise
  the dim sides would bias brightness down and duplicate text for OCR.
- brightness_mean / brightness_std (contrast), computed on the content area
- saturation_mean, colorfulness (Hasler & Suesstrunk 2003)
- dark_pixel_pct (<20% luma) and light_pixel_pct (>80% luma)
- edge_density: share of pixels with strong gradient (visual "busyness")
- dominant hue bucket (red/orange/yellow/green/cyan/blue/purple/pink/neutral)
- face/text are NOT here: OCR lives in thumbnail_ocr_v1.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FEATURE_VERSION = "thumbnail_v1"
DARK_LUMA = 0.20
LIGHT_LUMA = 0.80
BAR_LUMA = 0.06  # rows/cols darker than this (mean) are treated as letterbox bars
# Pillarbox rule, calibrated by eye on the 1,506 pilot thumbnails (2026-10-01):
# sides much darker than the centre, or moderately darker plus a sharp seam.
PILLAR_STRONG_RATIO = 0.50
PILLAR_WEAK_RATIO = 0.65
PILLAR_SEAM = 0.05
HUE_BUCKETS = [
    (0, 15, "red"), (15, 40, "orange"), (40, 70, "yellow"), (70, 160, "green"),
    (160, 200, "cyan"), (200, 260, "blue"), (260, 290, "purple"), (290, 340, "pink"), (340, 361, "red"),
]


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def latest_run(base: Path) -> str:
    runs = sorted(p.name for p in base.iterdir() if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs in {base}")
    return runs[-1]


def content_box(luma: np.ndarray) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) of the non-letterbox content area."""
    rows = luma.mean(axis=1) > BAR_LUMA
    cols = luma.mean(axis=0) > BAR_LUMA
    if not rows.any() or not cols.any():
        return 0, luma.shape[0], 0, luma.shape[1]
    r, c = np.where(rows)[0], np.where(cols)[0]
    return int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1


def pillarbox_box(luma: np.ndarray) -> tuple[int, int] | None:
    """Return (left, right) of a centred 9:16 panel if the image is pillarboxed."""
    h, w = luma.shape
    pw = h * 9 / 16
    if w < pw * 1.3:
        return None
    x0, x1 = int(round((w - pw) / 2)), int(round((w + pw) / 2))
    inner = luma[:, x0 + 5 : x1 - 5]
    outer = np.concatenate([luma[:, : max(x0 - 5, 1)], luma[:, x1 + 5 :]], axis=1)
    if inner.size == 0 or outer.size == 0:
        return None
    ratio = float(outer.mean()) / (float(inner.mean()) + 1e-3)
    seam = 0.0
    for b in (x0, x1):
        for d in range(-4, 5):
            c = b + d
            if 2 <= c < w - 2:
                seam = max(seam, float(np.median(np.abs(luma[:, c + 1] - luma[:, c - 2]))))
    if ratio < PILLAR_STRONG_RATIO or (ratio < PILLAR_WEAK_RATIO and seam >= PILLAR_SEAM):
        return x0, x1
    return None


def content_crop(rgb_img: Image.Image) -> tuple[tuple[int, int, int, int], bool]:
    """(left, top, right, bottom) of the real creative content, and pillarbox flag."""
    luma = np.asarray(rgb_img.convert("L"), dtype=np.float32) / 255.0
    top, bottom, left, right = content_box(luma)
    inner = pillarbox_box(luma[top:bottom, left:right])
    if inner:
        return (left + inner[0], top, left + inner[1], bottom), True
    return (left, top, right, bottom), False


def dominant_hue(hsv: np.ndarray) -> str:
    h, s, v = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]
    chromatic = (s > 0.25) & (v > 0.20)
    if chromatic.mean() < 0.10:
        return "neutral"
    hist = {}
    hues = h[chromatic]
    for lo, hi, name in HUE_BUCKETS:
        hist[name] = hist.get(name, 0) + int(((hues >= lo) & (hues < hi)).sum())
    return max(hist, key=hist.get)


def features(path: Path) -> dict:
    with Image.open(path) as image:
        rgb_img = image.convert("RGB")
        width, height = rgb_img.size
        hsv_img = rgb_img.convert("HSV")
    rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
    hsv = np.asarray(hsv_img, dtype=np.float32) / 255.0
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    top, bottom, left, right = content_box(luma)
    letterbox_fraction = 1.0 - (bottom - top) * (right - left) / float(width * height)
    (left, top, right, bottom), pillarbox = content_crop(rgb_img)
    c_rgb = rgb[top:bottom, left:right]
    c_luma = luma[top:bottom, left:right]
    c_hsv = hsv[top:bottom, left:right]

    r, g, b = c_rgb[..., 0], c_rgb[..., 1], c_rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))

    gy, gx = np.gradient(c_luma)
    edge_density = float((np.hypot(gx, gy) > 0.10).mean())

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 4),
        "is_low_resolution": int(width < 1280),
        "letterbox_fraction": round(float(letterbox_fraction), 4),
        "is_vertical_pillarbox": int(pillarbox),
        "content_width": right - left,
        "content_height": bottom - top,
        "brightness_mean": round(float(c_luma.mean()), 4),
        "brightness_std": round(float(c_luma.std()), 4),
        "saturation_mean": round(float(c_hsv[..., 1].mean()), 4),
        "colorfulness": round(colorfulness, 4),
        "dark_pixel_pct": round(float((c_luma < DARK_LUMA).mean()), 4),
        "light_pixel_pct": round(float((c_luma > LIGHT_LUMA).mean()), 4),
        "edge_density": round(edge_density, 4),
        "dominant_hue": dominant_hue(c_hsv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest-run-id", help="Silver thumbnail_manifest run (default: latest)")
    args = parser.parse_args()

    manifest_base = ROOT / "data" / "silver" / "thumbnail_manifest"
    manifest_run = args.manifest_run_id or latest_run(manifest_base)
    manifest = pd.read_csv(manifest_base / manifest_run / "thumbnail_manifest.csv")
    valid = manifest.loc[manifest["validation_status"].eq("valid")]

    run_id = run_id_now()
    rows, errors = [], []
    for record in valid.itertuples(index=False):
        path = ROOT / str(record.local_path)
        try:
            rows.append({"video_id": record.video_id, **features(path)})
        except Exception as exc:  # keep going; one bad file must not drop the run
            errors.append({"video_id": record.video_id, "local_path": str(record.local_path), "error": repr(exc)})

    out_dir = ROOT / "data" / "silver" / "features" / FEATURE_VERSION / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.insert(0, "feature_version", FEATURE_VERSION)
    table.insert(0, "manifest_run_id", manifest_run)
    table.insert(0, "run_id", run_id)
    table.to_csv(out_dir / "thumbnail_features.csv", index=False)

    numeric = table.select_dtypes("number").drop(columns=["width", "height"], errors="ignore")
    summary = {
        "run_id": run_id,
        "feature_version": FEATURE_VERSION,
        "manifest_run_id": manifest_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_valid_thumbnails": int(len(valid)),
        "rows_written": int(len(table)),
        "errors": errors,
        "dominant_hue_counts": table["dominant_hue"].value_counts().to_dict() if len(table) else {},
        "feature_stats": numeric.describe().round(4).to_dict() if len(table) else {},
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(table)} rows -> {out_dir.relative_to(ROOT)} ({len(errors)} errors)")


if __name__ == "__main__":
    main()
