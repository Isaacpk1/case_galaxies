"""Minimal versioned Feature Store on the local filesystem.

Layout
    data/feature_store/registry.json                       one entry per feature set
    data/feature_store/<name>/<feature_set_id>/features.csv one row per video_id
    data/feature_store/<name>/<feature_set_id>/metadata.json

A feature set is identified by ``<name>__<source_hash[:10]>``.  The hash covers
the extractor name and version, its parameters and the sha256 of every input
file, so the same inputs + code version always map to the same id, and any
change in Silver or in the extractor produces a new id instead of overwriting.

Rules enforced on write (the "regra critica" of the Gold layer):
- ``video_id`` is the key and must be unique;
- no column may be one of the video's own outcome metrics or a target;
- every set declares ``available_before_publish``; Gold refuses sets where it is false.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STORE = ROOT / "data" / "feature_store"
REGISTRY = STORE / "registry.json"
FORBIDDEN = {"like_count", "view_count", "comment_count"}
FORBIDDEN_PREFIXES = ("target_",)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash(name: str, extractor_name: str, extractor_version: str, params: dict, sources: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{name}|{extractor_name}|{extractor_version}|{json.dumps(params, sort_keys=True, default=str)}".encode())
    for path in sorted(sources, key=str):
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def read_registry() -> list[dict]:
    return json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else []


def check_columns(frame: pd.DataFrame) -> None:
    if "video_id" not in frame or frame["video_id"].duplicated().any():
        raise ValueError("Feature set must have a unique video_id column")
    leaked = [c for c in frame.columns if c in FORBIDDEN or c.startswith(FORBIDDEN_PREFIXES)]
    if leaked:
        raise ValueError(f"Outcome/target columns cannot be features: {leaked}")


def write_feature_set(
    *,
    name: str,
    group: str,
    extractor_name: str,
    extractor_version: str,
    frame: pd.DataFrame,
    sources: list[Path],
    params: dict,
    description: str,
    available_before_publish: bool = True,
    coverage_column: str | None = None,
    columns: dict[str, str] | None = None,
) -> dict:
    check_columns(frame)
    full_hash = source_hash(name, extractor_name, extractor_version, params, sources)
    feature_set_id = f"{name}__{full_hash[:10]}"
    out_dir = STORE / name / feature_set_id
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "features.csv", index=False)
    feature_columns = [c for c in frame.columns if c != "video_id"]
    metadata = {
        "feature_set_id": feature_set_id,
        "name": name,
        "group": group,
        "extractor_name": extractor_name,
        "extractor_version": extractor_version,
        "source_hash": full_hash,
        "sources": [str(p.relative_to(ROOT)).replace("\\", "/") for p in sources],
        "params": params,
        "available_before_publish": available_before_publish,
        "description": description,
        "rows": int(len(frame)),
        "coverage_column": coverage_column,
        "coverage_rate": float(frame[coverage_column].mean()) if coverage_column else 1.0,
        "feature_columns": feature_columns,
        "column_docs": columns or {},
        "path": str((out_dir / "features.csv").relative_to(ROOT)).replace("\\", "/"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    registry = [e for e in read_registry() if e["feature_set_id"] != feature_set_id]
    registry.append({k: metadata[k] for k in (
        "feature_set_id", "name", "group", "extractor_name", "extractor_version", "source_hash",
        "available_before_publish", "rows", "coverage_rate", "path", "created_at")})
    STORE.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def latest_feature_set(name: str) -> dict:
    entries = [e for e in read_registry() if e["name"] == name]
    if not entries:
        raise KeyError(f"No feature set named {name} in {REGISTRY}")
    return max(entries, key=lambda e: e["created_at"])


def load_feature_set(feature_set_id: str) -> tuple[pd.DataFrame, dict]:
    entry = next((e for e in read_registry() if e["feature_set_id"] == feature_set_id), None)
    if entry is None:
        raise KeyError(f"Unknown feature_set_id {feature_set_id}")
    folder = (ROOT / entry["path"]).parent
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return pd.read_csv(folder / "features.csv", dtype={"video_id": "string"}), metadata
