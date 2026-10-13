"""Build the Gold training snapshot from Silver + Feature Store.

One row per video_id, joining the selected feature sets, the targets and a
per-channel temporal split.  The snapshot is immutable and identified by a
``dataset_version``; its manifest records the Silver run, every feature_set_id
with extractor version and source hash, the temporal cut-offs and the result
of the leakage checks.

Targets (never features):
  target_log_likes                 log1p(likes) of the Short
  target_residual_log_likes        target_log_likes - baseline_log_likes (E-07)
  target_relative_to_prior_median  (likes+1) / (prior channel median + 1)

Usage:
  python scripts/build_gold_snapshot.py --silver-run-id 20261002T003000Z
  python scripts/build_gold_snapshot.py --silver-run-id ... --feature-set-ids context_v1__a19f640408 text_v1__...
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from feature_store import FORBIDDEN, FORBIDDEN_PREFIXES, ROOT, latest_feature_set, load_feature_set

DEFAULT_SETS = ["context_v1", "text_v1", "thumbnail_v1", "transcript_v1"]
SPINE = ["video_id", "channel_id", "channel_name", "published_at"]


def temporal_split(data: pd.DataFrame, test_fraction: float) -> pd.Series:
    """Most recent share of every channel is test; simultaneous posts stay on the same side."""
    split = pd.Series("train", index=data.index)
    for _, channel in data.sort_values(["published_at", "video_id"]).groupby("channel_id"):
        test_size = int(np.ceil(len(channel) * test_fraction))
        test_start = channel.iloc[-test_size]["published_at"]
        split.loc[channel.index[channel["published_at"].ge(test_start)]] = "test"
    return split


def check_baseline_is_strictly_prior(gold: pd.DataFrame, silver: pd.DataFrame, sample: int = 300) -> dict:
    """Recompute B0 from raw Silver likes for a sample of rows: only earlier posts may enter."""
    rng = np.random.default_rng(7)
    rows = gold.iloc[rng.choice(len(gold), size=min(sample, len(gold)), replace=False)]
    by_channel = {k: g for k, g in silver.groupby("channel_id")}
    max_error = 0.0
    for row in rows.itertuples(index=False):
        history = by_channel[row.channel_id]
        prior = history.loc[history["published_at"] < row.published_at, "like_count"]
        expected = float(np.log1p(prior.median())) if len(prior) else np.nan
        max_error = max(max_error, abs(expected - row.baseline_log_likes))
    return {"rows_checked": int(len(rows)), "max_abs_error": round(max_error, 9), "passed": bool(max_error < 1e-6)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--silver-run-id", required=True)
    parser.add_argument("--feature-set-ids", nargs="+", help="explicit ids; default = latest of each group")
    parser.add_argument("--min-prior-posts", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()

    silver_path = ROOT / "data" / "silver" / "eligible_shorts" / args.silver_run_id / "eligible_shorts.csv"
    silver = pd.read_csv(silver_path, dtype={"video_id": "string"})
    silver["published_at"] = pd.to_datetime(silver["published_at"], utc=True)
    silver["like_count"] = pd.to_numeric(silver["like_count"], errors="coerce")

    ids = args.feature_set_ids or [latest_feature_set(name)["feature_set_id"] for name in DEFAULT_SETS]
    gold = silver.loc[silver["meets_minimum_history"].astype(str).str.lower().eq("true"), SPINE + ["like_count"]].copy()
    feature_sets, owners = [], {}
    for feature_set_id in ids:
        frame, meta = load_feature_set(feature_set_id)
        if not meta["available_before_publish"]:
            raise ValueError(f"{feature_set_id} is not available before publication and cannot enter Gold")
        clash = [c for c in frame.columns if c != "video_id" and (c in gold.columns or c in owners)]
        if clash:
            raise ValueError(f"Column clash from {feature_set_id}: {clash}")
        owners.update({c: feature_set_id for c in frame.columns if c != "video_id"})
        gold = gold.merge(frame, on="video_id", how="left", validate="one_to_one")
        feature_sets.append({k: meta[k] for k in ("feature_set_id", "name", "group", "extractor_name", "extractor_version",
                                                  "source_hash", "coverage_rate", "feature_columns")})
    if "baseline_log_likes" not in gold:
        raise ValueError("Gold needs the context feature set (baseline_log_likes) to build the residual target")

    gold = gold.loc[gold["channel_prior_posts"].ge(args.min_prior_posts)].copy()
    gold["target_log_likes"] = np.log1p(gold["like_count"])
    gold["target_residual_log_likes"] = gold["target_log_likes"] - gold["baseline_log_likes"]
    gold["target_relative_to_prior_median"] = (gold["like_count"] + 1) / (gold["channel_prior_median_likes"] + 1)
    gold["split"] = temporal_split(gold, args.test_fraction)

    leakage_check = check_baseline_is_strictly_prior(gold, silver)
    gold = gold.drop(columns=["like_count"])
    feature_columns = list(owners)
    leaked = [c for c in feature_columns if c in FORBIDDEN or c.startswith(FORBIDDEN_PREFIXES)]
    order_ok = all(
        g.loc[g["split"].eq("train"), "published_at"].max() < g.loc[g["split"].eq("test"), "published_at"].min()
        for _, g in gold.groupby("channel_id")
    )
    if leaked or not order_ok or not leakage_check["passed"]:
        raise AssertionError(f"Leakage checks failed: leaked={leaked} order_ok={order_ok} baseline={leakage_check}")

    targets = ["target_log_likes", "target_residual_log_likes", "target_relative_to_prior_median"]
    gold = gold[SPINE + ["split"] + targets + feature_columns].sort_values(["channel_id", "published_at", "video_id"])

    version_hash = hashlib.sha256(json.dumps([args.silver_run_id, sorted(ids), args.min_prior_posts, args.test_fraction]).encode()).hexdigest()[:8]
    dataset_version = f"gold_v1_{args.silver_run_id}_{version_hash}"
    out_dir = ROOT / "data" / "gold" / "snapshots" / dataset_version
    out_dir.mkdir(parents=True, exist_ok=True)
    gold.to_csv(out_dir / "training_snapshot.csv", index=False)

    test_start = gold.loc[gold["split"].eq("test")].groupby("channel_name")["published_at"].min()
    coverage = {c: float(gold[c].mean()) for c in ("has_thumbnail", "has_transcript") if c in gold}
    manifest = {
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "silver_run_id": args.silver_run_id,
        "feature_set_ids": ids,
        "feature_sets": feature_sets,
        "targets": targets,
        "primary_target": "target_log_likes",
        "rules": {
            "min_prior_posts": args.min_prior_posts,
            "test_fraction_per_channel": args.test_fraction,
            "channels": "meets_minimum_history (>= 30 Shorts elegiveis na Silver)",
            "never_features": sorted(FORBIDDEN) + [p + "*" for p in FORBIDDEN_PREFIXES],
        },
        "temporal_cut": {
            "train_published_from": str(gold.loc[gold["split"].eq("train"), "published_at"].min()),
            "train_published_to": str(gold.loc[gold["split"].eq("train"), "published_at"].max()),
            "test_published_from": str(gold.loc[gold["split"].eq("test"), "published_at"].min()),
            "test_published_to": str(gold.loc[gold["split"].eq("test"), "published_at"].max()),
            "test_start_by_channel": {k: str(v) for k, v in test_start.items()},
        },
        "rows": {
            "total": int(len(gold)),
            "by_split": gold["split"].value_counts().to_dict(),
            "by_sector": gold["sector"].value_counts().to_dict(),
            "by_channel_type": gold["channel_type"].value_counts().to_dict(),
            "channels": int(gold["channel_id"].nunique()),
        },
        "coverage": coverage,
        "leakage_checks": {
            "no_outcome_columns_in_features": not leaked,
            "train_before_test_in_every_channel": order_ok,
            "baseline_recomputed_from_strictly_prior_posts": leakage_check,
            "all_feature_sets_available_before_publish": True,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    latest = ROOT / "data" / "gold" / "snapshots" / "LATEST.json"
    latest.write_text(json.dumps({"dataset_version": dataset_version, "path": str((out_dir / "training_snapshot.csv").relative_to(ROOT)).replace("\\", "/")}, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("dataset_version", "rows", "coverage", "leakage_checks")}, indent=2, default=str))


if __name__ == "__main__":
    main()
