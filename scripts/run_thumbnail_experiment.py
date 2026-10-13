"""E-06: do thumbnail features (visual + OCR) improve the rolling-origin baseline?

Same protocol as E-00..E-05: Ridge M1/M2/M3, expanding rolling-origin windows
inside the train split, alpha chosen on a chronological holdout of each fit
window, macro MAE per channel, paired bootstrap + exact sign permutation by
channel.  The final test split stays closed.

Scope: only the automotivo sector, because Silver thumbnails exist only for
those channels.  The reference models (without thumbnails) are re-fitted on the
same automotivo rows, so the comparison is apples to apples.

Arms (each fitted as M1, M2 and M3):
  BASE   baseline features only
  VIS    + thumbnail_v1 visual features
  OCR    + thumbnail_ocr_v1 flags/counts
  THUMB  + both
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_baseline_experiments import (
    ALPHAS,
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    BOOTSTRAP_SAMPLES,
    RANDOM_SEED,
    TARGET,
    RidgeTabularRegressor,
    evaluate,
    mae_by_channel,
    rolling_origin_splits,
    validation_split,
)

ROOT = Path(__file__).resolve().parents[1]
CONSUMPTION_PATH = ROOT / "data" / "gold" / "pilot_consumption_table.csv"
VISUAL_DIR = ROOT / "data" / "silver" / "features" / "thumbnail_v1"
OCR_DIR = ROOT / "data" / "silver" / "features" / "thumbnail_ocr_v1"

VISUAL_NUMERIC = [
    "is_vertical_pillarbox", "letterbox_fraction", "is_low_resolution", "brightness_mean",
    "brightness_std", "saturation_mean", "colorfulness", "dark_pixel_pct", "light_pixel_pct",
    "edge_density",
]
VISUAL_CATEGORICAL = ["dominant_hue"]
OCR_NUMERIC = [
    "has_text", "ocr_log_word_count", "ocr_text_area_pct", "ocr_uppercase_ratio",
    "ocr_has_digits", "ocr_has_question", "ocr_has_exclamation",
]
ARMS = {
    "BASE": ([], []),
    "VIS": (VISUAL_NUMERIC, VISUAL_CATEGORICAL),
    "OCR": (OCR_NUMERIC, []),
    "THUMB": (VISUAL_NUMERIC + OCR_NUMERIC, VISUAL_CATEGORICAL),
}


def latest_valid_run(base: Path) -> Path:
    runs = [p for p in sorted(base.iterdir()) if p.is_dir() and not (p / "SUPERSEDED.txt").exists()]
    return runs[-1]


def load_data() -> tuple[pd.DataFrame, dict[str, str]]:
    data = pd.read_csv(CONSUMPTION_PATH, parse_dates=["published_at"])
    visual_run, ocr_run = latest_valid_run(VISUAL_DIR), latest_valid_run(OCR_DIR)
    visual = pd.read_csv(visual_run / "thumbnail_features.csv")[["video_id"] + VISUAL_NUMERIC + VISUAL_CATEGORICAL]
    ocr = pd.read_csv(ocr_run / "thumbnail_ocr_features.csv")
    ocr["ocr_log_word_count"] = np.log1p(ocr["ocr_word_count"])
    ocr["ocr_uppercase_ratio"] = ocr["ocr_uppercase_ratio"].fillna(0.0)  # no text -> no uppercase
    ocr = ocr[["video_id"] + OCR_NUMERIC]
    data = data.loc[data["sector"].eq("automotivo") & data["split"].eq("train")]
    data = data.merge(visual, on="video_id", how="left", validate="one_to_one")
    data = data.merge(ocr, on="video_id", how="left", validate="one_to_one")
    coverage = {
        "rows": int(len(data)),
        "channels": int(data["channel_id"].nunique()),
        "visual_coverage": float(data["is_vertical_pillarbox"].notna().mean()),
        "ocr_coverage": float(data["has_text"].notna().mean()),
        "visual_run": visual_run.name,
        "ocr_run": ocr_run.name,
    }
    return data.reset_index(drop=True), coverage


def fit_predict(fit: pd.DataFrame, validation: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    development, holdout = validation_split(fit)
    scores = {}
    for alpha in ALPHAS:
        model = RidgeTabularRegressor(alpha, categorical, numeric).fit(development, development[TARGET])
        scores[alpha] = mae_by_channel(holdout, model.predict(holdout))
    alpha = min(ALPHAS, key=scores.get)
    return RidgeTabularRegressor(alpha, categorical, numeric).fit(fit, fit[TARGET]).predict(validation)


def arm_predictions(fit: pd.DataFrame, validation: pd.DataFrame, extra_num: list[str], extra_cat: list[str]) -> dict[str, np.ndarray]:
    numeric = BASE_NUMERIC_FEATURES + extra_num
    cat_all = BASE_CATEGORICAL_FEATURES + extra_cat
    cat_no_type = [c for c in cat_all if c != "channel_type"]
    out = {"M1": fit_predict(fit, validation, numeric, cat_all), "M3": fit_predict(fit, validation, numeric, cat_no_type)}
    m2 = np.empty(len(validation))
    for kind in sorted(fit["channel_type"].unique()):
        mask = validation["channel_type"].eq(kind).to_numpy()
        if mask.any():
            m2[mask] = fit_predict(fit.loc[fit["channel_type"].eq(kind)], validation.loc[mask], numeric, cat_no_type)
    out["M2"] = m2
    return out


def paired(per_channel: pd.DataFrame, candidate: str, reference: str) -> dict:
    values = per_channel.pivot(index="channel_name", columns="model", values="mae_log_likes")
    types = per_channel.drop_duplicates("channel_name").set_index("channel_name")["channel_type"]
    difference = (values[candidate] - values[reference])
    d = difference.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap = rng.choice(d, size=(BOOTSTRAP_SAMPLES, len(d)), replace=True).mean(axis=1)
    if len(d) <= 16:
        signs = np.array(list(itertools.product((-1, 1), repeat=len(d))))
        method = "exact"
    else:
        signs = rng.choice((-1, 1), size=(BOOTSTRAP_SAMPLES, len(d)))
        method = f"monte_carlo_{BOOTSTRAP_SAMPLES}"
    permutation = (signs * d).mean(axis=1)
    by_type = difference.groupby(types).mean()
    return {
        "comparison": f"{candidate} - {reference}",
        "macro_mae_difference": float(d.mean()),
        "bootstrap_ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_95_high": float(np.quantile(bootstrap, 0.975)),
        "paired_permutation_p_value": float((np.abs(permutation) >= abs(d.mean())).mean()),
        "permutation_method": method,
        "channels_improved": int((d < 0).sum()),
        "channels_worse": int((d > 0).sum()),
        "difference_by_channel_type": {k: float(v) for k, v in by_type.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "feature_experiments" / "e06_thumbnail")
    args = parser.parse_args()
    data, coverage = load_data()
    print(json.dumps(coverage, indent=2))

    all_predictions, all_metrics, definitions = [], [], []
    for origin, fit, validation in rolling_origin_splits(data):
        predictions = {"B0": validation["baseline_log_likes"].to_numpy()}
        for arm, (extra_num, extra_cat) in ARMS.items():
            for model, values in arm_predictions(fit, validation, extra_num, extra_cat).items():
                predictions[model if arm == "BASE" else f"{model}_{arm}"] = values
        for name, values in predictions.items():
            predicted, metric = evaluate(name, validation, values)
            predicted["origin"], metric["origin"] = origin, origin
            all_predictions.append(predicted)
            all_metrics.append(metric)
        definitions.append({"origin": origin, "train_rows": len(fit), "validation_rows": len(validation)})
        print(f"{origin} done", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    by_origin = metrics.groupby(["origin", "model"], as_index=False).agg(
        macro_mae_log_likes=("mae_log_likes", "mean"), macro_rmse_log_likes=("rmse_log_likes", "mean"),
        macro_bias_log_likes=("bias_log_likes", "mean"), macro_r2_log_likes=("r2_log_likes", "mean"),
        macro_pairwise_accuracy=("pairwise_accuracy", "mean"))
    summary = by_origin.groupby("model", as_index=False).mean(numeric_only=True).sort_values("macro_mae_log_likes")
    per_channel = metrics.groupby(["model", "channel_name", "channel_type"], as_index=False).mean(numeric_only=True)

    stats = {"coverage": coverage}
    for model in ("M1", "M2", "M3"):
        for arm in ("VIS", "OCR", "THUMB"):
            stats[f"{model.lower()}_{arm.lower()}_vs_{model.lower()}"] = paired(per_channel, f"{model}_{arm}", model)
    stats["m1_vs_b0"] = paired(per_channel, "M1", "B0")

    wins = by_origin.pivot(index="origin", columns="model", values="macro_mae_log_likes")
    stats["origins_where_thumb_beats_base"] = {
        m: int((wins[f"{m}_THUMB"] < wins[m]).sum()) for m in ("M1", "M2", "M3")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "rolling_origin_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "rolling_origin_per_channel_metrics.csv", index=False)
    by_origin.to_csv(args.output_dir / "rolling_origin_by_cut.csv", index=False)
    summary.to_csv(args.output_dir / "rolling_origin_summary.csv", index=False)
    pd.DataFrame(definitions).to_csv(args.output_dir / "rolling_origin_definitions.csv", index=False)
    (args.output_dir / "rolling_origin_comparison_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
