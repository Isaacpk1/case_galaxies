"""E-07: predict the deviation from the channel's historical median (B0).

Instead of predicting log likes from scratch (level), each Ridge learns
    residual = target_log_likes - baseline_log_likes
and the prediction is baseline_log_likes + predicted residual.  The model starts
from "what this channel usually gets" and only learns whether a given creative
should land above or below that.  This matches the business question: will
this creative beat the channel's normal?

Same protocol as E-00..E-06 (rolling-origin, alpha on chronological holdout,
paired bootstrap + exact/MC sign permutation per channel).  Test split closed.

Loss (--loss):
  ridge  squared error, same as E-00..E-06 (predicts the conditional mean)
  lad    absolute error via IRLS with the same ridge penalty (predicts the
         conditional median).  MAE is the project's main metric and log-likes
         residuals are skewed (a few viral Shorts), so a mean-predictor loses
         MAE to B0, which is itself a median.  LAD aligns loss and metric.

Scopes:
  automotivo  rows with thumbnails; arms BASE and THUMB, level and residual
  all         automotivo + moda; arm BASE only (no thumbnails for moda yet)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_baseline_experiments import (
    ALPHAS,
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    TARGET,
    RidgeTabularRegressor,
    evaluate,
    mae_by_channel,
    rolling_origin_splits,
    validation_split,
)
from run_thumbnail_experiment import (
    ARMS,
    CONSUMPTION_PATH,
    arm_predictions,
    load_data,
    paired,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "baseline_log_likes"


def lad_model(alpha: float, categorical: list[str], numeric: list[str], data: pd.DataFrame, iterations: int = 30) -> RidgeTabularRegressor:
    """Ridge-penalized least absolute deviations via iteratively reweighted least squares."""
    model = RidgeTabularRegressor(alpha, categorical, numeric).fit(data, data[TARGET])  # sets scalers/categories
    matrix = model._transform(data)
    target = data[TARGET].to_numpy(dtype=float)
    penalty = np.eye(matrix.shape[1])
    penalty[0, 0] = 0.0
    weights = np.ones(len(target))
    for _ in range(iterations):
        weighted = matrix * weights[:, None]
        coefficients = np.linalg.solve(matrix.T @ weighted + alpha * penalty, weighted.T @ target)
        weights = 1.0 / np.maximum(np.abs(target - matrix @ coefficients), 1e-3)
    model.coefficients_ = coefficients
    return model


def lad_fit_predict(fit: pd.DataFrame, validation: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    development, holdout = validation_split(fit)
    scores = {a: mae_by_channel(holdout, lad_model(a, categorical, numeric, development).predict(holdout)) for a in ALPHAS}
    alpha = min(ALPHAS, key=scores.get)
    return lad_model(alpha, categorical, numeric, fit).predict(validation)


def lad_arm_predictions(fit: pd.DataFrame, validation: pd.DataFrame, extra_num, extra_cat) -> dict[str, np.ndarray]:
    numeric = BASE_NUMERIC_FEATURES + extra_num
    cat_all = BASE_CATEGORICAL_FEATURES + extra_cat
    cat_no_type = [c for c in cat_all if c != "channel_type"]
    out = {"M1": lad_fit_predict(fit, validation, numeric, cat_all), "M3": lad_fit_predict(fit, validation, numeric, cat_no_type)}
    m2 = np.empty(len(validation))
    for kind in sorted(fit["channel_type"].unique()):
        mask = validation["channel_type"].eq(kind).to_numpy()
        if mask.any():
            m2[mask] = lad_fit_predict(fit.loc[fit["channel_type"].eq(kind)], validation.loc[mask], numeric, cat_no_type)
    out["M2"] = m2
    return out


def residual_predictions(fit: pd.DataFrame, validation: pd.DataFrame, extra_num, extra_cat, loss: str = "ridge") -> dict[str, np.ndarray]:
    shifted = fit.copy()
    shifted[TARGET] = fit[TARGET] - fit[BASELINE]
    predictor = lad_arm_predictions if loss == "lad" else arm_predictions
    out = predictor(shifted, validation, extra_num, extra_cat)
    offset = validation[BASELINE].to_numpy()
    return {name: values + offset for name, values in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scope", choices=["automotivo", "all"], default="automotivo")
    parser.add_argument("--loss", choices=["ridge", "lad"], default="ridge")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.output_dir or ROOT / "artifacts" / "feature_experiments" / f"e07_residual_{args.loss}_{args.scope}"

    if args.scope == "automotivo":
        data, coverage = load_data()
        arms = {k: ARMS[k] for k in ("BASE", "THUMB")}
    else:
        data = pd.read_csv(CONSUMPTION_PATH, parse_dates=["published_at"])
        data = data.loc[data["split"].eq("train")].reset_index(drop=True)
        coverage = {"rows": int(len(data)), "channels": int(data["channel_id"].nunique()),
                    "sectors": data["sector"].value_counts().to_dict()}
        arms = {"BASE": ARMS["BASE"]}
    print(json.dumps(coverage, indent=2), flush=True)

    all_metrics, all_predictions = [], []
    for origin, fit, validation in rolling_origin_splits(data):
        predictions = {"B0": validation[BASELINE].to_numpy()}
        for arm, (extra_num, extra_cat) in arms.items():
            suffix = "" if arm == "BASE" else f"_{arm}"
            for model, values in arm_predictions(fit, validation, extra_num, extra_cat).items():
                predictions[f"{model}{suffix}"] = values
            for model, values in residual_predictions(fit, validation, extra_num, extra_cat, args.loss).items():
                predictions[f"R_{model}{suffix}"] = values
        for name, values in predictions.items():
            predicted, metric = evaluate(name, validation, values)
            predicted["origin"], metric["origin"] = origin, origin
            all_predictions.append(predicted)
            all_metrics.append(metric)
        print(f"{origin} done", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    by_origin = metrics.groupby(["origin", "model"], as_index=False).agg(
        macro_mae_log_likes=("mae_log_likes", "mean"), macro_rmse_log_likes=("rmse_log_likes", "mean"),
        macro_bias_log_likes=("bias_log_likes", "mean"), macro_r2_log_likes=("r2_log_likes", "mean"),
        macro_pairwise_accuracy=("pairwise_accuracy", "mean"))
    summary = by_origin.groupby("model", as_index=False).mean(numeric_only=True).sort_values("macro_mae_log_likes")
    per_channel = metrics.groupby(["model", "channel_name", "channel_type"], as_index=False).mean(numeric_only=True)

    stats: dict = {"scope": args.scope, "loss": args.loss, "coverage": coverage}
    wins = by_origin.pivot(index="origin", columns="model", values="macro_mae_log_likes")
    for model in ("M1", "M2", "M3"):
        stats[f"r_{model.lower()}_vs_b0"] = paired(per_channel, f"R_{model}", "B0")
        stats[f"r_{model.lower()}_vs_{model.lower()}"] = paired(per_channel, f"R_{model}", model)
        if "THUMB" in arms:
            stats[f"r_{model.lower()}_thumb_vs_r_{model.lower()}"] = paired(per_channel, f"R_{model}_THUMB", f"R_{model}")
            stats[f"r_{model.lower()}_thumb_vs_b0"] = paired(per_channel, f"R_{model}_THUMB", "B0")
    stats["origins_beating_b0"] = {m: int((wins[m] < wins["B0"]).sum()) for m in wins.columns if m != "B0"}

    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "rolling_origin_predictions.csv", index=False)
    metrics.to_csv(out_dir / "rolling_origin_per_channel_metrics.csv", index=False)
    by_origin.to_csv(out_dir / "rolling_origin_by_cut.csv", index=False)
    summary.to_csv(out_dir / "rolling_origin_summary.csv", index=False)
    (out_dir / "rolling_origin_comparison_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
