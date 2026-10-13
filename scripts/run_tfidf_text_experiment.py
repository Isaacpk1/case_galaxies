"""Evaluate leakage-safe TF-IDF text features with temporal rolling origins.

E-05 combines title, description and tags available before publication with
the frozen tabular baseline.  The final test split is deliberately excluded:
this script is for choosing whether semantic text merits a final evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_baseline_experiments import (
    ALPHAS,
    BOOTSTRAP_SAMPLES,
    RANDOM_SEED,
    TARGET,
    evaluate,
    pairwise_accuracy,
    rolling_origin_splits,
)


ROOT = Path(__file__).resolve().parents[1]
CONSUMPTION_PATH = ROOT / "data" / "gold" / "pilot_consumption_table.csv"
READY_PATH = ROOT / "data" / "gold" / "pilot_model_ready.csv"
BASE_NUMERIC = [
    "duration_seconds", "channel_prior_posts", "baseline_log_likes",
    "days_since_previous_post", "title_length", "description_length", "tag_count",
]
BASE_CATEGORICAL = ["channel_id", "sector", "channel_type", "publish_hour_utc", "publish_weekday"]


def make_text(data: pd.DataFrame) -> pd.Series:
    """Join only creative fields known at prediction time."""
    tags = data["tags"].fillna("").str.replace(r"[\[\]\"',]", " ", regex=True)
    return (data["title"].fillna("") + " " + data["description"].fillna("") + " " + tags).str.strip()


class TextTabularRidge:
    def __init__(self, alpha: float, categorical: list[str]):
        self.alpha = alpha
        self.categorical = categorical

    def fit(self, data: pd.DataFrame) -> "TextTabularRidge":
        numeric = [column for column in BASE_NUMERIC if column in data]
        self.tabular = ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), self.categorical),
        ])
        self.vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=2, max_features=500, sublinear_tf=True,
        )
        matrix = hstack([self.tabular.fit_transform(data), self.vectorizer.fit_transform(data["creative_text"])])
        self.model = Ridge(alpha=self.alpha, solver="lsqr").fit(matrix, data[TARGET])
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        matrix = hstack([self.tabular.transform(data), self.vectorizer.transform(data["creative_text"])])
        return self.model.predict(matrix)


def macro_mae(data: pd.DataFrame, prediction: np.ndarray) -> float:
    frame = data[["channel_id", TARGET]].copy()
    frame["prediction"] = prediction
    return float(frame.groupby("channel_id").apply(lambda group: np.abs(group[TARGET] - group["prediction"]).mean(), include_groups=False).mean())


def chronological_validation(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    development, validation = [], []
    for _, channel in data.sort_values(["channel_id", "published_at", "video_id"]).groupby("channel_id"):
        size = max(1, int(np.ceil(len(channel) * 0.20)))
        development.extend(channel.index[:-size])
        validation.extend(channel.index[-size:])
    return data.loc[development].copy(), data.loc[validation].copy()


def select_alpha(data: pd.DataFrame, categorical: list[str]) -> tuple[float, dict[str, float]]:
    development, validation = chronological_validation(data)
    scores = {}
    for alpha in ALPHAS:
        model = TextTabularRidge(alpha, categorical).fit(development)
        scores[str(alpha)] = macro_mae(validation, model.predict(validation))
    return min(ALPHAS, key=lambda value: scores[str(value)]), scores


def paired(per_channel: pd.DataFrame, candidate: str, reference: str) -> dict[str, float | str]:
    values = per_channel.pivot(index="channel_name", columns="model", values="mae_log_likes")
    difference = (values[candidate] - values[reference]).to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap = rng.choice(difference, size=(BOOTSTRAP_SAMPLES, len(difference)), replace=True).mean(axis=1)
    if len(difference) <= 16:
        signs = np.array(list(itertools.product((-1, 1), repeat=len(difference))))
        permutation_method = "exact"
    else:
        signs = rng.choice((-1, 1), size=(BOOTSTRAP_SAMPLES, len(difference)))
        permutation_method = f"monte_carlo_{BOOTSTRAP_SAMPLES}"
    permutation = (signs * difference).mean(axis=1)
    return {"comparison": f"{candidate} - {reference}", "macro_mae_difference": float(difference.mean()),
            "bootstrap_ci_95_low": float(np.quantile(bootstrap, .025)), "bootstrap_ci_95_high": float(np.quantile(bootstrap, .975)),
            "paired_permutation_p_value": float((np.abs(permutation) >= abs(difference.mean())).mean()),
            "permutation_method": permutation_method}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "feature_experiments" / "e05_tfidf")
    args = parser.parse_args()
    consumption = pd.read_csv(CONSUMPTION_PATH, parse_dates=["published_at"])
    raw = pd.read_csv(READY_PATH, usecols=["video_id", "title", "description", "tags"])
    data = consumption.merge(raw, on="video_id", how="left", validate="one_to_one")
    data["creative_text"] = make_text(data)
    train = data.loc[data["split"].eq("train")].copy()
    categorical_without_type = [column for column in BASE_CATEGORICAL if column != "channel_type"]
    all_predictions, all_metrics, definitions = [], [], []

    for origin, fit, validation in rolling_origin_splits(train):
        predictions = {"B0": validation["baseline_log_likes"].to_numpy()}
        alpha, _ = select_alpha(fit, BASE_CATEGORICAL)
        predictions["M1_TFIDF"] = TextTabularRidge(alpha, BASE_CATEGORICAL).fit(fit).predict(validation)
        alpha, _ = select_alpha(fit, categorical_without_type)
        predictions["M3_TFIDF"] = TextTabularRidge(alpha, categorical_without_type).fit(fit).predict(validation)
        m2 = np.empty(len(validation))
        for kind in sorted(fit["channel_type"].unique()):
            subgroup_fit = fit.loc[fit["channel_type"].eq(kind)]
            subgroup_validation = validation.loc[validation["channel_type"].eq(kind)]
            alpha, _ = select_alpha(subgroup_fit, categorical_without_type)
            m2[validation["channel_type"].eq(kind).to_numpy()] = TextTabularRidge(alpha, categorical_without_type).fit(subgroup_fit).predict(subgroup_validation)
        predictions["M2_TFIDF"] = m2
        for name, values in predictions.items():
            predicted, metric = evaluate(name, validation, values)
            predicted["origin"], metric["origin"] = origin, origin
            all_predictions.append(predicted); all_metrics.append(metric)
        definitions.append({"origin": origin, "train_rows": len(fit), "validation_rows": len(validation),
                            "temporal_order_valid_per_channel": True})

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    by_origin = metrics.groupby(["origin", "model"], as_index=False).agg(
        macro_mae_log_likes=("mae_log_likes", "mean"), macro_rmse_log_likes=("rmse_log_likes", "mean"),
        macro_bias_log_likes=("bias_log_likes", "mean"), macro_pairwise_accuracy=("pairwise_accuracy", "mean"))
    summary = by_origin.groupby("model", as_index=False).mean(numeric_only=True).sort_values("macro_mae_log_likes")
    per_channel = metrics.groupby(["model", "channel_name", "channel_type"], as_index=False).mean(numeric_only=True)
    stats = {"m1_tfidf_vs_b0": paired(per_channel, "M1_TFIDF", "B0"), "m2_tfidf_vs_m1_tfidf": paired(per_channel, "M2_TFIDF", "M1_TFIDF")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "rolling_origin_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "rolling_origin_per_channel_metrics.csv", index=False)
    by_origin.to_csv(args.output_dir / "rolling_origin_by_cut.csv", index=False)
    summary.to_csv(args.output_dir / "rolling_origin_summary.csv", index=False)
    pd.DataFrame(definitions).to_csv(args.output_dir / "rolling_origin_definitions.csv", index=False)
    (args.output_dir / "rolling_origin_comparison_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(summary.to_string(index=False)); print(json.dumps(stats, indent=2)); print(f"Artifacts saved in {args.output_dir}")


if __name__ == "__main__":
    main()
