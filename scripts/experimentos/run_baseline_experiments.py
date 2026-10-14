"""Run leakage-safe tabular baseline experiments for the Shorts pilot.

Experiments:
  B0: prior median of likes for the channel.
  M1: global Ridge regression, including channel_type.
  M2: one global Ridge regression per channel_type.
  M3: global Ridge regression without channel_type.

The final test split is never used for alpha selection.  No scikit-learn
dependency is required: the small Ridge implementation keeps the pilot
reproducible with the current project environment.
"""

from __future__ import annotations

import itertools
import json
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "gold" / "pilot_consumption_table.csv"
OUTPUT_DIR = ROOT / "artifacts" / "baseline"
TARGET = "target_log_likes"
ALPHAS = (0.1, 1.0, 10.0, 100.0)
RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 10_000
PERMUTATION_REPEATS = 100
ROLLING_ORIGIN_FRACTIONS = (0.50, 0.65, 0.80)

BASE_NUMERIC_FEATURES = [
    "duration_seconds",
    "channel_prior_posts",
    "baseline_log_likes",
    "days_since_previous_post",
    "title_length",
    "description_length",
    "tag_count",
]
BASE_CATEGORICAL_FEATURES = [
    "channel_id",
    "sector",
    "channel_type",
    "publish_hour_utc",
    "publish_weekday",
]
TEXT_METADATA_NUMERIC_FEATURES = [
    "title_word_count", "description_word_count", "has_question", "exclamation_count",
    "emoji_count", "hashtag_count", "mention_count", "url_count", "digit_count",
    "uppercase_ratio", "has_cta", "title_tag_overlap", "publish_hour_sin",
    "publish_hour_cos", "publish_weekday_sin", "publish_weekday_cos", "is_weekend",
]
TEXT_METADATA_CATEGORICAL_FEATURES = ["duration_bucket"]
SELECTED_TEXT_NUMERIC_FEATURES = ["has_cta", "title_tag_overlap"]
SELECTED_TEXT_CATEGORICAL_FEATURES = ["duration_bucket"]
HISTORICAL_NUMERIC_FEATURES = [
    "channel_prior_mean_likes", "channel_prior_like_std", "channel_recent_5_median_likes",
    "channel_recent_10_median_likes", "channel_recent_5_like_trend",
    "channel_recent_5_like_volatility", "channel_prior_median_views",
    "channel_prior_median_comments", "channel_prior_like_view_rate",
]
SELECTED_HISTORY_NUMERIC_FEATURES = [
    "channel_recent_10_median_likes",
    "channel_recent_5_like_volatility",
]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES.copy()
CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES.copy()
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class RidgeTabularRegressor:
    """A compact one-hot plus standardized-numeric Ridge regressor."""

    alpha: float
    categorical_features: list[str]
    numeric_features: list[str]

    def fit(self, data: pd.DataFrame, target: pd.Series) -> "RidgeTabularRegressor":
        self.numeric_means_ = data[self.numeric_features].mean()
        self.numeric_scales_ = data[self.numeric_features].std().replace(0, 1).fillna(1)
        self.categories_ = {
            column: sorted(data[column].fillna("<missing>").astype(str).unique())
            for column in self.categorical_features
        }
        matrix = self._transform(data)
        penalty = np.eye(matrix.shape[1])
        penalty[0, 0] = 0.0  # Do not penalize the intercept.
        self.coefficients_ = np.linalg.solve(
            matrix.T @ matrix + self.alpha * penalty,
            matrix.T @ target.to_numpy(dtype=float),
        )
        return self

    def _transform(self, data: pd.DataFrame) -> np.ndarray:
        numeric = (
            data[self.numeric_features].fillna(self.numeric_means_)
            .sub(self.numeric_means_)
            .div(self.numeric_scales_)
            .to_numpy(dtype=float)
        )
        parts = [np.ones((len(data), 1)), numeric]
        for column in self.categorical_features:
            values = data[column].fillna("<missing>").astype(str).to_numpy()
            categories = self.categories_[column]
            parts.append(np.column_stack([(values == value).astype(float) for value in categories]))
        return np.hstack(parts)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self._transform(data) @ self.coefficients_


def validation_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make a second chronological holdout from training data only."""
    development_indexes: list[int] = []
    validation_indexes: list[int] = []
    for _, channel in data.sort_values(["channel_id", "published_at"]).groupby("channel_id"):
        validation_size = max(1, int(np.ceil(len(channel) * 0.20)))
        development_indexes.extend(channel.index[:-validation_size])
        validation_indexes.extend(channel.index[-validation_size:])
    return data.loc[development_indexes].copy(), data.loc[validation_indexes].copy()


def mae_by_channel(data: pd.DataFrame, predictions: np.ndarray) -> float:
    frame = data[["channel_id", TARGET]].copy()
    frame["prediction"] = predictions
    per_channel_mae = frame.groupby("channel_id").apply(
        lambda group: np.mean(np.abs(group[TARGET] - group["prediction"])),
        include_groups=False,
    )
    return float(per_channel_mae.mean())


def select_alpha(
    train: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[float, dict[str, float]]:
    development, validation = validation_split(train)
    scores: dict[str, float] = {}
    for alpha in ALPHAS:
        model = RidgeTabularRegressor(alpha, categorical_features, NUMERIC_FEATURES).fit(
            development, development[TARGET]
        )
        scores[str(alpha)] = mae_by_channel(validation, model.predict(validation))
    selected = min(ALPHAS, key=lambda alpha: scores[str(alpha)])
    return selected, scores


def pairwise_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    left, right = np.triu_indices(len(actual), k=1)
    actual_order = np.sign(actual[left] - actual[right])
    predicted_order = np.sign(predicted[left] - predicted[right])
    eligible = actual_order != 0
    if not eligible.any():
        return float("nan")
    return float((actual_order[eligible] == predicted_order[eligible]).mean())


def evaluate(name: str, test: pd.DataFrame, predictions: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = test[["video_id", "channel_id", "channel_name", "channel_type", "sector", "published_at", TARGET]].copy()
    result["model"] = name
    result["prediction_log_likes"] = predictions
    result["absolute_error"] = np.abs(result[TARGET] - result["prediction_log_likes"])
    metrics: list[dict[str, object]] = []
    for channel_name, channel in result.groupby("channel_name"):
        actual = channel[TARGET].to_numpy()
        predicted = channel["prediction_log_likes"].to_numpy()
        absolute_error = np.abs(actual - predicted)
        total_variation = np.sum((actual - actual.mean()) ** 2)
        metrics.append(
            {
                "model": name,
                "channel_name": channel_name,
                "channel_type": channel["channel_type"].iat[0],
                "test_rows": len(channel),
                "mae_log_likes": np.mean(absolute_error),
                "rmse_log_likes": np.sqrt(np.mean((actual - predicted) ** 2)),
                "bias_log_likes": np.mean(predicted - actual),
                "r2_log_likes": 1 - np.sum((actual - predicted) ** 2) / total_variation
                if total_variation > 0 else np.nan,
                "p50_absolute_error": np.quantile(absolute_error, 0.50),
                "p90_absolute_error": np.quantile(absolute_error, 0.90),
                "p95_absolute_error": np.quantile(absolute_error, 0.95),
                "pairwise_accuracy": pairwise_accuracy(actual, predicted),
            }
        )
    return result, pd.DataFrame(metrics)


def summarize_train_test_metrics(per_channel: pd.DataFrame) -> pd.DataFrame:
    """Macro-average diagnostics so channels with more Shorts do not dominate."""
    summary = (
        per_channel.groupby(["evaluation_split", "model"], as_index=False)
        .agg(
            channels=("channel_name", "nunique"),
            rows=("test_rows", "sum"),
            macro_mae_log_likes=("mae_log_likes", "mean"),
            macro_rmse_log_likes=("rmse_log_likes", "mean"),
            macro_bias_log_likes=("bias_log_likes", "mean"),
            macro_r2_log_likes=("r2_log_likes", "mean"),
            macro_p50_absolute_error=("p50_absolute_error", "mean"),
            macro_p90_absolute_error=("p90_absolute_error", "mean"),
            macro_p95_absolute_error=("p95_absolute_error", "mean"),
            macro_pairwise_accuracy=("pairwise_accuracy", "mean"),
        )
    )
    baseline_mae = summary.loc[summary["model"].eq("B0")].set_index("evaluation_split")["macro_mae_log_likes"]
    summary["delta_mae_vs_b0"] = summary["macro_mae_log_likes"] - summary["evaluation_split"].map(baseline_mae)
    return summary.sort_values(["evaluation_split", "macro_mae_log_likes"])


def paired_comparison(
    per_channel: pd.DataFrame, candidate: str, reference: str
) -> dict[str, float | str]:
    paired = per_channel.pivot(index="channel_name", columns="model", values="mae_log_likes")
    difference = (paired[candidate] - paired[reference]).to_numpy()
    observed = float(difference.mean())
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap = rng.choice(difference, size=(BOOTSTRAP_SAMPLES, len(difference)), replace=True).mean(axis=1)
    # Exact sign permutations grow as 2^channels.  Keep the exact test for
    # small pilots and use a reproducible Monte Carlo approximation otherwise.
    if len(difference) <= 16:
        signs = np.array(list(itertools.product((-1, 1), repeat=len(difference))))
        permutation_method = "exact"
    else:
        signs = rng.choice((-1, 1), size=(BOOTSTRAP_SAMPLES, len(difference)))
        permutation_method = f"monte_carlo_{BOOTSTRAP_SAMPLES}"
    permutation_means = (signs * difference).mean(axis=1)
    return {
        "comparison": f"{candidate} - {reference}",
        "macro_mae_difference": observed,
        "bootstrap_ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_95_high": float(np.quantile(bootstrap, 0.975)),
        "paired_permutation_p_value": float((np.abs(permutation_means) >= abs(observed)).mean()),
        "permutation_method": permutation_method,
    }


def permutation_importance(
    model: RidgeTabularRegressor, test: pd.DataFrame
) -> pd.DataFrame:
    """Estimate M1 feature reliance as degradation in macro MAE after shuffling."""
    baseline_score = mae_by_channel(test, model.predict(test))
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    for feature in MODEL_FEATURES:
        degradations = []
        for _ in range(PERMUTATION_REPEATS):
            permuted = test.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy())
            degradations.append(mae_by_channel(permuted, model.predict(permuted)) - baseline_score)
        rows.append(
            {
                "feature": feature,
                "baseline_macro_mae": baseline_score,
                "mean_mae_increase": float(np.mean(degradations)),
                "ci_95_low": float(np.quantile(degradations, 0.025)),
                "ci_95_high": float(np.quantile(degradations, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mae_increase", ascending=False)


def save_diagnostic_figures(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    per_channel: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    """Save model-diagnostic graphics for the frozen M1 test predictions."""
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    ordered = summary.sort_values("macro_mae_log_likes")
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(ordered["model"], ordered["macro_mae_log_likes"], color="#386cb0")
    axis.set(title="Erro absoluto medio por modelo", xlabel="Modelo", ylabel="MAE macro em log(1 + curtidas)")
    axis.bar_label(axis.containers[0], fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(figures_dir / "01_mae_por_modelo.png", dpi=160)
    plt.close(fig)

    shown = importance.sort_values("mean_mae_increase").tail(12)
    error_low = shown["mean_mae_increase"] - shown["ci_95_low"]
    error_high = shown["ci_95_high"] - shown["mean_mae_increase"]
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(shown["feature"], shown["mean_mae_increase"], xerr=[error_low, error_high], color="#7fc97f")
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set(title="Importancia por permutacao (M1)", xlabel="Aumento do MAE macro apos embaralhar", ylabel="Variavel")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_importancia_por_permutacao_m1.png", dpi=160)
    plt.close(fig)

    m1 = predictions.loc[predictions["model"].eq("M1")].copy()
    fig, axis = plt.subplots(figsize=(6, 6))
    colors = {"brand": "#386cb0", "creator": "#fdc086"}
    for channel_type, group in m1.groupby("channel_type"):
        axis.scatter(group[TARGET], group["prediction_log_likes"], s=24, alpha=0.65, label=channel_type, color=colors[channel_type])
    low = min(m1[TARGET].min(), m1["prediction_log_likes"].min())
    high = max(m1[TARGET].max(), m1["prediction_log_likes"].max())
    axis.plot([low, high], [low, high], color="#333333", linewidth=1, linestyle="--")
    axis.set(title="Previsao versus resultado (M1)", xlabel="Resultado: log(1 + curtidas)", ylabel="Previsao: log(1 + curtidas)")
    axis.legend(title="Tipo de canal")
    fig.tight_layout()
    fig.savefig(figures_dir / "03_previsao_versus_resultado_m1.png", dpi=160)
    plt.close(fig)

    channel_errors = (
        per_channel.loc[per_channel["model"].isin(["B0", "M1"])]
        .pivot(index="channel_name", columns="model", values="mae_log_likes")
        .sort_values("M1")
    )
    positions = np.arange(len(channel_errors))
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.hlines(positions, channel_errors["B0"], channel_errors["M1"], color="#bdbdbd", linewidth=2)
    axis.scatter(channel_errors["B0"], positions, label="B0", color="#666666")
    axis.scatter(channel_errors["M1"], positions, label="M1", color="#386cb0")
    axis.set(yticks=positions, yticklabels=channel_errors.index, title="Erro por canal: M1 versus baseline", xlabel="MAE em log(1 + curtidas)", ylabel="Canal")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "04_erro_por_canal_m1_vs_b0.png", dpi=160)
    plt.close(fig)

    m1["residual"] = m1["prediction_log_likes"] - m1[TARGET]
    m1 = m1.sort_values("published_at")
    fig, axis = plt.subplots(figsize=(10, 4))
    for channel_type, group in m1.groupby("channel_type"):
        axis.scatter(group["published_at"], group["residual"], s=20, alpha=0.65, label=channel_type, color=colors[channel_type])
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set(title="Residuos de M1 ao longo dos videos de teste", xlabel="Data de publicacao", ylabel="Previsao - resultado em log(1 + curtidas)")
    axis.legend(title="Tipo de canal")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures_dir / "05_residuos_por_video_no_tempo_m1.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    groups = [m1.loc[m1["channel_type"].eq(kind), "absolute_error"] for kind in ("brand", "creator")]
    axis.boxplot(groups, tick_labels=["Marcas", "Criadores"], showfliers=False)
    axis.set(title="Distribuicao do erro absoluto de M1", xlabel="Tipo de canal", ylabel="Erro absoluto em log(1 + curtidas)")
    fig.tight_layout()
    fig.savefig(figures_dir / "06_distribuicao_de_erros_por_tipo_m1.png", dpi=160)
    plt.close(fig)


def comparison_statistics(per_channel: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    return {
        "m1_vs_b0": paired_comparison(per_channel, "M1", "B0"),
        "m2_vs_m1": paired_comparison(per_channel, "M2", "M1"),
        "m1_vs_m3": paired_comparison(per_channel, "M1", "M3"),
    }


def rolling_origin_splits(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Create expanding training windows and following validation blocks per channel."""
    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for position, train_fraction in enumerate(ROLLING_ORIGIN_FRACTIONS):
        next_fraction = (
            ROLLING_ORIGIN_FRACTIONS[position + 1]
            if position + 1 < len(ROLLING_ORIGIN_FRACTIONS)
            else 1.0
        )
        fit_indexes: list[int] = []
        validation_indexes: list[int] = []
        for _, channel in data.sort_values(["channel_id", "published_at", "video_id"]).groupby("channel_id"):
            train_end = max(5, int(np.floor(len(channel) * train_fraction)))
            validation_end = max(train_end + 1, int(np.floor(len(channel) * next_fraction)))
            validation_end = min(validation_end, len(channel))
            fit_indexes.extend(channel.index[:train_end])
            validation_indexes.extend(channel.index[train_end:validation_end])
        folds.append(
            (
                f"origin_{position + 1}",
                data.loc[fit_indexes].copy(),
                data.loc[validation_indexes].copy(),
            )
        )
    return folds


def rolling_origin_evaluation(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float | str]], pd.DataFrame]:
    """Evaluate every baseline over historical expanding windows without using final test rows."""
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    definitions: list[dict[str, object]] = []
    categorical_without_type = [column for column in CATEGORICAL_FEATURES if column != "channel_type"]

    for origin, fit_data, validation_data in rolling_origin_splits(train):
        predictions: dict[str, np.ndarray] = {"B0": validation_data["baseline_log_likes"].to_numpy()}

        alpha_m1, _ = select_alpha(fit_data, CATEGORICAL_FEATURES)
        m1 = RidgeTabularRegressor(alpha_m1, CATEGORICAL_FEATURES, NUMERIC_FEATURES).fit(
            fit_data, fit_data[TARGET]
        )
        predictions["M1"] = m1.predict(validation_data)

        alpha_m3, _ = select_alpha(fit_data, categorical_without_type)
        m3 = RidgeTabularRegressor(alpha_m3, categorical_without_type, NUMERIC_FEATURES).fit(
            fit_data, fit_data[TARGET]
        )
        predictions["M3"] = m3.predict(validation_data)

        m2_predictions = np.empty(len(validation_data))
        for channel_type in sorted(fit_data["channel_type"].unique()):
            subgroup_fit = fit_data.loc[fit_data["channel_type"].eq(channel_type)]
            subgroup_validation = validation_data.loc[validation_data["channel_type"].eq(channel_type)]
            alpha, _ = select_alpha(subgroup_fit, categorical_without_type)
            model = RidgeTabularRegressor(alpha, categorical_without_type, NUMERIC_FEATURES).fit(
                subgroup_fit, subgroup_fit[TARGET]
            )
            m2_predictions[validation_data["channel_type"].eq(channel_type).to_numpy()] = model.predict(
                subgroup_validation
            )
        predictions["M2"] = m2_predictions

        for name, values in predictions.items():
            prediction_frame, metric_frame = evaluate(name, validation_data, values)
            prediction_frame["origin"] = origin
            metric_frame["origin"] = origin
            all_predictions.append(prediction_frame)
            all_metrics.append(metric_frame)
        temporal_order_valid = all(
            fit_data.loc[fit_data["channel_id"].eq(channel_id), "published_at"].max()
            < validation_data.loc[validation_data["channel_id"].eq(channel_id), "published_at"].min()
            for channel_id in fit_data["channel_id"].unique()
        )
        if not temporal_order_valid:
            raise AssertionError(f"Temporal ordering failed for {origin}.")
        definitions.append(
            {
                "origin": origin,
                "train_rows": len(fit_data),
                "validation_rows": len(validation_data),
                "train_max_published_at": fit_data["published_at"].max(),
                "validation_min_published_at": validation_data["published_at"].min(),
                "temporal_order_valid_per_channel": temporal_order_valid,
            }
        )

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    by_origin = (
        metrics.groupby(["origin", "model"], as_index=False)
        .agg(
            macro_mae_log_likes=("mae_log_likes", "mean"),
            macro_rmse_log_likes=("rmse_log_likes", "mean"),
            macro_bias_log_likes=("bias_log_likes", "mean"),
            macro_pairwise_accuracy=("pairwise_accuracy", "mean"),
        )
    )
    summary = (
        by_origin.groupby("model", as_index=False)
        .agg(
            rolling_macro_mae_log_likes=("macro_mae_log_likes", "mean"),
            rolling_macro_rmse_log_likes=("macro_rmse_log_likes", "mean"),
            rolling_macro_bias_log_likes=("macro_bias_log_likes", "mean"),
            rolling_macro_pairwise_accuracy=("macro_pairwise_accuracy", "mean"),
        )
        .sort_values("rolling_macro_mae_log_likes")
    )
    channel_average = (
        metrics.groupby(["model", "channel_name", "channel_type"], as_index=False)
        .agg(
            mae_log_likes=("mae_log_likes", "mean"),
            rmse_log_likes=("rmse_log_likes", "mean"),
            bias_log_likes=("bias_log_likes", "mean"),
            pairwise_accuracy=("pairwise_accuracy", "mean"),
        )
    )
    return predictions, metrics, summary, comparison_statistics(channel_average), pd.DataFrame(definitions)


def save_rolling_origin_figure(by_origin: pd.DataFrame) -> None:
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    for model, group in by_origin.groupby("model"):
        group = group.sort_values("origin")
        axis.plot(group["origin"], group["macro_mae_log_likes"], marker="o", label=model)
    axis.set(title="MAE macro ao longo dos cortes rolling-origin", xlabel="Corte temporal", ylabel="MAE macro em log(1 + curtidas)")
    axis.legend(title="Modelo")
    fig.tight_layout()
    fig.savefig(figures_dir / "07_rolling_origin_mae_por_modelo.png", dpi=160)
    plt.close(fig)


def report(
    summary: pd.DataFrame,
    statistics: dict[str, dict[str, float | str]],
    selections: dict[str, object],
) -> str:
    rows = "\n".join(
        f"| {row.model} | {row.macro_mae_log_likes:.4f} | {row.macro_rmse_log_likes:.4f} | "
        f"{row.macro_bias_log_likes:.4f} | {row.macro_pairwise_accuracy:.1%} |"
        for row in summary.itertuples(index=False)
    )
    return f"""# Resultado do baseline tabular

## Protocolo

- Fonte: `data/gold/pilot_consumption_table.csv`.
- Alvo: `target_log_likes`.
- Selecao de regularizacao: corte temporal interno nas linhas de treino.
- Avaliacao final: 375 posts mais recentes por canal, sem uso para selecao.
- Metrica primaria: MAE medio entre canais.

## Resultado no teste

| Modelo | MAE macro | RMSE macro | Vies macro | Ranking por pares |
| --- | ---: | ---: | ---: | ---: |
{rows}

## Comparacoes pareadas por canal

| Comparacao | Diferenca de MAE macro | IC bootstrap 95% | p-valor de permutacao |
| --- | ---: | ---: | ---: |
| M1 - B0 | {statistics['m1_vs_b0']['macro_mae_difference']:.4f} | [{statistics['m1_vs_b0']['bootstrap_ci_95_low']:.4f}, {statistics['m1_vs_b0']['bootstrap_ci_95_high']:.4f}] | {statistics['m1_vs_b0']['paired_permutation_p_value']:.4f} |
| M2 - M1 | {statistics['m2_vs_m1']['macro_mae_difference']:.4f} | [{statistics['m2_vs_m1']['bootstrap_ci_95_low']:.4f}, {statistics['m2_vs_m1']['bootstrap_ci_95_high']:.4f}] | {statistics['m2_vs_m1']['paired_permutation_p_value']:.4f} |
| M1 - M3 | {statistics['m1_vs_m3']['macro_mae_difference']:.4f} | [{statistics['m1_vs_m3']['bootstrap_ci_95_low']:.4f}, {statistics['m1_vs_m3']['bootstrap_ci_95_high']:.4f}] | {statistics['m1_vs_m3']['paired_permutation_p_value']:.4f} |

Valores negativos favorecem o primeiro modelo da comparacao. Como existem apenas 12 canais, os intervalos e o teste de permutacao devem orientar a decisao junto da magnitude pratica do ganho.

## Hiperparametros selecionados

```json
{json.dumps(selections, ensure_ascii=False, indent=2)}
```

O resultado deve ser interpretado junto de `per_channel_metrics.csv`, que mostra se um ganho agregado esconde piora para marcas ou criadores.
"""


def main() -> None:
    global OUTPUT_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-set",
        choices=("baseline", "text_metadata", "selected_text", "historical", "selected_history"),
        default="baseline",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    NUMERIC_FEATURES = BASE_NUMERIC_FEATURES.copy()
    CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES.copy()
    if args.feature_set == "text_metadata":
        NUMERIC_FEATURES += TEXT_METADATA_NUMERIC_FEATURES
        CATEGORICAL_FEATURES += TEXT_METADATA_CATEGORICAL_FEATURES
    if args.feature_set == "selected_text":
        NUMERIC_FEATURES += SELECTED_TEXT_NUMERIC_FEATURES
        CATEGORICAL_FEATURES += SELECTED_TEXT_CATEGORICAL_FEATURES
    if args.feature_set == "historical":
        NUMERIC_FEATURES += HISTORICAL_NUMERIC_FEATURES
    if args.feature_set == "selected_history":
        NUMERIC_FEATURES += SELECTED_HISTORY_NUMERIC_FEATURES
    MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    data = pd.read_csv(INPUT_PATH, parse_dates=["published_at"])
    train = data.loc[data["split"].eq("train")].copy()
    test = data.loc[data["split"].eq("test")].copy()
    if train.empty or test.empty:
        raise ValueError("Expected both train and test rows in the consumption table.")

    predictions: dict[str, np.ndarray] = {"B0": test["baseline_log_likes"].to_numpy()}
    train_predictions: dict[str, np.ndarray] = {"B0": train["baseline_log_likes"].to_numpy()}
    selections: dict[str, object] = {}

    alpha_m1, scores_m1 = select_alpha(train, CATEGORICAL_FEATURES)
    m1 = RidgeTabularRegressor(alpha_m1, CATEGORICAL_FEATURES, NUMERIC_FEATURES).fit(train, train[TARGET])
    predictions["M1"] = m1.predict(test)
    train_predictions["M1"] = m1.predict(train)
    selections["M1"] = {"alpha": alpha_m1, "validation_macro_mae": scores_m1}

    categorical_without_type = [column for column in CATEGORICAL_FEATURES if column != "channel_type"]
    alpha_m3, scores_m3 = select_alpha(train, categorical_without_type)
    m3 = RidgeTabularRegressor(alpha_m3, categorical_without_type, NUMERIC_FEATURES).fit(train, train[TARGET])
    predictions["M3"] = m3.predict(test)
    train_predictions["M3"] = m3.predict(train)
    selections["M3"] = {"alpha": alpha_m3, "validation_macro_mae": scores_m3}

    m2_predictions = np.empty(len(test))
    m2_train_predictions = np.empty(len(train))
    selections["M2"] = {}
    for channel_type in sorted(train["channel_type"].unique()):
        subgroup_train = train.loc[train["channel_type"].eq(channel_type)]
        subgroup_test = test.loc[test["channel_type"].eq(channel_type)]
        alpha, scores = select_alpha(subgroup_train, categorical_without_type)
        model = RidgeTabularRegressor(alpha, categorical_without_type, NUMERIC_FEATURES).fit(
            subgroup_train, subgroup_train[TARGET]
        )
        m2_predictions[test["channel_type"].eq(channel_type).to_numpy()] = model.predict(subgroup_test)
        m2_train_predictions[train["channel_type"].eq(channel_type).to_numpy()] = model.predict(subgroup_train)
        selections["M2"][channel_type] = {"alpha": alpha, "validation_macro_mae": scores}
    predictions["M2"] = m2_predictions
    train_predictions["M2"] = m2_train_predictions

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    for name, values in predictions.items():
        prediction_frame, metric_frame = evaluate(name, test, values)
        all_predictions.append(prediction_frame)
        all_metrics.append(metric_frame)
    prediction_output = pd.concat(all_predictions, ignore_index=True)
    per_channel = pd.concat(all_metrics, ignore_index=True)
    train_prediction_frames: list[pd.DataFrame] = []
    train_metric_frames: list[pd.DataFrame] = []
    for name, values in train_predictions.items():
        prediction_frame, metric_frame = evaluate(name, train, values)
        prediction_frame["evaluation_split"] = "train"
        metric_frame["evaluation_split"] = "train"
        train_prediction_frames.append(prediction_frame)
        train_metric_frames.append(metric_frame)
    test_prediction_output = prediction_output.assign(evaluation_split="test")
    test_per_channel = per_channel.assign(evaluation_split="test")
    train_test_predictions = pd.concat([*train_prediction_frames, test_prediction_output], ignore_index=True)
    train_test_per_channel = pd.concat([*train_metric_frames, test_per_channel], ignore_index=True)
    train_test_summary = summarize_train_test_metrics(train_test_per_channel)
    summary = (
        per_channel.groupby("model", as_index=False)
        .agg(
            macro_mae_log_likes=("mae_log_likes", "mean"),
            macro_rmse_log_likes=("rmse_log_likes", "mean"),
            macro_bias_log_likes=("bias_log_likes", "mean"),
            macro_pairwise_accuracy=("pairwise_accuracy", "mean"),
        )
        .sort_values("macro_mae_log_likes")
    )
    statistics = comparison_statistics(per_channel)
    importance = permutation_importance(m1, test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    per_channel.to_csv(OUTPUT_DIR / "per_channel_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    train_test_predictions.to_csv(OUTPUT_DIR / "train_test_predictions.csv", index=False)
    train_test_per_channel.to_csv(OUTPUT_DIR / "train_test_per_channel_metrics.csv", index=False)
    train_test_summary.to_csv(OUTPUT_DIR / "train_test_metrics_summary.csv", index=False)
    importance.to_csv(OUTPUT_DIR / "permutation_importance_m1.csv", index=False)
    (OUTPUT_DIR / "comparison_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "baseline_report.md").write_text(
        report(summary, statistics, selections), encoding="utf-8"
    )
    save_diagnostic_figures(summary, prediction_output, per_channel, importance)

    (
        rolling_predictions,
        rolling_metrics,
        rolling_summary,
        rolling_statistics,
        rolling_definitions,
    ) = rolling_origin_evaluation(train)
    rolling_by_origin = (
        rolling_metrics.groupby(["origin", "model"], as_index=False)
        .agg(
            macro_mae_log_likes=("mae_log_likes", "mean"),
            macro_rmse_log_likes=("rmse_log_likes", "mean"),
            macro_bias_log_likes=("bias_log_likes", "mean"),
            macro_pairwise_accuracy=("pairwise_accuracy", "mean"),
        )
    )
    rolling_predictions.to_csv(OUTPUT_DIR / "rolling_origin_predictions.csv", index=False)
    rolling_metrics.to_csv(OUTPUT_DIR / "rolling_origin_per_channel_metrics.csv", index=False)
    rolling_by_origin.to_csv(OUTPUT_DIR / "rolling_origin_by_cut.csv", index=False)
    rolling_summary.to_csv(OUTPUT_DIR / "rolling_origin_summary.csv", index=False)
    rolling_definitions.to_csv(OUTPUT_DIR / "rolling_origin_definitions.csv", index=False)
    (OUTPUT_DIR / "rolling_origin_comparison_statistics.json").write_text(
        json.dumps(rolling_statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rolling_rows = "\n".join(
        f"| {row.model} | {row.rolling_macro_mae_log_likes:.4f} | "
        f"{row.rolling_macro_rmse_log_likes:.4f} | {row.rolling_macro_bias_log_likes:.4f} | "
        f"{row.rolling_macro_pairwise_accuracy:.1%} |"
        for row in rolling_summary.itertuples(index=False)
    )
    (OUTPUT_DIR / "rolling_origin_report.md").write_text(
        "# Resultado rolling-origin\n\n"
        "A avaliacao usa tres janelas expansivas dentro das linhas de treino; o teste final permanece intocado. "
        "A metrica primaria e o MAE medio por canal.\n\n"
        "| Modelo | MAE macro | RMSE macro | Vies macro | Ranking por pares |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        + rolling_rows
        + "\n\n## Comparacoes pareadas por canal\n\n```json\n"
        + json.dumps(rolling_statistics, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    save_rolling_origin_figure(rolling_by_origin)

    print(summary.to_string(index=False))
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    print("\nRolling-origin summary:")
    print(rolling_summary.to_string(index=False))
    print(f"Artifacts saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
