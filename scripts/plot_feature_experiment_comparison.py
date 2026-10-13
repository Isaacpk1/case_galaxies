"""Create comparison charts for the baseline (E-00) and text/metadata batch (E-01)."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "artifacts" / "feature_experiments"
OUTPUT = EXPERIMENTS / "comparison_figures"
MODELS = ["B0", "M1", "M2", "M3"]
COLORS = {"E-00": "#666666", "E-01": "#386cb0"}


def load_summary(experiment: str, filename: str, metric: str) -> pd.DataFrame:
    data = pd.read_csv(EXPERIMENTS / experiment / filename)
    return data[["model", metric]].assign(experiment=experiment.upper().replace("E", "E-"))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    final = pd.concat([
        load_summary("e00", "metrics_summary.csv", "macro_mae_log_likes"),
        load_summary("e01", "metrics_summary.csv", "macro_mae_log_likes"),
    ])
    rolling = pd.concat([
        load_summary("e00", "rolling_origin_summary.csv", "rolling_macro_mae_log_likes"),
        load_summary("e01", "rolling_origin_summary.csv", "rolling_macro_mae_log_likes"),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for axis, data, metric, title in [
        (axes[0], final, "macro_mae_log_likes", "Teste final (referencia; nao usado para escolher features)"),
        (axes[1], rolling, "rolling_macro_mae_log_likes", "Rolling-origin (metrica de selecao)"),
    ]:
        pivot = data.pivot(index="model", columns="experiment", values=metric).reindex(MODELS)
        pivot.plot.bar(ax=axis, color=[COLORS["E-00"], COLORS["E-01"]], rot=0)
        axis.set(title=title, xlabel="Modelo", ylabel="MAE macro em log(1 + curtidas)")
        axis.legend(title="Features")
    fig.tight_layout()
    fig.savefig(OUTPUT / "01_mae_e00_vs_e01.png", dpi=160)
    plt.close(fig)

    comparison = rolling.pivot(index="model", columns="experiment", values="rolling_macro_mae_log_likes").reindex(MODELS)
    comparison["delta_e01_minus_e00"] = comparison["E-01"] - comparison["E-00"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(comparison.index, comparison["delta_e01_minus_e00"], color="#d95f02")
    axis.axhline(0, color="#333333", linewidth=1)
    axis.set(title="Mudanca no MAE rolling-origin apos E-01", xlabel="Modelo", ylabel="MAE E-01 - MAE E-00")
    axis.bar_label(axis.containers[0], fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "02_delta_rolling_por_modelo.png", dpi=160)
    plt.close(fig)

    e00 = pd.read_csv(EXPERIMENTS / "e00" / "rolling_origin_per_channel_metrics.csv")
    e01 = pd.read_csv(EXPERIMENTS / "e01" / "rolling_origin_per_channel_metrics.csv")
    e00_m1 = e00.loc[e00["model"].eq("M1")].groupby("channel_name", as_index=False)["mae_log_likes"].mean()
    e01_m1 = e01.loc[e01["model"].eq("M1")].groupby("channel_name", as_index=False)["mae_log_likes"].mean()
    channels = e00_m1.merge(e01_m1, on="channel_name", suffixes=("_e00", "_e01"))
    channels["delta"] = channels["mae_log_likes_e01"] - channels["mae_log_likes_e00"]
    channels = channels.sort_values("delta")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(channels["channel_name"], channels["delta"], color="#d95f02")
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set(title="Efeito do lote E-01 no erro de M1 por canal", xlabel="MAE rolling E-01 - E-00", ylabel="Canal")
    fig.tight_layout()
    fig.savefig(OUTPUT / "03_delta_m1_por_canal.png", dpi=160)
    plt.close(fig)

    importance = pd.read_csv(EXPERIMENTS / "e01" / "permutation_importance_m1.csv").head(15).sort_values("mean_mae_increase")
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.barh(importance["feature"], importance["mean_mae_increase"], color="#7fc97f")
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set(title="Importancia por permutacao: E-01, M1", xlabel="Aumento do MAE apos embaralhar", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(OUTPUT / "04_importancia_e01_m1.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
