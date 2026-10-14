"""Treina e empacota o modelo servido pela tool (agente preditivo / MCP).

Modelo final = CB_FINAL do E-13 (aprovado pela regra pre-registrada e confirmado no teste final):
CatBoost, alvo residual sobre o B0, perda MAE, hiperparametros do E-08, com
- embedding SigLIP 2 da thumbnail (PCA 16),
- comparaveis visuais (5 Shorts anteriores do canal com thumbnail mais parecida),
- rubrica do video + fala extraidas pelo Gemini.

Modelos de reserva, usados quando falta uma entrada (a tool marca confianca baixa):
- CB_EMB (sem rubrica: criativo que nao e um Short do YouTube, sem chave do Gemini ou Gemini fora);
- CB_BASE (sem imagem).

Escopo do treino: por padrao TODO o Gold (treino + teste), porque o teste final ja foi aberto uma vez
(E-13) e o modelo servido deve usar os Shorts mais recentes. `--train-only` reproduz exatamente o
modelo avaliado no teste. Intervalo de 80%: quantis conformais dos residuos fora da amostra do
rolling-origin (E-13 para o CB_FINAL; E-08 para as reservas), por tipo de canal.

Saida imutavel: artifacts/tool_model/<model_version>/ e artifacts/tool_model/LATEST.json.

    python scripts/modelo/train_tool_model.py               # treina em todo o Gold e empacota
    python scripts/modelo/train_tool_model.py --train-only  # so split=train (modelo do teste final)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from galaxies_agent.features import GEMINI_CAT, GEMINI_NUM, KNN_COLS, knn_features, rubric_features  # noqa: E402

OUT_BASE = ROOT / "artifacts" / "tool_model"
E08 = ROOT / "artifacts" / "feature_experiments" / "e08_catboost_residual"
E13 = ROOT / "artifacts" / "feature_experiments" / "e13_final"
EMB_PATH = ROOT / "data" / "silver" / "features" / "thumbnail_siglip2_v1" / "20261013T010000Z" / "embeddings.npz"
RUBRIC_PATH = ROOT / "data" / "silver" / "gemini_video" / "gemini_video_features.csv"
SILVER_RUN = "20261002T013000Z"

TARGET = "target_log_likes"
BASELINE = "baseline_log_likes"
NUMERIC = ["duration_seconds", "channel_prior_posts", "baseline_log_likes", "days_since_previous_post",
           "title_length", "description_length", "tag_count"]
CATEGORICAL = ["channel_id", "sector", "channel_type", "publish_hour_utc", "publish_weekday"]
PCA_DIM = 16
EMB_COLS = [f"emb_pc{i + 1:02d}" for i in range(PCA_DIM)]
SEED = 42
INTERVAL = (0.10, 0.90)

MODELS = {  # nome: (numericas, categoricas, experimento de referencia)
    "final": (NUMERIC + EMB_COLS + KNN_COLS + GEMINI_NUM, CATEGORICAL + GEMINI_CAT, "E-13 CB_FINAL"),
    "with_embedding": (NUMERIC + EMB_COLS, CATEGORICAL, "E-08 CB_EMB"),
    "without_embedding": (NUMERIC, CATEGORICAL, "E-08 CB_BASE"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gold_path() -> Path:
    latest = json.loads((ROOT / "data" / "gold" / "snapshots" / "LATEST.json").read_text(encoding="utf-8"))
    return ROOT / latest["path"]


def e08_params(arm: str) -> dict:
    """CB_EMB: parametros finais do E-08 (os mesmos usados por todos os bracos do E-13). CB_BASE: origem 3."""
    if arm == "CB_EMB":
        final = json.loads((E08 / "run_info.json").read_text())["final_params"]
        return {k: (int(v) if k in ("depth", "iterations") else v) for k, v in final.items()}
    chosen = pd.read_csv(E08 / "chosen_params.csv")
    row = chosen.query("arm == @arm and origin == 'origin_3'").iloc[0].to_dict()
    params = {k: row[k] for k in ("depth", "learning_rate", "l2_leaf_reg", "random_strength", "bagging_temperature", "iterations")}
    params["depth"], params["iterations"] = int(params["depth"]), int(params["iterations"])
    return params


def fit_pca(matrix: np.ndarray, dim: int) -> dict:
    mean = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    components = vt[:dim]
    signs = np.sign(components[np.arange(dim), np.abs(components).argmax(axis=1)])  # sinal deterministico
    return {"mean": mean.astype(np.float32), "components": (components * signs[:, None]).astype(np.float32)}


def to_pool(frame: pd.DataFrame, numeric: list[str], categorical: list[str], label: bool = True) -> Pool:
    x = frame[numeric + categorical].copy()
    for column in categorical:
        x[column] = x[column].fillna("<missing>").astype(str)
    y = (frame[TARGET] - frame[BASELINE]).to_numpy() if label else None
    return Pool(x, y, cat_features=categorical)


def add_knn(frame: pd.DataFrame, matrix: np.ndarray) -> None:
    """Comparaveis: so Shorts anteriores do mesmo canal (mesma regra do E-12/E-13)."""
    resid = (frame[TARGET] - frame[BASELINE]).to_numpy()
    values = np.full((len(frame), len(KNN_COLS)), np.nan)
    for _, idx in frame.groupby("channel_id").groups.items():
        idx = np.asarray(idx)
        t = frame.loc[idx, "published_at"].to_numpy()
        for j, i in enumerate(idx):
            prior = idx[t < t[j]]
            feats = knn_features(matrix[prior], resid[prior], matrix[i])
            values[i] = [feats[c] for c in KNN_COLS]
    frame[KNN_COLS] = values


def add_rubric(frame: pd.DataFrame) -> float:
    rubric = pd.read_csv(RUBRIC_PATH, dtype={"video_id": str}).set_index("video_id")
    rows = []
    for video_id, duration in zip(frame["video_id"], frame["duration_seconds"]):
        rows.append(rubric_features(rubric.loc[video_id].to_dict() if video_id in rubric.index else None, duration))
    feats = pd.DataFrame(rows, index=frame.index)
    for column in feats.columns:
        frame[column] = feats[column]
    return float(feats["gemini_coverage"].mean())


def residual_quantiles(pred: pd.DataFrame, column: str, source: str) -> dict:
    error = pred[TARGET] - pred[column]
    entry = {"global": [float(q) for q in np.quantile(error, INTERVAL)], "source": source}
    for kind, group in pred.assign(error=error).groupby("channel_type"):
        entry[kind] = [float(q) for q in np.quantile(group["error"], INTERVAL)]
    cal, test = pred[pred["origin"] != "origin_3"], pred[pred["origin"] == "origin_3"]
    lo, hi = np.quantile(cal[TARGET] - cal[column], INTERVAL)
    e3 = test[TARGET] - test[column]
    entry["holdout_check"] = {"calibrated_on": "origin_1+origin_2", "evaluated_on": "origin_3",
                              "coverage": float(((e3 >= lo) & (e3 <= hi)).mean()), "nominal": INTERVAL[1] - INTERVAL[0]}
    return entry


def calibration() -> dict:
    e08 = pd.read_csv(E08 / "rolling_origin_predictions.csv")
    e13 = pd.concat(pd.read_csv(f) for f in sorted((E13 / "tasks").glob("CB_FINAL__origin_*.csv")))
    out = {"final": residual_quantiles(e13, "pred", "E-13 rolling-origin (CB_FINAL)"),
           "with_embedding": residual_quantiles(e08, "pred_CB_EMB", "E-08 rolling-origin (CB_EMB)"),
           "without_embedding": residual_quantiles(e08, "pred_CB_BASE", "E-08 rolling-origin (CB_BASE)")}
    report = E13 / "test_final" / "test_report.json"
    if report.exists():
        out["final"]["final_test_coverage"] = json.loads(report.read_text(encoding="utf-8"))["intervalo_80"]["cobertura_teste"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-only", action="store_true", help="Treina so com split=train (modelo avaliado no teste)")
    args = parser.parse_args()

    gold_file = gold_path()
    gold = pd.read_csv(gold_file, parse_dates=["published_at"])
    data = (gold[gold["split"] == "train"] if args.train_only else gold)
    data = data.sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)

    emb_npz = np.load(EMB_PATH, allow_pickle=True)
    emb_by_id = dict(zip(emb_npz["video_id"].astype(str), emb_npz["embeddings"].astype(np.float32)))
    missing = [v for v in data["video_id"] if v not in emb_by_id]
    if missing:
        raise SystemExit(f"{len(missing)} videos sem embedding")
    matrix = np.vstack([emb_by_id[v] for v in data["video_id"]])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    pca = fit_pca(matrix, PCA_DIM)
    data[EMB_COLS] = (matrix - pca["mean"]) @ pca["components"].T
    add_knn(data, matrix)
    coverage = add_rubric(data)
    print(f"linhas {len(data)} | canais {data['channel_id'].nunique()} | cobertura Gemini {coverage:.2%}", flush=True)

    params = {"final": e08_params("CB_EMB"), "with_embedding": e08_params("CB_EMB"), "without_embedding": e08_params("CB_BASE")}
    version = f"tool_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = OUT_BASE / version
    out.mkdir(parents=True, exist_ok=False)
    features = {}
    for name, (numeric, categorical, ref) in MODELS.items():
        model = CatBoostRegressor(loss_function="MAE", random_seed=SEED, verbose=False, allow_writing_files=False,
                                  thread_count=-1, **params[name]).fit(to_pool(data, numeric, categorical))
        model.save_model(str(out / f"model_{name}.cbm"))
        features[name] = {"numeric": numeric, "categorical": categorical, "reference": ref, "params": params[name]}
        print(f"modelo {name} ok", flush=True)
    np.savez(out / "pca.npz", **pca)

    silver = pd.read_csv(ROOT / "data" / "silver" / "eligible_shorts" / SILVER_RUN / "eligible_shorts.csv",
                         parse_dates=["published_at"])
    history = silver[["video_id", "channel_id", "channel_name", "sector", "channel_type", "published_at",
                      "like_count", "title", "duration_seconds"]].dropna(subset=["like_count"])
    history.to_csv(out / "channel_history.csv", index=False)

    # indice de comparaveis: tambem alimenta as features KNN de criativos novos de canais da base
    titles = silver.set_index("video_id")["title"]
    comp = data[["video_id", "channel_id", "channel_name", "sector", "channel_type", "published_at", TARGET, BASELINE]].copy()
    comp["title"] = comp["video_id"].map(titles).fillna("")
    comp["likes"] = np.expm1(comp[TARGET]).round().astype(int)
    comp["resid"] = comp[TARGET] - comp[BASELINE]
    comp["relative_to_channel"] = np.exp(comp["resid"])
    comp.drop(columns=[TARGET, BASELINE]).to_csv(out / "comparables.csv", index=False)
    np.save(out / "comparables_embeddings.npy", matrix.astype(np.float32))

    channel_stats = (history.groupby("channel_id")
                     .agg(channel_name=("channel_name", "first"), sector=("sector", "first"),
                          channel_type=("channel_type", "first"), posts=("video_id", "count"),
                          median_likes=("like_count", "median"), last_post_at=("published_at", "max"))
                     .reset_index())
    channel_stats["in_training"] = channel_stats["channel_id"].isin(data["channel_id"])
    channel_stats.to_csv(out / "channels.csv", index=False)

    test_report = E13 / "test_final" / "test_report.json"
    test = json.loads(test_report.read_text(encoding="utf-8")) if test_report.exists() else None
    metadata = {
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_experiment": "E-13 (CB_FINAL), aprovado pela regra pre-registrada; reservas CB_EMB e CB_BASE (E-08)",
        "dataset_version": gold_file.parent.name,
        "training_scope": "split=train" if args.train_only else "todo o Gold (treino + teste); teste final ja aberto uma vez no E-13",
        "training_rows": int(len(data)), "training_channels": int(data["channel_id"].nunique()),
        "test_split": ("fechado: modelos treinados so com split=train" if args.train_only else
                       "aberto uma unica vez no E-13 (modelo treinado so com split=train); este pacote foi re-treinado em todo o Gold"),
        "final_test": None if test is None else {
            "rows": test["linhas_teste"], "channels": test["canais"],
            "mae_macro": {k: v["mae_macro_canal"] for k, v in test["metricas"].items()},
            "pair_accuracy": {k: v["pair_fair_macro_canal"] for k, v in test["metricas"].items()},
            "interval_coverage": test["intervalo_80"]["cobertura_teste"]},
        "target": "target_log_likes - baseline_log_likes (desvio sobre a mediana historica do canal), perda MAE",
        "models": features,
        "gemini_coverage_training": round(coverage, 4),
        "embedding": {"model": str(emb_npz["source"]), "dim": int(matrix.shape[1]), "pca_dim": PCA_DIM,
                      "preprocess": "painel vertical 9:16 recortado quando a thumbnail e pillarbox; lado menor 256 px",
                      "normalize": "L2 antes do PCA"},
        "rubric": {"model": "gemini-3.5-flash-lite", "prompt_version": "rubrica_tap_v1", "media_resolution": "MEDIA_RESOLUTION_LOW",
                   "source": "URL publica do YouTube (video nao e baixado)"},
        "interval": {"nominal": INTERVAL[1] - INTERVAL[0],
                     "method": "conformal split: quantis 10%/90% dos residuos fora da amostra do rolling-origin, por tipo de canal",
                     "calibration": calibration()},
        "baseline_rule": "B0 = log1p(mediana das curtidas de todos os Shorts do canal publicados antes do criativo)",
        "inputs": {"gold": {"path": str(gold_file.relative_to(ROOT)), "sha256": sha256(gold_file)},
                   "embeddings": {"path": str(EMB_PATH.relative_to(ROOT)), "sha256": sha256(EMB_PATH)},
                   "rubric": {"path": str(RUBRIC_PATH.relative_to(ROOT)), "sha256": sha256(RUBRIC_PATH)},
                   "e08_predictions_sha256": sha256(E08 / "rolling_origin_predictions.csv")},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT_BASE / "LATEST.json").write_text(json.dumps({"model_version": version, "path": str(out.relative_to(ROOT))}, indent=2), encoding="utf-8")
    print(f"pacote: {out}")
    print("cobertura conformal (calibra o1+o2, avalia o3):",
          {k: round(v["holdout_check"]["coverage"], 3) for k, v in metadata["interval"]["calibration"].items()})


if __name__ == "__main__":
    main()
