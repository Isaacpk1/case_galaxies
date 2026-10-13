"""Treina e empacota o modelo servido pela tool (agente preditivo / MCP).

Reproduz a configuracao vencedora do E-08 (CB_EMB: CatBoost, alvo residual sobre o B0, perda MAE,
embedding SigLIP 2 da thumbnail reduzido por PCA) e acrescenta o que a tool precisa:

- modelo principal (com embedding) e modelo reserva (sem embedding, para quando a thumbnail nao puder
  ser embutida); ambos treinados SOMENTE nas linhas de treino do Gold (teste final continua fechado);
- PCA ajustado no treino, salvo como matrizes (sem depender da versao do scikit-learn na inferencia);
- intervalo de 80% por calibracao conformal: quantis 10%/90% dos residuos fora da amostra do
  rolling-origin do E-08, por tipo de canal;
- indice de comparaveis (embeddings completos + metadados dos Shorts de treino);
- historico de curtidas por canal (Silver) para calcular o B0 de um criativo novo.

Saida imutavel: artifacts/tool_model/<model_version>/ e artifacts/tool_model/LATEST.json.

    python scripts/train_tool_model.py            # treina e empacota
    python scripts/train_tool_model.py --validate # tambem refaz o rolling-origin com os parametros fixos
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "artifacts" / "tool_model"
E08 = ROOT / "artifacts" / "feature_experiments" / "e08_catboost_residual"
EMB_PATH = ROOT / "data" / "silver" / "features" / "thumbnail_siglip2_v1" / "20261013T010000Z" / "embeddings.npz"
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
ROLLING_ORIGIN_FRACTIONS = (0.50, 0.65, 0.80)


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
    chosen = pd.read_csv(E08 / "chosen_params.csv")
    row = chosen.query("arm == @arm and origin == 'origin_3'").iloc[0].to_dict()
    params = {k: row[k] for k in ("depth", "learning_rate", "l2_leaf_reg", "random_strength", "bagging_temperature")}
    params["depth"] = int(params["depth"])
    if arm == "CB_EMB":
        params["iterations"] = int(json.loads((E08 / "run_info.json").read_text())["final_params"]["iterations"])
    else:
        params["iterations"] = int(row["iterations"])
    return params


def fit_pca(matrix: np.ndarray, dim: int) -> dict:
    mean = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    components = vt[:dim]
    # sinal deterministico: maior componente absoluto de cada eixo positivo
    signs = np.sign(components[np.arange(dim), np.abs(components).argmax(axis=1)])
    return {"mean": mean.astype(np.float32), "components": (components * signs[:, None]).astype(np.float32)}


def project(matrix: np.ndarray, pca: dict) -> np.ndarray:
    return (matrix - pca["mean"]) @ pca["components"].T


def to_pool(frame: pd.DataFrame, numeric: list[str], label: bool = True) -> Pool:
    x = frame[numeric + CATEGORICAL].copy()
    for column in CATEGORICAL:
        x[column] = x[column].fillna("<missing>").astype(str)
    y = (frame[TARGET] - frame[BASELINE]).to_numpy() if label else None
    return Pool(x, y, cat_features=CATEGORICAL)


def fit_model(frame: pd.DataFrame, numeric: list[str], params: dict) -> CatBoostRegressor:
    model = CatBoostRegressor(loss_function="MAE", random_seed=SEED, verbose=False, allow_writing_files=False,
                              thread_count=-1, **params)
    return model.fit(to_pool(frame, numeric))


def rolling_origin(frame: pd.DataFrame):
    for position, fraction in enumerate(ROLLING_ORIGIN_FRACTIONS):
        nxt = ROLLING_ORIGIN_FRACTIONS[position + 1] if position + 1 < len(ROLLING_ORIGIN_FRACTIONS) else 1.0
        fit_idx, val_idx = [], []
        for _, channel in frame.sort_values(["channel_id", "published_at", "video_id"]).groupby("channel_id"):
            end = max(5, int(np.floor(len(channel) * fraction)))
            stop = min(max(end + 1, int(np.floor(len(channel) * nxt))), len(channel))
            fit_idx.extend(channel.index[:end])
            val_idx.extend(channel.index[end:stop])
        yield f"origin_{position + 1}", frame.loc[fit_idx], frame.loc[val_idx]


def calibration_from_e08() -> dict:
    """Quantis conformais dos residuos fora da amostra do E-08 (todas as origens), por tipo de canal."""
    pred = pd.read_csv(E08 / "rolling_origin_predictions.csv")
    out = {}
    for model_key, column in (("with_embedding", "pred_CB_EMB"), ("without_embedding", "pred_CB_BASE")):
        error = pred[TARGET] - pred[column]
        entry = {"global": [float(q) for q in np.quantile(error, INTERVAL)]}
        for kind, group in pred.assign(error=error).groupby("channel_type"):
            entry[kind] = [float(q) for q in np.quantile(group["error"], INTERVAL)]
        # checagem honesta: calibra em origin_1+2 e mede cobertura em origin_3
        cal, test = pred[pred["origin"] != "origin_3"], pred[pred["origin"] == "origin_3"]
        lo, hi = np.quantile(cal[TARGET] - cal[column], INTERVAL)
        e3 = test[TARGET] - test[column]
        entry["holdout_check"] = {"calibrated_on": "origin_1+origin_2", "evaluated_on": "origin_3",
                                  "coverage": float(((e3 >= lo) & (e3 <= hi)).mean()), "nominal": INTERVAL[1] - INTERVAL[0]}
        out[model_key] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--validate", action="store_true", help="Refaz o rolling-origin com os parametros fixos")
    args = parser.parse_args()

    gold_file = gold_path()
    gold = pd.read_csv(gold_file, parse_dates=["published_at"])
    train = gold.loc[gold["split"].eq("train")].reset_index(drop=True)
    emb_npz = np.load(EMB_PATH, allow_pickle=True)
    emb_by_id = dict(zip(emb_npz["video_id"].astype(str), emb_npz["embeddings"].astype(np.float32)))
    missing = [v for v in train["video_id"] if v not in emb_by_id]
    if missing:
        raise SystemExit(f"{len(missing)} videos de treino sem embedding")
    matrix = np.vstack([emb_by_id[v] for v in train["video_id"]])

    pca = fit_pca(matrix, PCA_DIM)
    train[EMB_COLS] = project(matrix, pca)
    params_emb, params_base = e08_params("CB_EMB"), e08_params("CB_BASE")

    if args.validate:
        rows = []
        for origin, fit, val in rolling_origin(train):
            fit, val = fit.copy(), val.copy()
            local = fit_pca(np.vstack([emb_by_id[v] for v in fit["video_id"]]), PCA_DIM)
            fit[EMB_COLS] = project(np.vstack([emb_by_id[v] for v in fit["video_id"]]), local)
            val[EMB_COLS] = project(np.vstack([emb_by_id[v] for v in val["video_id"]]), local)
            pred = val[BASELINE] + fit_model(fit, NUMERIC + EMB_COLS, params_emb).predict(to_pool(val, NUMERIC + EMB_COLS, False))
            err = (val[TARGET] - pred).abs().groupby(val["channel_id"]).mean().mean()
            b0 = (val[TARGET] - val[BASELINE]).abs().groupby(val["channel_id"]).mean().mean()
            rows.append({"origin": origin, "mae_cb_emb_fixed_params": round(float(err), 4), "mae_b0": round(float(b0), 4)})
            print(rows[-1], flush=True)
        print("media:", pd.DataFrame(rows).mean(numeric_only=True).round(4).to_dict(), "(E-08 com busca por origem: CB_EMB 0,8815 / B0 0,9338)")

    model_emb = fit_model(train, NUMERIC + EMB_COLS, params_emb)
    model_base = fit_model(train, NUMERIC, params_base)

    version = f"tool_v1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = OUT_BASE / version
    out.mkdir(parents=True, exist_ok=False)
    model_emb.save_model(str(out / "model_with_embedding.cbm"))
    model_base.save_model(str(out / "model_without_embedding.cbm"))
    np.savez(out / "pca.npz", **pca)

    silver = pd.read_csv(ROOT / "data" / "silver" / "eligible_shorts" / SILVER_RUN / "eligible_shorts.csv",
                         parse_dates=["published_at"])
    history = silver[["video_id", "channel_id", "channel_name", "sector", "channel_type", "published_at",
                      "like_count", "title", "duration_seconds"]].dropna(subset=["like_count"])
    history.to_csv(out / "channel_history.csv", index=False)

    titles = silver.set_index("video_id")["title"]
    comp = train[["video_id", "channel_id", "channel_name", "sector", "channel_type", "published_at",
                  TARGET, BASELINE]].copy()
    comp["title"] = comp["video_id"].map(titles).fillna("")
    comp["likes"] = np.expm1(comp[TARGET]).round().astype(int)
    comp["relative_to_channel"] = np.exp(comp[TARGET] - comp[BASELINE])
    comp.drop(columns=[TARGET, BASELINE]).to_csv(out / "comparables.csv", index=False)
    norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    np.save(out / "comparables_embeddings.npy", norm.astype(np.float32))

    calibration = calibration_from_e08()
    channel_stats = (history.groupby("channel_id")
                     .agg(channel_name=("channel_name", "first"), sector=("sector", "first"),
                          channel_type=("channel_type", "first"), posts=("video_id", "count"),
                          median_likes=("like_count", "median"), last_post_at=("published_at", "max"))
                     .reset_index())
    channel_stats["in_training"] = channel_stats["channel_id"].isin(train["channel_id"])
    channel_stats.to_csv(out / "channels.csv", index=False)

    metadata = {
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_experiment": "E-08 (CB_EMB)",
        "dataset_version": gold_file.parent.name,
        "training_rows": int(len(train)), "training_channels": int(train["channel_id"].nunique()),
        "test_split": "fechado: modelos treinados so com split=train",
        "target": "target_log_likes - baseline_log_likes (desvio sobre a mediana historica do canal), perda MAE",
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL, "embedding_features": EMB_COLS,
        "params_with_embedding": params_emb, "params_without_embedding": params_base,
        "embedding": {"model": str(emb_npz["source"]), "dim": int(matrix.shape[1]), "pca_dim": PCA_DIM,
                      "preprocess": "painel vertical 9:16 recortado quando a thumbnail e pillarbox; lado menor 256 px",
                      "normalize": "L2 antes do PCA"},
        "interval": {"nominal": INTERVAL[1] - INTERVAL[0], "method": "conformal split: quantis 10%/90% dos residuos do rolling-origin do E-08, por tipo de canal",
                     "calibration": calibration},
        "baseline_rule": "B0 = log1p(mediana das curtidas de todos os Shorts do canal publicados antes do criativo)",
        "inputs": {"gold": {"path": str(gold_file.relative_to(ROOT)), "sha256": sha256(gold_file)},
                   "embeddings": {"path": str(EMB_PATH.relative_to(ROOT)), "sha256": sha256(EMB_PATH)},
                   "e08_predictions_sha256": sha256(E08 / "rolling_origin_predictions.csv")},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT_BASE / "LATEST.json").write_text(json.dumps({"model_version": version, "path": str(out.relative_to(ROOT))}, indent=2), encoding="utf-8")
    print(f"pacote: {out}")
    print("cobertura conformal (calibra o1+o2, avalia o3):",
          {k: round(v["holdout_check"]["coverage"], 3) for k, v in calibration.items()})


if __name__ == "__main__":
    main()
