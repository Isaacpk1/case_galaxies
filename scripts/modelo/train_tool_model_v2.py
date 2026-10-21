"""Treina e empacota o modelo servido pela tool no Gold v2 (decisao pre-registrada do E-15).

Mesma arquitetura e mesmos hiperparametros do pacote anterior (train_tool_model.py: CB_FINAL do E-13 com
parametros do E-08; reservas CB_EMB e CB_BASE), com tres mudancas de dados:
  - base: Gold v2.1 (68 canais, automotivo + moda, pessoa / marca / midia), todas as linhas
    (treino + teste: o teste foi aberto uma unica vez no E-15);
  - embedding: SigLIP 2 v1 (Shorts do Gold v1) + v2 (Shorts novos), mesmo modelo e preprocess,
    unidos num unico arquivo de cache para a tool;
  - historico dos canais: Silver v2 completa (todos os Shorts elegiveis coletados de cada canal); os canais
    do Gold v1 que ficaram fora do v2 mantem o historico da Silver v1, fora do treino (confianca baixa).

Intervalo de 80%: recalibrado no Gold v2. O E-14 mostrou que os quantis do Gold v1 cobrem so 72,8% nos
dados novos. Mesmo metodo do pacote anterior (conformal split, quantis 10%/90% dos residuos fora da
amostra, por tipo de canal), agora com rolling-origin (50% / 65% / 80% de cada canal) rodado no Gold v2.1
para os tres modelos; `holdout_check` calibra nas origens 1+2 e mede a cobertura na origem 3.

Saida imutavel: artifacts/tool_model/tool_v3_<timestamp>/ e artifacts/tool_model/LATEST.json.

    python scripts/modelo/train_tool_model_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_tool_model as ttm  # noqa: E402  (mesmas features, PCA, pools e hiperparametros do pacote anterior)

ROOT = ttm.ROOT
TARGET, BASELINE = ttm.TARGET, ttm.BASELINE
GOLD = ROOT / "data" / "gold" / "snapshots" / "gold_v2_1_20261021T113931Z_e1fd77c6" / "training_snapshot.csv"
EMB_V1 = ttm.EMB_PATH
EMB_V2 = ROOT / "data" / "silver" / "features" / "thumbnail_siglip2_v2" / "20261021T172500Z" / "embeddings.npz"
EMB_MERGED = ROOT / "data" / "silver" / "features" / "thumbnail_siglip2_v1_v2" / "20261021T172500Z" / "embeddings.npz"
SILVER_IDS = ["v2auto_20261020T232948Z", "v2moda_20261021T000925Z"]
E14_E15 = ROOT / "artifacts" / "feature_experiments" / "e14_e15_gold_v2"
FRACTIONS = (0.50, 0.65, 0.80)  # rolling-origin do projeto (E-08 / E-13)


def merge_embeddings() -> dict[str, np.ndarray]:
    """Uniao v1 + v2 (conjuntos disjuntos de video_id; mesmo modelo e preprocess). Grava o cache da tool."""
    emb: dict[str, np.ndarray] = {}
    source = None
    for path in (EMB_V1, EMB_V2):
        npz = np.load(path, allow_pickle=True)
        source = str(npz["source"])
        for vid, vec in zip(npz["video_id"].astype(str), npz["embeddings"].astype(np.float32)):
            if vid in emb:
                raise SystemExit(f"video_id repetido entre v1 e v2: {vid}")
            emb[vid] = vec
    if not EMB_MERGED.exists():
        EMB_MERGED.parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(emb)
        np.savez(EMB_MERGED, embeddings=np.vstack([emb[v] for v in ids]), video_id=np.array(ids), source=np.array(source))
    return emb


def folds(frame: pd.DataFrame):
    """Janelas expansivas por canal, iguais ao E-13: treino ate a fracao, validacao ate a proxima."""
    for pos, frac in enumerate(FRACTIONS):
        nxt = FRACTIONS[pos + 1] if pos + 1 < len(FRACTIONS) else 1.0
        fit, val = [], []
        for _, ch in frame.groupby("channel_id"):
            end = max(5, int(np.floor(len(ch) * frac)))
            stop = min(max(end + 1, int(np.floor(len(ch) * nxt))), len(ch))
            fit.extend(ch.index[:end])
            val.extend(ch.index[end:stop])
        yield f"origin_{pos + 1}", np.array(fit), np.array(val)


def rolling_origin(data: pd.DataFrame, matrix: np.ndarray, params: dict) -> pd.DataFrame:
    """Residuos fora da amostra dos tres modelos no Gold v2.1 (PCA ajustado so na janela de treino)."""
    out = []
    for origin, fit_idx, val_idx in folds(data):
        pca = ttm.fit_pca(matrix[fit_idx], ttm.PCA_DIM)
        fit, val = data.loc[fit_idx].copy(), data.loc[val_idx].copy()
        fit[ttm.EMB_COLS] = (matrix[fit_idx] - pca["mean"]) @ pca["components"].T
        val[ttm.EMB_COLS] = (matrix[val_idx] - pca["mean"]) @ pca["components"].T
        pred = val[["video_id", "channel_id", "channel_type", "sector", TARGET, BASELINE]].copy()
        pred["origin"] = origin
        for name, (numeric, categorical, _) in ttm.MODELS.items():
            model = CatBoostRegressor(loss_function="MAE", random_seed=ttm.SEED, verbose=False, allow_writing_files=False,
                                      thread_count=-1, **params[name]).fit(ttm.to_pool(fit, numeric, categorical))
            pred[f"pred_{name}"] = val[BASELINE].to_numpy() + model.predict(ttm.to_pool(val, numeric, categorical, label=False))
        print(f"rolling-origin {origin}: treino {len(fit_idx)} | validacao {len(val_idx)}", flush=True)
        out.append(pred)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    data = pd.read_csv(GOLD, parse_dates=["published_at"], dtype={"video_id": str})
    data = data.sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)
    emb = merge_embeddings()
    missing = [v for v in data["video_id"] if v not in emb]
    if missing:
        raise SystemExit(f"{len(missing)} videos sem embedding")
    matrix = np.vstack([emb[v] for v in data["video_id"]])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    ttm.add_knn(data, matrix)  # so Shorts anteriores do mesmo canal: vale para rolling-origin e treino final
    coverage = ttm.add_rubric(data)
    print(f"linhas {len(data)} | canais {data['channel_id'].nunique()} | cobertura Gemini {coverage:.2%}", flush=True)
    params = {"final": ttm.e08_params("CB_EMB"), "with_embedding": ttm.e08_params("CB_EMB"),
              "without_embedding": ttm.e08_params("CB_BASE")}

    rolling = rolling_origin(data, matrix, params)
    cal = {name: ttm.residual_quantiles(rolling, f"pred_{name}", f"rolling-origin no Gold v2.1 ({ref})")
           for name, (_, _, ref) in ttm.MODELS.items()}

    pca = ttm.fit_pca(matrix, ttm.PCA_DIM)
    data[ttm.EMB_COLS] = (matrix - pca["mean"]) @ pca["components"].T
    version = f"tool_v3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = ttm.OUT_BASE / version
    out.mkdir(parents=True, exist_ok=False)
    features = {}
    for name, (numeric, categorical, ref) in ttm.MODELS.items():
        model = CatBoostRegressor(loss_function="MAE", random_seed=ttm.SEED, verbose=False, allow_writing_files=False,
                                  thread_count=-1, **params[name]).fit(ttm.to_pool(data, numeric, categorical))
        model.save_model(str(out / f"model_{name}.cbm"))
        features[name] = {"numeric": numeric, "categorical": categorical, "reference": ref, "params": params[name]}
        print(f"modelo {name} ok", flush=True)
    np.savez(out / "pca.npz", **pca)
    rolling.to_csv(out / "calibration_rolling_origin.csv", index=False)

    parts = [pd.read_csv(ROOT / "data" / "silver" / "eligible_shorts" / sid / "history_all_eligible.csv",
                         dtype={"video_id": str}, parse_dates=["published_at"]) for sid in SILVER_IDS]
    silver = pd.concat(parts, ignore_index=True).drop_duplicates("video_id")
    # canais do Gold v1 que ficaram fora do v2 (D-022): o historico continua na tool para o canal ser
    # atendido pelo channel_id, com in_training=False (a tool marca confianca baixa e diz o motivo)
    v1 = pd.read_csv(ROOT / "data" / "silver" / "eligible_shorts" / ttm.SILVER_RUN / "eligible_shorts.csv",
                     dtype={"video_id": str}, parse_dates=["published_at"])
    v1_only = v1[~v1["channel_id"].isin(set(silver["channel_id"]))]
    silver = pd.concat([silver, v1_only], ignore_index=True)
    silver["like_count"] = pd.to_numeric(silver["like_count"], errors="coerce")
    history = silver[["video_id", "channel_id", "channel_name", "sector", "channel_type", "published_at",
                      "like_count", "title", "duration_seconds"]].dropna(subset=["like_count"])
    history.to_csv(out / "channel_history.csv", index=False)

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

    e15 = json.loads((E14_E15 / "e15" / "report.json").read_text(encoding="utf-8"))
    e14 = json.loads((E14_E15 / "e14" / "report.json").read_text(encoding="utf-8"))
    test = e15["principal"]
    metadata = {
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_experiment": "E-15 (CB_FINAL retreinado no Gold v2), aprovado pela regra pre-registrada; "
                                "mesma arquitetura e hiperparametros do E-13 / E-08; reservas CB_EMB e CB_BASE",
        "previous_model": json.loads((ttm.OUT_BASE / "LATEST.json").read_text(encoding="utf-8"))["model_version"],
        "dataset_version": GOLD.parent.name,
        "training_scope": "todo o Gold v2.1 (treino + teste); teste aberto uma unica vez no E-15",
        "training_rows": int(len(data)), "training_channels": int(data["channel_id"].nunique()),
        "test_split": "aberto uma unica vez no E-15 (modelo treinado so com split=train); este pacote foi re-treinado em todo o Gold v2.1",
        "final_test": {
            "experiment": "E-15", "rows": test["linhas"], "channels": test["canais"],
            "mae_macro": {"B0": test["metricas"]["B0"]["mae_macro_canal"], "CB_FINAL": test["metricas"]["CB_FINAL_v2"]["mae_macro_canal"]},
            "pair_accuracy": {"B0": test["metricas"]["B0"]["pair_fair_macro_canal"], "CB_FINAL": test["metricas"]["CB_FINAL_v2"]["pair_fair_macro_canal"]},
            "vs_previous_model_same_rows": {k: {"mae_macro": v["mae_macro_canal"], "pair_accuracy": v["pair_fair_macro_canal"]}
                                            for k, v in e15["vs_congelado_mesmas_linhas"]["metricas"].items()},
            "interval_coverage": cal["final"]["holdout_check"]["coverage"],
        },
        "frozen_previous_model_on_new_data": {
            "experiment": "E-14", "rows": e14["principal"]["linhas"], "channels": e14["principal"]["canais"],
            "mae_macro": {k: v["mae_macro_canal"] for k, v in e14["principal"]["metricas"].items()},
            "pair_accuracy": {k: v["pair_fair_macro_canal"] for k, v in e14["principal"]["metricas"].items()},
            "interval_coverage_v1_calibration": e14["intervalo_80"]["cobertura"]},
        "target": "target_log_likes - baseline_log_likes (desvio sobre a mediana historica do canal), perda MAE",
        "models": features,
        "gemini_coverage_training": round(coverage, 4),
        "embedding": {"model": "google/siglip2-base-patch16-224", "dim": int(matrix.shape[1]), "pca_dim": ttm.PCA_DIM,
                      "preprocess": "painel vertical 9:16 recortado quando a thumbnail e pillarbox; lado menor 256 px",
                      "normalize": "L2 antes do PCA"},
        "rubric": {"model": "gemini-3.5-flash-lite", "prompt_version": "rubrica_tap_v1", "media_resolution": "MEDIA_RESOLUTION_LOW",
                   "source": "URL publica do YouTube (video nao e baixado)"},
        "interval": {"nominal": ttm.INTERVAL[1] - ttm.INTERVAL[0],
                     "method": "conformal split: quantis 10%/90% dos residuos fora da amostra do rolling-origin no Gold v2.1 "
                               "(50%/65%/80% de cada canal), por tipo de canal",
                     "calibration": cal},
        "baseline_rule": "B0 = log1p(mediana das curtidas de todos os Shorts do canal publicados antes do criativo)",
        "inputs": {"gold": {"path": str(GOLD.relative_to(ROOT)), "sha256": ttm.sha256(GOLD)},
                   "embeddings": {"path": str(EMB_MERGED.relative_to(ROOT)), "sha256": ttm.sha256(EMB_MERGED),
                                  "built_from": {str(p.relative_to(ROOT)): ttm.sha256(p) for p in (EMB_V1, EMB_V2)}},
                   "rubric": {"path": str(ttm.RUBRIC_PATH.relative_to(ROOT)), "sha256": ttm.sha256(ttm.RUBRIC_PATH)},
                   "silver_history": SILVER_IDS,
                   "silver_history_v1_only_channels": {"silver_id": ttm.SILVER_RUN,
                                                       "channels": int(v1_only["channel_id"].nunique()),
                                                       "in_training": False}},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (ttm.OUT_BASE / "LATEST.json").write_text(json.dumps({"model_version": version, "path": str(out.relative_to(ROOT))}, indent=2),
                                              encoding="utf-8")
    print(f"pacote: {out}")
    print("cobertura conformal (calibra o1+o2, avalia o3):",
          {k: round(v["holdout_check"]["coverage"], 3) for k, v in cal.items()})


if __name__ == "__main__":
    main()
