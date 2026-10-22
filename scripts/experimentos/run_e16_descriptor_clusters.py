"""E-16: clusters K-Means dos descritores do VLM (camada 2 recomendada do TAP). Pre-registro em docs/07.

Os descritores curtos do Gemini (`elementos_video_top5`, `elementos_audio_top5`) viram vetores (TF-IDF de palavras +
caracteres -> SVD 64 -> L2), sao agrupados com K-Means (k pelo melhor silhouette em {8, 12, 16, 20}, so com os
descritores das linhas de treino da janela) e cada Short ganha a contagem dos seus descritores por cluster.

Bracos: CB_FINAL (features do pacote atual) vs CB_FINAL_KM (+ contagens), hiperparametros do E-08, rolling-origin
(50/65/80% de cada canal) SO dentro de split = train do Gold v2.1 (o teste foi aberto no E-15).
Regra: retida se o MAE macro por canal cair com p < 0,05 e a acuracia por pares nao piorar mais de 0,5 p.p.

Saida: artifacts/feature_experiments/e16_descriptor_clusters/ (report.json, predicoes, clusters descritivos)

    python scripts/experimentos/run_e16_descriptor_clusters.py
"""

from __future__ import annotations

import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "modelo"))
import run_e13_test_final as t13  # noqa: E402  (metricas e teste pareado por canal)
import train_tool_model as ttm  # noqa: E402
import train_tool_model_v2 as ttm2  # noqa: E402  (Gold v2.1, embeddings v1+v2, janelas do rolling-origin)

ROOT = ttm.ROOT
OUT = ROOT / "artifacts" / "feature_experiments" / "e16_descriptor_clusters"
TARGET, BASELINE = ttm.TARGET, ttm.BASELINE
FIELDS = {"kv": "elementos_video_top5", "ka": "elementos_audio_top5"}
K_GRID = (8, 12, 16, 20)
SEED = ttm.SEED


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower()).encode("ascii", "ignore").decode()
    return " ".join(text.replace("-", " ").split())


def split_descriptors(value) -> list[str]:
    if not isinstance(value, str):
        return []
    return [c for c in (clean(x) for x in value.split("|")) if c]


class DescriptorClusters:
    """Vetoriza descritores distintos e agrupa com K-Means; k pelo silhouette (sem usar o alvo)."""

    def __init__(self, descriptors: list[str]):
        self.vocab = sorted(set(descriptors))
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        x = hstack([self.word.fit_transform(self.vocab), self.char.fit_transform(self.vocab)]).tocsr()
        self.svd = TruncatedSVD(64, random_state=SEED).fit(x)
        z = normalize(self.svd.transform(x))
        counts = pd.Series(descriptors).value_counts()
        weights = counts.reindex(self.vocab).to_numpy()
        sample = np.random.default_rng(SEED).choice(len(self.vocab), size=min(3000, len(self.vocab)), replace=False)
        best = None
        for k in K_GRID:
            km = KMeans(k, n_init=5, random_state=SEED).fit(z, sample_weight=weights)
            score = silhouette_score(z[sample], km.labels_[sample])
            if best is None or score > best[0]:
                best = (score, k, km)
        self.silhouette, self.k, self.kmeans = best
        self.lookup = dict(zip(self.vocab, self.kmeans.labels_))

    def assign(self, descriptors: list[str]) -> np.ndarray:
        known = [self.lookup[d] for d in descriptors if d in self.lookup]
        unknown = [d for d in descriptors if d not in self.lookup]
        if unknown:
            x = hstack([self.word.transform(unknown), self.char.transform(unknown)]).tocsr()
            known += list(self.kmeans.predict(normalize(self.svd.transform(x))))
        return np.bincount(np.asarray(known, dtype=int), minlength=self.k).astype(float)

    def describe(self, descriptors: list[str], top: int = 6) -> dict[int, list[str]]:
        counts = pd.Series(descriptors).value_counts()
        out = {}
        for cluster in range(self.k):
            members = [d for d in counts.index if self.lookup.get(d) == cluster]
            out[cluster] = members[:top]
        return out


def cluster_features(frame: pd.DataFrame, lists: dict[str, pd.Series], fit_idx: np.ndarray) -> tuple[pd.DataFrame, dict]:
    feats, models = [], {}
    for prefix, series in lists.items():
        model = DescriptorClusters([d for lst in series.loc[fit_idx] for d in lst])
        mat = np.vstack([model.assign(lst) for lst in series])
        feats.append(pd.DataFrame(mat, index=frame.index, columns=[f"{prefix}_{i:02d}" for i in range(model.k)]))
        models[prefix] = model
    return pd.concat(feats, axis=1), models


def main() -> None:
    gold = pd.read_csv(ttm2.GOLD, parse_dates=["published_at"], dtype={"video_id": str})
    data = gold[gold["split"].eq("train")].sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)
    emb = ttm2.merge_embeddings()
    matrix = np.vstack([emb[v] for v in data["video_id"]])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    ttm.add_knn(data, matrix)
    coverage = ttm.add_rubric(data)
    rubric = pd.read_csv(ttm.RUBRIC_PATH, dtype={"video_id": str}).drop_duplicates("video_id").set_index("video_id")
    lists = {p: data["video_id"].map(rubric[col]).map(split_descriptors) for p, col in FIELDS.items()}
    print(f"treino Gold v2.1: {len(data)} linhas, {data['channel_id'].nunique()} canais, rubrica {coverage:.2%}", flush=True)

    numeric, categorical, _ = ttm.MODELS["final"]
    params = ttm.e08_params("CB_EMB")
    preds, ks = [], {}
    for origin, fit_idx, val_idx in ttm2.folds(data):
        pca = ttm.fit_pca(matrix[fit_idx], ttm.PCA_DIM)
        frame = data.copy()
        frame[ttm.EMB_COLS] = (matrix - pca["mean"]) @ pca["components"].T
        clusters, models = cluster_features(frame, lists, fit_idx)
        frame = pd.concat([frame, clusters], axis=1)
        ks[origin] = {p: {"k": m.k, "silhouette": round(float(m.silhouette), 4)} for p, m in models.items()}
        fit, val = frame.loc[fit_idx], frame.loc[val_idx]
        out = val[["video_id", "channel_id", "channel_name", "channel_type", "sector", TARGET, BASELINE]].copy()
        out["origin"] = origin
        for arm, extra in (("CB_FINAL", []), ("CB_FINAL_KM", list(clusters.columns))):
            model = CatBoostRegressor(loss_function="MAE", random_seed=SEED, verbose=False, allow_writing_files=False,
                                      thread_count=-1, **params).fit(ttm.to_pool(fit, numeric + extra, categorical))
            out[f"pred_{arm}"] = val[BASELINE].to_numpy() + model.predict(ttm.to_pool(val, numeric + extra, categorical, label=False))
        out["pred_B0"] = out[BASELINE]
        preds.append(out)
        print(f"{origin}: treino {len(fit_idx)} | validacao {len(val_idx)} | k {ks[origin]}", flush=True)
    pred = pd.concat(preds, ignore_index=True)

    per, summary = {}, {}
    for arm in ("B0", "CB_FINAL", "CB_FINAL_KM"):
        per[arm], summary[arm] = t13.metrics(pred, f"pred_{arm}")
    comp = {m: t13.paired(per["CB_FINAL_KM"], per["CB_FINAL"], m) for m in ("mae", "pair_fair")}
    by_origin = {o: {arm: t13.metrics(g, f"pred_{arm}")[1]["mae_macro_canal"] for arm in ("B0", "CB_FINAL", "CB_FINAL_KM")}
                 for o, g in pred.groupby("origin")}
    retained = comp["mae"]["delta"] < 0 and comp["mae"]["p"] < 0.05 and comp["pair_fair"]["delta"] >= -0.005

    # descritivo: clusters ajustados em todo o treino e desvio medio (real - mediana do canal) dos Shorts com cada cluster
    full_clusters, full_models = cluster_features(data, lists, data.index.to_numpy())
    resid = data[TARGET] - data[BASELINE]
    descriptive = {}
    for prefix, model in full_models.items():
        names = model.describe([d for lst in lists[prefix] for d in lst])
        rows = []
        for c in range(model.k):
            has = full_clusters[f"{prefix}_{c:02d}"] > 0
            rows.append({"cluster": f"{prefix}_{c:02d}", "exemplos": names[c], "shorts": int(has.sum()),
                         "desvio_medio_log": round(float(resid[has].mean()), 3),
                         "x_mediana_do_canal": round(float(np.exp(resid[has].mean())), 2),
                         "desvio_medio_sem_o_cluster": round(float(resid[~has].mean()), 3)})
        descriptive[FIELDS[prefix]] = {"k": model.k, "silhouette": round(float(model.silhouette), 4),
                                       "clusters": sorted(rows, key=lambda r: -r["desvio_medio_log"])}

    report = {
        "experimento": "E-16", "pre_registro": "docs/07, commit d22615a",
        "linhas_validacao": int(len(pred)), "canais": int(pred["channel_id"].nunique()), "k_por_origem": ks,
        "metricas": summary, "mae_macro_por_origem": by_origin,
        "CB_FINAL_KM_vs_CB_FINAL": comp,
        "decisao_pre_registrada": "retida" if retained else "sem_ganho",
        "descritivo_treino_completo": descriptive,
        "rodado_em": datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pred.to_csv(OUT / "rolling_origin_predictions.csv", index=False)
    print(json.dumps({k: report[k] for k in ("metricas", "mae_macro_por_origem", "CB_FINAL_KM_vs_CB_FINAL",
                                             "decisao_pre_registrada")}, indent=1, ensure_ascii=False))
    for field, d in descriptive.items():
        print(f"\n{field}: k={d['k']} silhouette={d['silhouette']}")
        for r in d["clusters"]:
            print(f"  {r['cluster']} {r['x_mediana_do_canal']}x n={r['shorts']} {r['exemplos'][:4]}")


if __name__ == "__main__":
    main()
