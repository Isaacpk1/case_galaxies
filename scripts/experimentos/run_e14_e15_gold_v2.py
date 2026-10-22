"""E-14 (modelo servido congelado em dados novos) e E-15 (mesma arquitetura retreinada no Gold v2).

Protocolo pre-registrado em docs/07 (secao "E-14 e E-15"). Cada experimento roda UMA vez com
cobertura >= 95% de embedding e rubrica; o script recusa rodar de novo se o relatorio ja existir.

    python scripts/experimentos/run_e14_e15_gold_v2.py --cobertura            # so mostra a cobertura
    python scripts/experimentos/run_e14_e15_gold_v2.py --exp e14 --emb-v2 data/silver/features/thumbnail_siglip2_v2/<run>/embeddings.npz
    python scripts/experimentos/run_e14_e15_gold_v2.py --exp e15 --emb-v2 ...
    python scripts/experimentos/run_e14_e15_gold_v2.py --exp e14 --smoke      # teste do codigo em linhas v1 (sem valor de avaliacao)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "modelo"))
from galaxies_agent.features import KNN_COLS, knn_features, rubric_features  # noqa: E402
import run_e13_test_final as t13  # noqa: E402  (metricas e comparacao pareada do teste final)
import train_tool_model as ttm  # noqa: E402  (hiperparametros do E-08 e PCA)

GOLD = ROOT / "data" / "gold" / "snapshots" / "gold_v2_1_20261021T113931Z_e1fd77c6" / "training_snapshot.csv"
PACKAGE = ROOT / "artifacts" / "tool_model" / "tool_v2_20261014T000408Z"
EMB_V1 = ROOT / "data" / "silver" / "features" / "thumbnail_siglip2_v1" / "20261013T010000Z" / "embeddings.npz"
RUBRIC = ROOT / "data" / "silver" / "gemini_video" / "gemini_video_features.csv"
OUT = ROOT / "artifacts" / "feature_experiments" / "e14_e15_gold_v2"
TARGET, BASELINE = "target_log_likes", "baseline_log_likes"
MIN_COVERAGE = 0.95
STRATA = ("origem_v2", "sector", "grupo")


def load(emb_paths: list[Path]) -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(GOLD, parse_dates=["published_at"], dtype={"video_id": str, "publish_hour_utc": str, "publish_weekday": str})
    data = data.sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)
    emb: dict[str, np.ndarray] = {}
    for path in emb_paths:
        if path and Path(path).exists():
            npz = np.load(path, allow_pickle=True)
            emb.update(zip(npz["video_id"].astype(str), npz["embeddings"].astype(np.float32)))
    matrix = np.full((len(data), 768), np.nan, dtype=np.float32)
    for i, vid in enumerate(data["video_id"]):
        if vid in emb:
            v = emb[vid]
            matrix[i] = v / np.linalg.norm(v)
    data["has_emb"] = ~np.isnan(matrix[:, 0])
    rubric = pd.read_csv(RUBRIC, dtype={"video_id": str}).drop_duplicates("video_id", keep="last").set_index("video_id")
    feats = [rubric_features(rubric.loc[v].to_dict() if v in rubric.index else None, d)
             for v, d in zip(data["video_id"], data["duration_seconds"])]
    data = pd.concat([data, pd.DataFrame(feats, index=data.index)], axis=1)
    data["has_rubric"] = data["gemini_coverage"].eq(1.0)
    return data, matrix


def add_knn(data: pd.DataFrame, matrix: np.ndarray) -> None:
    """Comparaveis: Shorts ANTERIORES do mesmo canal (com embedding) no Gold v2."""
    resid = (data[TARGET] - data[BASELINE]).to_numpy()
    values = np.full((len(data), len(KNN_COLS)), np.nan)
    for _, idx in data.groupby("channel_id").groups.items():
        idx = np.asarray(idx)
        t = data.loc[idx, "published_at"].to_numpy()
        ok = data.loc[idx, "has_emb"].to_numpy()
        for j, i in enumerate(idx):
            if not ok[j]:
                continue
            prior = idx[(t < t[j]) & ok]
            feats = knn_features(matrix[prior], resid[prior], matrix[i])
            values[i] = [feats[c] for c in KNN_COLS]
    data[KNN_COLS] = values


def project(data: pd.DataFrame, matrix: np.ndarray, mean: np.ndarray, components: np.ndarray, cols: list[str]) -> None:
    proj = np.full((len(data), len(cols)), np.nan)
    ok = data["has_emb"].to_numpy()
    proj[ok] = (matrix[ok] - mean) @ components.T
    data[cols] = proj


def pool(frame: pd.DataFrame, numeric: list[str], categorical: list[str], label: bool) -> Pool:
    x = frame[numeric + categorical].copy()
    for c in categorical:
        x[c] = x[c].fillna("<missing>").astype(str)
    return Pool(x, (frame[TARGET] - frame[BASELINE]).to_numpy() if label else None, cat_features=categorical)


def evaluate(frame: pd.DataFrame, arms: list[str], ref: str = "B0") -> dict:
    out, per_channel = {"linhas": int(len(frame)), "canais": int(frame["channel_id"].nunique()), "metricas": {}}, {}
    for arm in arms:
        per_channel[arm], out["metricas"][arm] = t13.metrics(frame, f"pred_{arm}")
    out["comparacoes"] = {f"{a}_vs_{ref}": {m: t13.paired(per_channel[a], per_channel[ref], m) for m in ("mae", "pair_fair")}
                          for a in arms if a != ref and frame["channel_id"].nunique() >= 3}
    out["por_canal"] = {a: v.round(4).to_dict("records") for a, v in per_channel.items()}
    return out


def strata(frame: pd.DataFrame, arms: list[str]) -> dict:
    res = {}
    for key in STRATA:
        res[key] = {}
        for value, g in frame.groupby(key):
            pcs = {a: t13.metrics(g, f"pred_{a}")[1] for a in arms}
            res[key][str(value)] = {"linhas": int(len(g)), "canais": int(g["channel_id"].nunique()),
                                    **{a: {"mae_macro_canal": pcs[a]["mae_macro_canal"], "pair_fair_macro_canal": pcs[a]["pair_fair_macro_canal"]} for a in arms}}
    return res


def interval(frame: pd.DataFrame, col: str, calibration: dict) -> dict:
    lo = frame["channel_type"].map(lambda k: calibration.get(k, calibration["global"])[0])
    hi = frame["channel_type"].map(lambda k: calibration.get(k, calibration["global"])[1])
    inside = (frame[TARGET] >= frame[col] + lo) & (frame[TARGET] <= frame[col] + hi)
    return {"cobertura": round(float(inside.mean()), 4), "alvo": 0.8,
            "por_origem": {k: round(float(v), 4) for k, v in inside.groupby(frame["origem_v2"]).mean().items()}}


def run_e14(data: pd.DataFrame, matrix: np.ndarray, smoke: bool) -> dict:
    meta = json.loads((PACKAGE / "metadata.json").read_text(encoding="utf-8"))
    specs = meta["models"]
    pca = np.load(PACKAGE / "pca.npz")
    emb_cols = [c for c in specs["with_embedding"]["numeric"] if c.startswith("emb_pc")]
    project(data, matrix, pca["mean"], pca["components"], emb_cols)
    add_knn(data, matrix)
    rows = data[data["origem_v2"].eq("v1")] if smoke else data[data["avaliacao_congelada"]]
    rows = rows.copy()
    rows["pred_B0"] = rows[BASELINE]
    names = {"without_embedding": "CB_BASE", "with_embedding": "CB_EMB", "final": "CB_FINAL"}
    for variant, arm in names.items():
        model = CatBoostRegressor().load_model(str(PACKAGE / f"model_{variant}.cbm"))
        need = rows["has_emb"] if variant != "without_embedding" else pd.Series(True, index=rows.index)
        if variant == "final":
            need = need & rows["has_rubric"]
        rows[f"pred_{arm}"] = np.nan
        sub = rows[need]
        if len(sub):
            rows.loc[need, f"pred_{arm}"] = sub[BASELINE].to_numpy() + model.predict(pool(sub, specs[variant]["numeric"], specs[variant]["categorical"], False))
    complete = rows.dropna(subset=["pred_CB_FINAL"])
    arms = ["B0", "CB_BASE", "CB_EMB", "CB_FINAL"]
    report = {"experimento": "E-14", "modelo": meta["model_version"], "smoke": smoke,
              "linhas_avaliacao": int(len(rows)), "linhas_completas": int(len(complete)),
              "cobertura": {"embedding": round(float(rows["has_emb"].mean()), 4), "rubrica": round(float(rows["has_rubric"].mean()), 4)},
              "principal": evaluate(complete, arms), "estratos": strata(complete, arms),
              "intervalo_80": interval(complete, "pred_CB_FINAL", meta["interval"]["calibration"]["final"]),
              "so_metadados_todas_as_linhas": evaluate(rows, ["B0", "CB_BASE"])}
    keep = ["video_id", "channel_id", "channel_name", "grupo", "sector", "channel_type", "origem_v2", "published_at", TARGET, BASELINE] + [f"pred_{a}" for a in arms]
    return report, rows[keep]


def run_e15(data: pd.DataFrame, matrix: np.ndarray, frozen: pd.DataFrame | None) -> dict:
    train_mask = data["split"].eq("train") & data["has_emb"]
    pca = ttm.fit_pca(matrix[train_mask.to_numpy()], ttm.PCA_DIM)
    project(data, matrix, pca["mean"], pca["components"], ttm.EMB_COLS)
    add_knn(data, matrix)
    numeric, categorical, _ = ttm.MODELS["final"]
    params = ttm.e08_params("CB_EMB")
    model = CatBoostRegressor(loss_function="MAE", random_seed=ttm.SEED, verbose=False, allow_writing_files=False,
                              thread_count=-1, **params).fit(pool(data[train_mask], numeric, categorical, True))
    test = data[data["split"].eq("test") & data["has_emb"] & data["has_rubric"]].copy()
    test["pred_B0"] = test[BASELINE]
    test["pred_CB_FINAL_v2"] = test[BASELINE].to_numpy() + model.predict(pool(test, numeric, categorical, False))
    report = {"experimento": "E-15", "treino": int(train_mask.sum()), "cobertura_rubrica_treino": round(float(data.loc[train_mask, "has_rubric"].mean()), 4),
              "principal": evaluate(test, ["B0", "CB_FINAL_v2"]), "estratos": strata(test, ["B0", "CB_FINAL_v2"])}
    if frozen is not None:
        both = test.merge(frozen[["video_id", "pred_CB_FINAL"]], on="video_id").dropna(subset=["pred_CB_FINAL"])
        report["vs_congelado_mesmas_linhas"] = evaluate(both, ["CB_FINAL", "CB_FINAL_v2", "B0"], ref="CB_FINAL")
    return report, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exp", choices=["e14", "e15"])
    parser.add_argument("--emb-v2", type=Path)
    parser.add_argument("--cobertura", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    data, matrix = load([EMB_V1, args.emb_v2])
    cov = {"embedding": round(float(data["has_emb"].mean()), 4), "rubrica": round(float(data["has_rubric"].mean()), 4),
           "por_origem": data.groupby("origem_v2")[["has_emb", "has_rubric"]].mean().round(4).to_dict("index")}
    print(json.dumps({"gold": GOLD.parent.name, "linhas": len(data), "cobertura": cov}, indent=1, ensure_ascii=False))
    if args.cobertura or not args.exp:
        return
    scope = data if args.exp == "e15" else data[data["avaliacao_congelada"]]
    if not args.smoke and min(scope["has_emb"].mean(), scope["has_rubric"].mean()) < MIN_COVERAGE:
        sys.exit(f"cobertura abaixo de {MIN_COVERAGE:.0%}; o protocolo pede esperar o Gemini/embedding")
    out = OUT / ("smoke" if args.smoke else args.exp)
    report_path = out / "report.json"
    if report_path.exists() and not args.smoke:
        sys.exit(f"{args.exp} ja rodou ({report_path}); o protocolo permite uma unica execucao")
    out.mkdir(parents=True, exist_ok=True)
    if args.exp == "e14":
        report, preds = run_e14(data, matrix, args.smoke)
    else:
        frozen_path = OUT / "e14" / "predictions.csv"
        frozen = pd.read_csv(frozen_path, dtype={"video_id": str}) if frozen_path.exists() else None
        report, preds = run_e15(data, matrix, frozen)
    report["rodado_em"] = datetime.now(timezone.utc).isoformat()
    preds.to_csv(out / "predictions.csv", index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    show = {k: v for k, v in report.items() if k not in ("estratos",)}
    for key in ("principal", "so_metadados_todas_as_linhas", "vs_congelado_mesmas_linhas"):
        if key in show:
            show[key] = {k: v for k, v in show[key].items() if k != "por_canal"}
    print(json.dumps(show, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
