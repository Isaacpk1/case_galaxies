"""E-13: comparacao unica CB_FINAL vs CB_EMB (protocolo pre-registrado em docs/07).

CB_FINAL = CB_EMB + comparaveis (E-12) + rubrica Gemini + fala derivada da transcricao Gemini.
Parametros fixos do E-08. Rolling-origin do projeto, teste final fechado.

    python scripts/run_e13_final.py --max-seconds 160          # retomavel (tarefas por origem/braco)
    python scripts/run_e13_final.py --report                   # decisao pela regra pre-registrada
    python scripts/run_e13_final.py --report --preliminar      # roda mesmo com cobertura < 90% (nao decide)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "feature_experiments" / "e13_final"
TASKS = OUT / "tasks"
E08 = ROOT / "artifacts" / "feature_experiments" / "e08_catboost_residual"
THUMB_EMB = ROOT / "data" / "silver" / "features" / "thumbnail_siglip2_v1" / "20261013T010000Z" / "embeddings.npz"
RUBRIC = ROOT / "data" / "silver" / "gemini_video" / "gemini_video_features.csv"
TARGET, BASELINE = "target_log_likes", "baseline_log_likes"
BASE_NUM = ["duration_seconds", "channel_prior_posts", "baseline_log_likes", "days_since_previous_post",
            "title_length", "description_length", "tag_count"]
BASE_CAT = ["channel_id", "sector", "channel_type", "publish_hour_utc", "publish_weekday"]
EMB_COLS = [f"emb_pc{i:02d}" for i in range(16)]
KNN_COLS = ["knn5_resid_mean", "knn5_sim_mean", "visual_novelty"]
SCORES = ["forca_hook_0a10", "densidade_texto_0a10", "ritmo_cortes_0a10", "energia_musica_0a10",
          "brilho_saturacao_0a10", "estetica_geral_0a10", "clareza_beneficio_produto_0a10", "potencial_viral_0a10"]
FLAGS = ["tem_rosto", "pessoa_fala_para_camera", "close_up_produto", "before_after_ou_transformacao",
         "try_on_ou_demonstracao", "texto_na_tela", "tem_musica", "chamada_para_acao"]
SPEECH = ["tem_fala", "fala_palavras_log", "fala_palavras_por_segundo"]
GEMINI_NUM = SCORES + FLAGS + SPEECH + ["gemini_coverage"]
GEMINI_CAT = ["tom_narracao"]
FRACTIONS = (0.50, 0.65, 0.80)
SEED = 42
ARMS = {  # nome: (numericas extras, categoricas extras)
    "CB_EMB": ([], []),
    "CB_EMB_KNN": (KNN_COLS, []),
    "CB_EMB_GEMINI": (GEMINI_NUM, GEMINI_CAT),
    "CB_FINAL": (KNN_COLS + GEMINI_NUM, GEMINI_CAT),
}


def params() -> dict:
    final = json.loads((E08 / "run_info.json").read_text())["final_params"]
    return {k: (int(v) if k in ("depth", "iterations") else v) for k, v in final.items()}


def load() -> tuple[pd.DataFrame, np.ndarray, float]:
    latest = json.loads((ROOT / "data" / "gold" / "snapshots" / "LATEST.json").read_text())
    gold = pd.read_csv(ROOT / latest["path"], parse_dates=["published_at"])
    data = gold[gold["split"] == "train"].sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)
    data["resid"] = data[TARGET] - data[BASELINE]
    z = np.load(THUMB_EMB, allow_pickle=True)
    emb = dict(zip(z["video_id"].astype(str), z["embeddings"].astype(np.float32)))
    E = np.vstack([emb[v] for v in data["video_id"]])

    # comparaveis: so Shorts anteriores do mesmo canal (igual ao E-12)
    for col in KNN_COLS:
        data[col] = np.nan
    for _, idx in data.groupby("channel_id").groups.items():
        idx = np.asarray(idx)
        t = data.loc[idx, "published_at"].to_numpy()
        for j, i in enumerate(idx):
            prior = idx[t < t[j]]
            if len(prior) < 5:
                continue
            sims = E[prior] @ E[i]
            top = np.argsort(-sims)[:5]
            data.loc[i, "knn5_resid_mean"] = data.loc[prior[top], "resid"].mean()
            data.loc[i, "knn5_sim_mean"] = sims[top].mean()
            centroid = E[prior].mean(0)
            data.loc[i, "visual_novelty"] = 1 - float(centroid @ E[i]) / (np.linalg.norm(centroid) + 1e-9)

    rubric = pd.read_csv(RUBRIC) if RUBRIC.exists() else pd.DataFrame(columns=["video_id"])
    if len(rubric):
        rubric["transcricao"] = rubric["transcricao"].fillna("")
        words = rubric["transcricao"].str.split().str.len().fillna(0)
        rubric["tem_fala"] = (words > 0).astype(float)
        rubric["fala_palavras_log"] = np.log1p(words)
        rubric = rubric[["video_id"] + SCORES + FLAGS + GEMINI_CAT + ["tem_fala", "fala_palavras_log"]].assign(_words=words)
        rubric["gemini_coverage"] = 1.0
    data = data.merge(rubric, on="video_id", how="left")
    if "_words" in data:
        data["fala_palavras_por_segundo"] = data["_words"] / data["duration_seconds"].clip(lower=1)
        data = data.drop(columns="_words")
    for col in SCORES + FLAGS + SPEECH + ["gemini_coverage"]:
        if col not in data:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["gemini_coverage"] = data["gemini_coverage"].fillna(0.0)
    data["tom_narracao"] = data.get("tom_narracao", pd.Series(index=data.index, dtype=object)).fillna("<sem_rubrica>")
    return data, E, float(data["gemini_coverage"].mean())


def folds(frame):
    for pos, frac in enumerate(FRACTIONS):
        nxt = FRACTIONS[pos + 1] if pos + 1 < len(FRACTIONS) else 1.0
        fit, val = [], []
        for _, ch in frame.groupby("channel_id"):
            end = max(5, int(np.floor(len(ch) * frac)))
            stop = min(max(end + 1, int(np.floor(len(ch) * nxt))), len(ch))
            fit.extend(ch.index[:end]); val.extend(ch.index[end:stop])
        yield f"origin_{pos + 1}", np.array(fit), np.array(val)


def run_task(data, E, origin, arm) -> pd.DataFrame:
    fit_idx, val_idx = next((f, v) for o, f, v in folds(data) if o == origin)
    mean = E[fit_idx].mean(0)
    _, _, vt = np.linalg.svd(E[fit_idx] - mean, full_matrices=False)
    fit, val = data.loc[fit_idx].copy(), data.loc[val_idx].copy()
    fit[EMB_COLS] = (E[fit_idx] - mean) @ vt[:16].T
    val[EMB_COLS] = (E[val_idx] - mean) @ vt[:16].T
    extra_num, extra_cat = ARMS[arm]
    num, cat = BASE_NUM + EMB_COLS + extra_num, BASE_CAT + extra_cat

    def pool(frame, label):
        x = frame[num + cat].copy()
        for c in cat:
            x[c] = x[c].fillna("<missing>").astype(str)
        return Pool(x, frame["resid"].to_numpy() if label else None, cat_features=cat)

    model = CatBoostRegressor(loss_function="MAE", random_seed=SEED, verbose=False, allow_writing_files=False,
                              thread_count=-1, **params()).fit(pool(fit, True))
    out = val[["video_id", "channel_id", "channel_name", "channel_type", TARGET, BASELINE, "gemini_coverage"]].copy()
    out["pred"] = val[BASELINE].to_numpy() + model.predict(pool(val, False))
    out["model"], out["origin"] = arm, origin
    return out


def pairwise(a, p):
    l, r = np.triu_indices(len(a), k=1)
    sa, sp = np.sign(a[l] - a[r]), np.sign(p[l] - p[r])
    ok = sa != 0
    return float(np.where(sp[ok] == 0, 0.5, (sa[ok] == sp[ok]).astype(float)).mean()) if ok.any() else np.nan


def report(coverage: float, preliminar: bool) -> None:
    pred = pd.concat(pd.read_csv(f) for f in sorted(TASKS.glob("*.csv")))
    rows = []
    for (m, o, c), g in pred.groupby(["model", "origin", "channel_name"]):
        a, p = g[TARGET].to_numpy(), g["pred"].to_numpy()
        rows.append({"model": m, "origin": o, "channel_name": c, "mae": float(np.mean(np.abs(a - p))), "pair_fair": pairwise(a, p)})
    pc = pd.DataFrame(rows).groupby(["model", "channel_name"], as_index=False)[["mae", "pair_fair"]].mean()
    summary = pc.groupby("model")[["mae", "pair_fair"]].mean().round(4)

    def paired(cand, metric="mae"):
        v = pc.pivot(index="channel_name", columns="model", values=metric)
        d = (v[cand] - v["CB_EMB"]).dropna().to_numpy()
        rng = np.random.default_rng(SEED)
        s = rng.choice((-1, 1), (10000, len(d)))
        boot = rng.choice(d, (10000, len(d))).mean(1)
        return {"candidato": cand, "metrica": metric, "delta": round(float(d.mean()), 4),
                "ic95": [round(float(np.quantile(boot, .025)), 4), round(float(np.quantile(boot, .975)), 4)],
                "p": round(float((np.abs((s * d).mean(1)) >= abs(d.mean())).mean()), 4),
                "canais_melhor": int(((d < 0) if metric == "mae" else (d > 0)).sum()), "n": int(len(d))}

    decision = paired("CB_FINAL")
    approved = decision["delta"] < 0 and decision["p"] < 0.05
    result = {
        "cobertura_gemini_treino": round(coverage, 3), "preliminar": preliminar or coverage < 0.9,
        "resumo": summary.reset_index().to_dict("records"),
        "decisao_pre_registrada": {**decision, "aprovado": bool(approved) and coverage >= 0.9 and not preliminar,
                                   "modelo_final": ("CB_FINAL" if approved and coverage >= 0.9 and not preliminar else
                                                    "pendente (cobertura < 90% ou preliminar)" if coverage < 0.9 or preliminar else "CB_EMB")},
        "descritivo_ablacao": [paired(c, m) for c in ("CB_EMB_KNN", "CB_EMB_GEMINI", "CB_FINAL") for m in ("mae", "pair_fair")],
    }
    name = "report_preliminar.json" if result["preliminar"] else "report.json"
    (OUT / name).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=float, default=160)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--preliminar", action="store_true", help="Marca a rodada como nao decisoria")
    parser.add_argument("--reset", action="store_true", help="Apaga tarefas antigas (ex.: depois que a cobertura do Gemini mudou)")
    args = parser.parse_args()
    data, E, coverage = load()
    print(f"linhas de treino: {len(data)} | cobertura Gemini no treino: {coverage:.1%}")
    TASKS.mkdir(parents=True, exist_ok=True)
    stamp = OUT / "coverage.json"
    previous = json.loads(stamp.read_text())["coverage"] if stamp.exists() else None
    if args.reset or (previous is not None and abs(previous - coverage) > 1e-9):
        for f in TASKS.glob("*.csv"):
            f.unlink()
        print("tarefas antigas descartadas (cobertura mudou ou --reset)")
    stamp.write_text(json.dumps({"coverage": coverage}))
    if args.report:
        report(coverage, args.preliminar)
        return
    start = time.time()
    for origin in ("origin_1", "origin_2", "origin_3"):
        for arm in ARMS:
            path = TASKS / f"{arm}__{origin}.csv"
            if path.exists():
                continue
            if time.time() - start > args.max_seconds:
                print("tempo esgotado; rode de novo para continuar")
                return
            t0 = time.time()
            run_task(data, E, origin, arm).to_csv(path, index=False)
            print(f"ok {arm} {origin} ({time.time() - t0:.0f}s)", flush=True)
    print("todas as tarefas concluidas; rode --report")


if __name__ == "__main__":
    main()
