"""E-17: quadros de dentro do video (oar1-3 do i.ytimg.com) como features. Pre-registro em docs/07 (commit 69b8cd9).

Entrada: frames_oar.npz do Colab (SigLIP 2 dos tres quadros por Short, float16, NaN quando o quadro nao existe).
Features por Short:
  frames_pc01..16   PCA 16 da media normalizada dos quadros (PCA ajustado so na janela de treino)
  frames_variacao   1 - cosseno medio entre os pares de quadros (troca de cena / variedade visual)
  thumb_vs_frames   cosseno entre a thumbnail e a media dos quadros (a capa representa o conteudo?)
Bracos: CB_FINAL vs CB_FINAL_FR (+18) e CB_FINAL_FR2 (+2), rolling-origin (50/65/80%) so em split = train do
Gold v2.1, hiperparametros do E-08, Holm nas duas comparacoes de MAE. Regra: retido se MAE cair com p de Holm < 0,05
e pares nao piorarem mais de 0,5 p.p.; se os dois passarem, fica o de menor MAE.

Emenda antes de rodar (commit 515d71e): 534 Shorts sem oar1-3 usam hq1-3 (campo `fonte` no npz); sensibilidade so com `oar`.

    python scripts/experimentos/run_e17_video_frames.py --frames data/silver/features/video_frames_oar_v1/<run>/frames_oar.npz
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "modelo"))
import run_e13_test_final as t13  # noqa: E402
import train_tool_model as ttm  # noqa: E402
import train_tool_model_v2 as ttm2  # noqa: E402

ROOT = ttm.ROOT
TARGET, BASELINE = ttm.TARGET, ttm.BASELINE
FR_PCA = [f"frames_pc{i + 1:02d}" for i in range(16)]
FR_SCALARS = ["frames_variacao", "thumb_vs_frames"]
MIN_COVERAGE = 0.95


def frame_source(video_ids: pd.Series, path: Path) -> pd.Series:
    """oar | hq | nenhum por Short (emenda do pre-registro); 'oar' quando o npz nao tem o campo."""
    npz = np.load(path, allow_pickle=True)
    if "fonte" not in npz:
        return pd.Series("oar", index=video_ids.index)
    src = dict(zip(npz["video_id"].astype(str), npz["fonte"].astype(str)))
    return video_ids.map(src).fillna("nenhum")


def frame_features(video_ids: pd.Series, thumbs: np.ndarray, path: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Media normalizada dos quadros (N x 768, NaN sem quadros), escalares e numero de quadros por Short."""
    npz = np.load(path, allow_pickle=True)
    pos = {v: i for i, v in enumerate(npz["video_id"].astype(str))}
    raw = npz["embeddings"].astype(np.float32)
    mean = np.full((len(video_ids), raw.shape[2]), np.nan, dtype=np.float32)
    scalars = pd.DataFrame(np.nan, index=video_ids.index, columns=FR_SCALARS)
    n_frames = np.zeros(len(video_ids), dtype=int)
    for row, vid in enumerate(video_ids):
        if vid not in pos:
            continue
        frames = raw[pos[vid]]
        frames = frames[~np.isnan(frames[:, 0])]
        n_frames[row] = len(frames)
        if not len(frames):
            continue
        frames = frames / np.linalg.norm(frames, axis=1, keepdims=True)
        m = frames.mean(0)
        m = m / np.linalg.norm(m)
        mean[row] = m
        if len(frames) >= 2:
            sims = frames @ frames.T
            scalars.iloc[row, 0] = 1 - float(sims[np.triu_indices(len(frames), 1)].mean())
        scalars.iloc[row, 1] = float(thumbs[row] @ m)
    return mean, scalars, n_frames


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, pvalues[key] * (len(order) - rank)))
        out[key] = round(running, 4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "feature_experiments" / "e17_video_frames")
    parser.add_argument("--teste-codigo", action="store_true", help="roda com dados de mentira; nao vale como experimento")
    args = parser.parse_args()

    gold = pd.read_csv(ttm2.GOLD, parse_dates=["published_at"], dtype={"video_id": str})
    data = gold[gold["split"].eq("train")].sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)
    emb = ttm2.merge_embeddings()
    thumbs = np.vstack([emb[v] for v in data["video_id"]])
    thumbs = thumbs / np.linalg.norm(thumbs, axis=1, keepdims=True)
    ttm.add_knn(data, thumbs)
    ttm.add_rubric(data)
    frames_mean, scalars, n_frames = frame_features(data["video_id"], thumbs, args.frames)
    data[FR_SCALARS] = scalars
    data["fonte_quadros"] = frame_source(data["video_id"], args.frames)
    coverage = {"3_quadros": round(float((n_frames == 3).mean()), 4), "pelo_menos_1": round(float((n_frames >= 1).mean()), 4),
                "distribuicao": {int(k): int(v) for k, v in pd.Series(n_frames).value_counts().sort_index().items()}}
    print(f"treino {len(data)} linhas | cobertura dos quadros {coverage}", flush=True)
    if coverage["3_quadros"] < MIN_COVERAGE and not args.teste_codigo:
        sys.exit(f"cobertura com 3 quadros abaixo de {MIN_COVERAGE:.0%}; o protocolo nao permite rodar")

    numeric, categorical, _ = ttm.MODELS["final"]
    params = ttm.e08_params("CB_EMB")
    arms = {"CB_FINAL": [], "CB_FINAL_FR": FR_PCA + FR_SCALARS, "CB_FINAL_FR2": FR_SCALARS}
    has = ~np.isnan(frames_mean[:, 0])
    preds = []
    for origin, fit_idx, val_idx in ttm2.folds(data):
        frame = data.copy()
        pca = ttm.fit_pca(thumbs[fit_idx], ttm.PCA_DIM)
        frame[ttm.EMB_COLS] = (thumbs - pca["mean"]) @ pca["components"].T
        fit_ok = fit_idx[has[fit_idx]]
        fpca = ttm.fit_pca(frames_mean[fit_ok], 16)
        proj = np.full((len(frame), 16), np.nan)
        proj[has] = (frames_mean[has] - fpca["mean"]) @ fpca["components"].T
        frame[FR_PCA] = proj
        fit, val = frame.loc[fit_idx], frame.loc[val_idx]
        out = val[["video_id", "channel_id", "channel_name", "channel_type", "sector", "fonte_quadros", TARGET, BASELINE] + FR_SCALARS].copy()
        out["origin"], out["pred_B0"] = origin, out[BASELINE]
        for arm, extra in arms.items():
            model = CatBoostRegressor(loss_function="MAE", random_seed=ttm.SEED, verbose=False, allow_writing_files=False,
                                      thread_count=-1, **params).fit(ttm.to_pool(fit, numeric + extra, categorical))
            out[f"pred_{arm}"] = val[BASELINE].to_numpy() + model.predict(ttm.to_pool(val, numeric + extra, categorical, label=False))
        preds.append(out)
        print(f"{origin}: treino {len(fit_idx)} | validacao {len(val_idx)}", flush=True)
    pred = pd.concat(preds, ignore_index=True)

    per, summary = {}, {}
    for arm in ["B0"] + list(arms):
        per[arm], summary[arm] = t13.metrics(pred, f"pred_{arm}")
    comps = {f"{a}_vs_CB_FINAL": {m: t13.paired(per[a], per["CB_FINAL"], m) for m in ("mae", "pair_fair")}
             for a in ("CB_FINAL_FR", "CB_FINAL_FR2")}
    p_holm = holm({k: v["mae"]["p"] for k, v in comps.items()})
    passed = {k: (v["mae"]["delta"] < 0 and p_holm[k] < 0.05 and v["pair_fair"]["delta"] >= -0.005) for k, v in comps.items()}
    winners = [k.replace("_vs_CB_FINAL", "") for k, ok in passed.items() if ok]
    chosen = min(winners, key=lambda a: summary[a]["mae_macro_canal"]) if winners else None

    only_oar = pred[pred["fonte_quadros"].eq("oar")]
    sensitivity = {"linhas": int(len(only_oar)),
                   "mae_macro": {arm: t13.metrics(only_oar, f"pred_{arm}")[1]["mae_macro_canal"] for arm in ["B0"] + list(arms)},
                   "fonte_na_validacao": {k: int(v) for k, v in pred["fonte_quadros"].value_counts().items()}}

    resid = data[TARGET] - data[BASELINE]
    descriptive = {}
    for col in FR_SCALARS:
        q = pd.qcut(data[col], 5, labels=["Q1 (baixo)", "Q2", "Q3", "Q4", "Q5 (alto)"])
        g = resid.groupby(q, observed=True)
        descriptive[col] = {"spearman_com_desvio": round(float(data[col].corr(resid, method="spearman")), 4),
                            "por_quintil_x_mediana_do_canal": {str(k): round(float(np.exp(v)), 3) for k, v in g.mean().items()},
                            "faixa_por_quintil": {str(k): [round(float(a), 3), round(float(b), 3)]
                                                  for k, (a, b) in data.groupby(q, observed=True)[col].agg(["min", "max"]).iterrows()}}

    report = {"experimento": "E-17", "pre_registro": "docs/07, commit 69b8cd9", "teste_de_codigo": args.teste_codigo,
              "quadros": str(args.frames), "cobertura_treino": coverage,
              "linhas_validacao": int(len(pred)), "canais": int(pred["channel_id"].nunique()),
              "metricas": summary, "comparacoes": comps, "p_mae_holm": p_holm, "passou_na_regra": passed,
              "decisao_pre_registrada": f"retido: {chosen}" if chosen else "sem_ganho",
              "sensibilidade_so_oar": sensitivity, "descritivo_treino": descriptive, "rodado_em": datetime.now(timezone.utc).isoformat()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pred.to_csv(args.out / "rolling_origin_predictions.csv", index=False)
    print(json.dumps({k: report[k] for k in ("metricas", "p_mae_holm", "passou_na_regra", "decisao_pre_registrada")},
                     indent=1, ensure_ascii=False))
    for k, v in comps.items():
        print(k, {m: (x["delta"], x["ic95"], x["p"], f"{x['canais_melhor']}/{x['n']}") for m, x in v.items()})
    print("sensibilidade (so quadros oar):", json.dumps(sensitivity, ensure_ascii=False))
    print(json.dumps(descriptive, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
