"""Intervalo de 80%: qual calibracao cobre ~80% em todos os grupos? (analise; nao altera a tool sem decisao)

Motivo: no teste do E-15 o intervalo do tool_v3 (conformal por tipo de canal) cobriu 80% no geral, mas 87% em marcas e
74% em midias. Compara, SO dentro de split = train do Gold v2.1 (o teste ja foi aberto), calibrando com os residuos das
origens 1 e 2 do rolling-origin e medindo na origem 3:
  conformal global | por tipo de canal (atual) | por grupo (pessoa/marca/midia) | por grupo x setor
  quantis do GBM (CatBoost, perda Quantile 10% e 90%, citado no TAP) crus | CQR global | CQR por grupo
CQR = quantis do GBM corrigidos por conformal (Romano et al., 2019): alarga ou estreita a faixa pela folga observada.

Saida: artifacts/feature_experiments/interval_calibration/report.json

    python scripts/experimentos/analyze_interval_calibration.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "modelo"))
import train_tool_model as ttm  # noqa: E402
import train_tool_model_v2 as ttm2  # noqa: E402

ROOT = ttm.ROOT
OUT = ROOT / "artifacts" / "feature_experiments" / "interval_calibration"
TARGET, BASELINE = ttm.TARGET, ttm.BASELINE
LO, HI = ttm.INTERVAL
GROUPS = ("grupo", "sector", "channel_type")


def conformal(cal: pd.DataFrame, ev: pd.DataFrame, keys: list[str] | None) -> tuple[np.ndarray, np.ndarray]:
    err = cal[TARGET] - cal["pred"]
    glob = np.quantile(err, [LO, HI])
    if not keys:
        q = np.tile(glob, (len(ev), 1))
    else:
        table = {(k if isinstance(k, tuple) else (k,)): np.quantile(err.loc[g.index], [LO, HI]) for k, g in cal.groupby(keys)}
        q = np.vstack([table.get(k, glob) for k in ev[keys].itertuples(index=False, name=None)])  # chaves sempre tuplas
    return ev["pred"].to_numpy() + q[:, 0], ev["pred"].to_numpy() + q[:, 1]


def cqr(cal: pd.DataFrame, ev: pd.DataFrame, keys: list[str] | None) -> tuple[np.ndarray, np.ndarray]:
    level = HI - LO
    score = np.maximum(cal["q_lo"] - cal[TARGET], cal[TARGET] - cal["q_hi"])

    def corr(s: pd.Series) -> float:
        n = len(s)
        return float(np.quantile(s, min(1.0, np.ceil((n + 1) * level) / n)))

    if not keys:
        c = np.full(len(ev), corr(score))
    else:
        table = {(k if isinstance(k, tuple) else (k,)): corr(score.loc[g.index]) for k, g in cal.groupby(keys)}
        c = np.array([table.get(k, corr(score)) for k in ev[keys].itertuples(index=False, name=None)])
    return ev["q_lo"].to_numpy() - c, ev["q_hi"].to_numpy() + c


def summarize(ev: pd.DataFrame, lo: np.ndarray, hi: np.ndarray) -> dict:
    inside = (ev[TARGET] >= lo) & (ev[TARGET] <= hi)
    width = np.exp(hi - lo)
    out = {"cobertura": round(float(inside.mean()), 4), "largura_mediana_fator": round(float(np.median(width)), 2)}
    for col in GROUPS:
        out[f"por_{col}"] = {str(k): round(float(v), 4) for k, v in inside.groupby(ev[col]).mean().items()}
    cov_groups = list(out["por_grupo"].values()) + list(out["por_sector"].values())
    out["pior_desvio_de_80pct_entre_grupos_e_setores"] = round(float(max(abs(c - 0.8) for c in cov_groups)), 4)
    return out


def main() -> None:
    gold = pd.read_csv(ttm2.GOLD, parse_dates=["published_at"], dtype={"video_id": str})
    data = gold[gold["split"].eq("train")].sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)
    emb = ttm2.merge_embeddings()
    matrix = np.vstack([emb[v] for v in data["video_id"]])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    ttm.add_knn(data, matrix)
    ttm.add_rubric(data)
    numeric, categorical, _ = ttm.MODELS["final"]
    params = ttm.e08_params("CB_EMB")

    preds = []
    for origin, fit_idx, val_idx in ttm2.folds(data):
        pca = ttm.fit_pca(matrix[fit_idx], ttm.PCA_DIM)
        frame = data.copy()
        frame[ttm.EMB_COLS] = (matrix - pca["mean"]) @ pca["components"].T
        fit, val = frame.loc[fit_idx], frame.loc[val_idx]
        out = val[["video_id", "channel_id", "grupo", "sector", "channel_type", TARGET, BASELINE]].copy()
        out["origin"] = origin
        for name, loss in (("pred", "MAE"), ("q_lo", f"Quantile:alpha={LO}"), ("q_hi", f"Quantile:alpha={HI}")):
            model = CatBoostRegressor(loss_function=loss, random_seed=ttm.SEED, verbose=False, allow_writing_files=False,
                                      thread_count=-1, **params).fit(ttm.to_pool(fit, numeric, categorical))
            out[name] = val[BASELINE].to_numpy() + model.predict(ttm.to_pool(val, numeric, categorical, label=False))
        preds.append(out)
        print(f"{origin}: treino {len(fit_idx)} | validacao {len(val_idx)}", flush=True)
    pred = pd.concat(preds, ignore_index=True)
    pred[["q_lo", "q_hi"]] = np.sort(pred[["q_lo", "q_hi"]].to_numpy(), axis=1)  # evita cruzamento de quantis
    cal, ev = pred[pred["origin"] != "origin_3"], pred[pred["origin"] == "origin_3"]

    methods = {
        "conformal_global": conformal(cal, ev, None),
        "conformal_por_tipo_de_canal (atual)": conformal(cal, ev, ["channel_type"]),
        "conformal_por_grupo": conformal(cal, ev, ["grupo"]),
        "conformal_por_grupo_x_setor": conformal(cal, ev, ["grupo", "sector"]),
        "quantis_gbm_crus": (ev["q_lo"].to_numpy(), ev["q_hi"].to_numpy()),
        "cqr_global": cqr(cal, ev, None),
        "cqr_por_grupo": cqr(cal, ev, ["grupo"]),
    }
    results = {name: summarize(ev, lo, hi) for name, (lo, hi) in methods.items()}
    report = {"pergunta": "qual calibracao do intervalo de 80% cobre perto de 80% em todos os grupos e setores?",
              "protocolo": "rolling-origin so em split=train do Gold v2.1; calibra nas origens 1-2, avalia na origem 3",
              "linhas_calibracao": int(len(cal)), "linhas_avaliacao": int(len(ev)), "metodos": results,
              "rodado_em": datetime.now(timezone.utc).isoformat()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pred.to_csv(OUT / "rolling_origin_predictions.csv", index=False)
    print(f"\n| Metodo | Cobertura | Largura mediana | Pessoa | Marca | Midia | Automotivo | Moda | Pior desvio |\n|---|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        g, s = r["por_grupo"], r["por_sector"]
        print(f"| {name} | {r['cobertura']:.1%} | {r['largura_mediana_fator']}x | {g.get('pessoa', 0):.1%} | {g.get('marca', 0):.1%} | "
              f"{g.get('midia', 0):.1%} | {s.get('automotivo', 0):.1%} | {s.get('moda', 0):.1%} | {r['pior_desvio_de_80pct_entre_grupos_e_setores']:.3f} |")


if __name__ == "__main__":
    main()
