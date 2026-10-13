"""E-13, etapa final: abre o teste fechado UMA vez para o modelo escolhido pela regra pre-registrada.

Treina CB_FINAL (modelo aprovado) e CB_EMB (referencia) em todo o treino, preve as 612 linhas de teste
e compara com B0. O intervalo de 80% usa os quantis q10/q90 dos residuos do rolling-origin do CB_FINAL
(por channel_type), sem olhar o teste. Se o resultado ja existir, o script recusa rodar de novo.

    python scripts/run_e13_test_final.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e13_final as e13  # noqa: E402

OUT = e13.OUT / "test_final"
RESULT = OUT / "test_report.json"
BANDS = (-np.inf, -0.3, 0.3, 1.0, np.inf)
LABELS = ["abaixo", "na_media", "acima", "viral"]


def fit_predict(train, test, arm):
    extra_num, extra_cat = e13.ARMS[arm]
    num, cat = e13.BASE_NUM + e13.EMB_COLS + extra_num, e13.BASE_CAT + extra_cat

    def pool(frame, label):
        x = frame[num + cat].copy()
        for c in cat:
            x[c] = x[c].fillna("<missing>").astype(str)
        return Pool(x, frame["resid"].to_numpy() if label else None, cat_features=cat)

    model = CatBoostRegressor(loss_function="MAE", random_seed=e13.SEED, verbose=False, allow_writing_files=False,
                              thread_count=-1, **e13.params()).fit(pool(train, True))
    return test[e13.BASELINE].to_numpy() + model.predict(pool(test, False))


def band(x):
    return np.asarray(pd.cut(x, BANDS, labels=LABELS, right=True)).astype(str)


def f1_macro(true, pred):
    scores = []
    for lab in LABELS:
        tp = np.sum((true == lab) & (pred == lab))
        fp = np.sum((true != lab) & (pred == lab))
        fn = np.sum((true == lab) & (pred != lab))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def metrics(frame, col):
    per_channel = []
    for ch, g in frame.groupby("channel_name"):
        a, p = g[e13.TARGET].to_numpy(), g[col].to_numpy()
        per_channel.append({"channel_name": ch, "mae": float(np.mean(np.abs(a - p))), "pair_fair": e13.pairwise(a, p)})
    pc = pd.DataFrame(per_channel)
    a, p, b = frame[e13.TARGET].to_numpy(), frame[col].to_numpy(), frame[e13.BASELINE].to_numpy()
    pos = a > 0
    likes_a, likes_p = np.expm1(a), np.expm1(p)
    nz = likes_a > 0
    tb, pb = band(a - b), band(p - b)
    return pc, {
        "mae_macro_canal": round(float(pc["mae"].mean()), 4),
        "pair_fair_macro_canal": round(float(pc["pair_fair"].mean()), 4),
        "mae_global": round(float(np.mean(np.abs(a - p))), 4),
        "spearman_global": round(float(pd.Series(a).corr(pd.Series(p), method="spearman")), 4),
        "mape_log": round(float(np.mean(np.abs(a[pos] - p[pos]) / a[pos])), 4),
        "mdape_curtidas": round(float(np.median(np.abs(likes_a[nz] - likes_p[nz]) / likes_a[nz])), 4),
        "dentro_de_2x": round(float(np.mean(np.abs(a - p) <= np.log(2))), 4),
        "f1_macro_faixas": round(f1_macro(tb, pb), 4),
        "acuracia_faixa": round(float(np.mean(tb == pb)), 4),
    }


def paired(pc_a, pc_b, metric, seed=e13.SEED):
    v = pc_a.set_index("channel_name")[metric] - pc_b.set_index("channel_name")[metric]
    d = v.dropna().to_numpy()
    rng = np.random.default_rng(seed)
    s = rng.choice((-1, 1), (10000, len(d)))
    boot = rng.choice(d, (10000, len(d))).mean(1)
    better = (d < 0) if metric == "mae" else (d > 0)
    return {"metrica": metric, "delta": round(float(d.mean()), 4),
            "ic95": [round(float(np.quantile(boot, .025)), 4), round(float(np.quantile(boot, .975)), 4)],
            "p": round(float((np.abs((s * d).mean(1)) >= abs(d.mean())).mean()), 4),
            "canais_melhor": int(better.sum()), "n": int(len(d))}


def main() -> None:
    if RESULT.exists():
        sys.exit(f"teste final ja foi aberto ({RESULT}); nao rodar de novo")
    decision = json.loads((e13.OUT / "report.json").read_text(encoding="utf-8"))["decisao_pre_registrada"]
    final_arm = decision["modelo_final"]
    assert final_arm in e13.ARMS, f"decisao invalida: {final_arm}"

    data, E, coverage = e13.load(include_test=True)
    train_idx = np.flatnonzero(data["split"].to_numpy() == "train")
    test_idx = np.flatnonzero(data["split"].to_numpy() == "test")
    mean = E[train_idx].mean(0)
    _, _, vt = np.linalg.svd(E[train_idx] - mean, full_matrices=False)
    data[e13.EMB_COLS] = (E - mean) @ vt[:16].T
    train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
    print(f"treino {len(train)} | teste {len(test)} | cobertura Gemini (treino+teste) {coverage:.1%} | modelo {final_arm}")

    test["pred_B0"] = test[e13.BASELINE]
    test["pred_CB_EMB"] = fit_predict(train, test, "CB_EMB")
    test[f"pred_{final_arm}"] = fit_predict(train, test, final_arm)

    # intervalo de 80%: residuos do rolling-origin do modelo final, por channel_type (sem olhar o teste)
    oos = pd.concat(pd.read_csv(f) for f in sorted(e13.TASKS.glob(f"{final_arm}__*.csv")))
    oos["err"] = oos[e13.TARGET] - oos["pred"]
    q = oos.groupby("channel_type")["err"].quantile([0.1, 0.9]).unstack()
    lo = test["channel_type"].map(q[0.1]).fillna(oos["err"].quantile(0.1))
    hi = test["channel_type"].map(q[0.9]).fillna(oos["err"].quantile(0.9))
    p = test[f"pred_{final_arm}"]
    inside = (test[e13.TARGET] >= p + lo) & (test[e13.TARGET] <= p + hi)

    results, per_channel = {}, {}
    for name in ("B0", "CB_EMB", final_arm):
        per_channel[name], results[name] = metrics(test, f"pred_{name}")

    report = {
        "aberto_em": datetime.now(timezone.utc).isoformat(),
        "modelo_final": final_arm,
        "linhas_teste": int(len(test)), "canais": int(test["channel_id"].nunique()),
        "cobertura_gemini_teste": round(float(test["gemini_coverage"].mean()), 4),
        "metricas": results,
        "comparacoes": {
            f"{final_arm}_vs_B0": {m: paired(per_channel[final_arm], per_channel["B0"], m) for m in ("mae", "pair_fair")},
            f"{final_arm}_vs_CB_EMB": {m: paired(per_channel[final_arm], per_channel["CB_EMB"], m) for m in ("mae", "pair_fair")},
        },
        "intervalo_80": {"cobertura_teste": round(float(inside.mean()), 4), "alvo": 0.8,
                         "quantis_por_channel_type": {k: [round(float(v[0.1]), 4), round(float(v[0.9]), 4)] for k, v in q.iterrows()}},
        "por_canal": {k: v.round(4).to_dict("records") for k, v in per_channel.items()},
        "protocolo": "teste aberto uma unica vez apos a decisao pre-registrada do E-13 (docs/07)",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["video_id", "channel_id", "channel_name", "channel_type", "published_at", e13.TARGET, e13.BASELINE,
            "pred_B0", "pred_CB_EMB", f"pred_{final_arm}"]
    test[cols].assign(interval_low=p + lo, interval_high=p + hi).to_csv(OUT / "test_predictions.csv", index=False)
    RESULT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("modelo_final", "metricas", "comparacoes", "intervalo_80")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
