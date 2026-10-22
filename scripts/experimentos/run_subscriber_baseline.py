"""Baseline declarado do TAP (secao 6.1): regressao linear sobre inscritos, no teste do E-15 (Gold v2.1).

So reporta; nao escolhe modelo. Bracos, todos avaliados nas mesmas 1.236 linhas de teste do E-15:
  B0           mediana historica das curtidas do canal (baseline minimo do projeto)
  B_INS        OLS  log(1 + curtidas) ~ log(1 + inscritos)                  (ajustado em split = train)
  B_INS_SETOR  OLS  log(1 + curtidas) ~ log(1 + inscritos) + setor + tipo   (ajustado em split = train)
  CB_FINAL_v2  modelo do E-15 (previsoes ja salvas em e14_e15_gold_v2/e15/predictions.csv)

Inscritos: Silver channel_stats (fotografia da data da coleta, 20-21/10/2026). Para Shorts antigos isso favorece o
baseline (o canal cresceu depois do post, em parte por causa dele); por isso o numero fica so no baseline.

Saida: artifacts/feature_experiments/baseline_inscritos/report.json

    python scripts/experimentos/run_subscriber_baseline.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e13_test_final as t13  # noqa: E402  (metricas e comparacao pareada por canal)

GOLD = ROOT / "data" / "gold" / "snapshots" / "gold_v2_1_20261021T113931Z_e1fd77c6" / "training_snapshot.csv"
E15 = ROOT / "artifacts" / "feature_experiments" / "e14_e15_gold_v2" / "e15" / "predictions.csv"
STATS_BASE = ROOT / "data" / "silver" / "channel_stats"
OUT = ROOT / "artifacts" / "feature_experiments" / "baseline_inscritos"
TARGET = "target_log_likes"


def latest_stats() -> tuple[pd.DataFrame, str]:
    run = sorted(p.name for p in STATS_BASE.iterdir() if (p / "channel_stats.csv").exists())[-1]
    stats = pd.read_csv(STATS_BASE / run / "channel_stats.csv")
    return stats[["channel_id", "subscriber_count_at_collection"]], run


def design(frame: pd.DataFrame, with_sector: bool) -> np.ndarray:
    cols = [np.ones(len(frame)), np.log1p(frame["subscriber_count_at_collection"].to_numpy(dtype=float))]
    if with_sector:
        cols += [frame["sector"].eq("moda").to_numpy(float), frame["channel_type"].eq("creator").to_numpy(float)]
    return np.column_stack(cols)


def main() -> None:
    stats, run = latest_stats()
    gold = pd.read_csv(GOLD, dtype={"video_id": str}).merge(stats, on="channel_id", how="left")
    missing = gold["subscriber_count_at_collection"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} linhas sem inscritos")
    train = gold[gold["split"].eq("train")]
    test = pd.read_csv(E15, dtype={"video_id": str}).merge(stats, on="channel_id", how="left")

    coefs = {}
    for arm, with_sector in (("B_INS", False), ("B_INS_SETOR", True)):
        beta, *_ = np.linalg.lstsq(design(train, with_sector), train[TARGET].to_numpy(), rcond=None)
        test[f"pred_{arm}"] = design(test, with_sector) @ beta
        names = ["intercepto", "log1p_inscritos"] + (["moda", "criador"] if with_sector else [])
        coefs[arm] = {n: round(float(b), 4) for n, b in zip(names, beta)}

    arms = ["B0", "B_INS", "B_INS_SETOR", "CB_FINAL_v2"]
    per, summary = {}, {}
    for arm in arms:
        per[arm], summary[arm] = t13.metrics(test, f"pred_{arm}")
    comparisons = {f"{a}_vs_{b}": {m: t13.paired(per[a], per[b], m) for m in ("mae", "pair_fair")}
                   for a, b in (("B_INS", "B0"), ("B_INS_SETOR", "B0"), ("CB_FINAL_v2", "B_INS"), ("CB_FINAL_v2", "B_INS_SETOR"))}
    corr = float(np.corrcoef(np.log1p(train["subscriber_count_at_collection"]), train[TARGET])[0, 1])
    report = {
        "pergunta": "a regressao linear sobre inscritos (baseline do TAP) bate a mediana do canal? e o modelo final bate esse baseline?",
        "inscritos": {"silver_channel_stats": run, "aviso": "fotografia da data da coleta, nao da data do post"},
        "treino": {"linhas": int(len(train)), "canais": int(train["channel_id"].nunique()), "coeficientes": coefs,
                   "correlacao_log_inscritos_log_curtidas": round(corr, 4)},
        "teste": {"linhas": int(len(test)), "canais": int(test["channel_id"].nunique()), "metricas": summary},
        "comparacoes": comparisons,
        "nota": "inscritos sao constantes dentro do canal: os bracos B_INS nao ordenam Shorts do mesmo canal (pares = 0,5 por construcao)",
        "rodado_em": datetime.now(timezone.utc).isoformat(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    test[["video_id", "channel_id", "channel_name", "sector", "channel_type", TARGET, "baseline_log_likes",
          "subscriber_count_at_collection"] + [f"pred_{a}" for a in arms]].to_csv(OUT / "predictions.csv", index=False)
    print(json.dumps({k: report[k] for k in ("treino", "teste")}, indent=1, ensure_ascii=False))
    for name, comp in comparisons.items():
        print(name, {m: (v["delta"], v["ic95"], v["p"], f"{v['canais_melhor']}/{v['n']}") for m, v in comp.items()})


if __name__ == "__main__":
    main()
