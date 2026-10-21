"""LLM zero-shot vs modelo treinado (extensao opcional do TAP, secao 4.2).

O "palpite" do LLM e a nota `potencial_viral_0a10` que o Gemini da ao assistir o Short (rubrica_tap_v1),
sem ver curtidas nem historico. Como e uma nota 0-10, so faz sentido comparar ORDENACAO: dentro de cada
canal, quem o LLM poe na frente teve mais curtidas? Comparado com o CB_FINAL e com o B0 nas mesmas linhas.

Linhas: teste final do E-13 (612 Shorts, 26 canais), E-14 (modelo do v1 congelado em dados novos do Gold v2.1)
e teste do E-15 (modelo retreinado no Gold v2.1, 1.236 Shorts, 68 canais; o "CB_FINAL" desse bloco e o CB_FINAL_v2).
Nao escolhe modelo nem ajusta nada: so reporta. Observacao: a nota do LLM tambem e uma das features do
CB_FINAL; aqui ela e usada sozinha, como um LLM "chutando" sem o modelo.

    python scripts/experimentos/run_llm_zero_shot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e13_final as e13  # noqa: E402
import run_e13_test_final as t13  # noqa: E402

RUBRIC = ROOT / "data" / "silver" / "gemini_video" / "gemini_video_features.csv"
SOURCES = {  # nome: (previsoes, coluna do modelo treinado)
    "teste_final_E13": (ROOT / "artifacts" / "feature_experiments" / "e13_final" / "test_final" / "test_predictions.csv", "pred_CB_FINAL"),
    "dados_novos_E14": (ROOT / "artifacts" / "feature_experiments" / "e14_e15_gold_v2" / "e14" / "predictions.csv", "pred_CB_FINAL"),
    "teste_E15_gold_v2": (ROOT / "artifacts" / "feature_experiments" / "e14_e15_gold_v2" / "e15" / "predictions.csv", "pred_CB_FINAL_v2"),
}
OUT = ROOT / "artifacts" / "feature_experiments" / "llm_zero_shot"


def ranking(frame: pd.DataFrame, cols: dict[str, str]) -> tuple[dict, dict]:
    per = {name: [] for name in cols}
    for ch, g in frame.groupby("channel_name"):
        if len(g) < 3:
            continue
        a = g[e13.TARGET].to_numpy()
        for name, col in cols.items():
            p = g[col].to_numpy(dtype=float)
            rho = pd.Series(a).corr(pd.Series(p), method="spearman") if np.std(p) > 0 else np.nan
            per[name].append({"channel_name": ch, "pair_fair": e13.pairwise(a, p), "spearman": rho})
    per = {k: pd.DataFrame(v) for k, v in per.items()}
    summary = {k: {"pares_justos_macro_canal": round(float(v["pair_fair"].mean()), 4),
                   "spearman_intra_canal": round(float(v["spearman"].mean()), 4)} for k, v in per.items()}
    return summary, per


def vs_chance(values: pd.Series, seed: int = 42) -> dict:
    """Permutacao de sinais por canal de (pares - 0,5): o LLM ordena melhor que uma moeda?"""
    d = values.dropna().to_numpy() - 0.5
    s = np.random.default_rng(seed).choice((-1, 1), (10000, len(d)))
    return {"delta_vs_0.5": round(float(d.mean()), 4), "p": round(float((np.abs((s * d).mean(1)) >= abs(d.mean())).mean()), 4),
            "canais_acima_de_0.5": int((d > 0).sum()), "n": int(len(d))}


def main() -> None:
    rubric = pd.read_csv(RUBRIC, dtype={"video_id": str}).drop_duplicates("video_id", keep="last")
    rubric["llm_score"] = pd.to_numeric(rubric["potencial_viral_0a10"], errors="coerce")
    report = {"pergunta": "o LLM sozinho (nota potencial_viral_0a10) ordena Shorts do mesmo canal tao bem quanto o modelo treinado?"}
    for name, (path, model_col) in SOURCES.items():
        if not path.exists():
            report[name] = "ainda nao disponivel"
            continue
        frame = pd.read_csv(path, dtype={"video_id": str}).merge(rubric[["video_id", "llm_score"]], on="video_id")
        frame = frame.dropna(subset=["llm_score", model_col])
        cols = {"LLM_zero_shot": "llm_score", "CB_FINAL": model_col, "B0": "pred_B0"}
        summary, per = ranking(frame, cols)
        scores = frame["llm_score"]
        report[name] = {
            "linhas": int(len(frame)), "canais": int(frame["channel_name"].nunique()),
            "ordenacao": summary,
            "CB_FINAL_vs_LLM": t13.paired(per["CB_FINAL"], per["LLM_zero_shot"], "pair_fair"),
            "LLM_vs_acaso": vs_chance(per["LLM_zero_shot"]["pair_fair"]),
            "distribuicao_da_nota": {"media": round(float(scores.mean()), 2), "desvio": round(float(scores.std()), 2),
                                     "pct_entre_6_e_8": round(float(scores.between(6, 8).mean()), 4),
                                     "valores": {int(k): int(v) for k, v in scores.value_counts().sort_index().items()}},
            "spearman_global_nota_vs_desvio_real": round(float(scores.corr(frame[e13.TARGET] - frame[e13.BASELINE], method="spearman")), 4),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
