"""Analise de onde o modelo final erra (teste final do E-13, ja aberto; nao orienta escolha de modelo).

Le artifacts/feature_experiments/e13_final/test_final/test_predictions.csv e cruza com o Gold e a rubrica.
Saida: error_analysis.json na mesma pasta e um resumo em Markdown no terminal.

    python scripts/analyze_e13_errors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "artifacts" / "feature_experiments" / "e13_final" / "test_final"
TARGET, BASELINE, PRED = "target_log_likes", "baseline_log_likes", "pred_CB_FINAL"
BANDS = (-np.inf, -0.3, 0.3, 1.0, np.inf)
LABELS = ["abaixo", "na_media", "acima", "viral"]


def band(x):
    return pd.cut(x, BANDS, labels=LABELS).astype(str)


def summary(frame: pd.DataFrame) -> dict:
    err = frame[PRED] - frame[TARGET]
    b0 = frame[BASELINE] - frame[TARGET]
    return {"n": int(len(frame)), "mae": round(float(err.abs().mean()), 3), "mae_b0": round(float(b0.abs().mean()), 3),
            "vies": round(float(err.mean()), 3),
            "cobertura_80": round(float(((frame[TARGET] >= frame["lo"]) & (frame[TARGET] <= frame["hi"])).mean()), 3)}


def main() -> None:
    pred = pd.read_csv(TEST / "test_predictions.csv", parse_dates=["published_at"])
    latest = json.loads((ROOT / "data" / "gold" / "snapshots" / "LATEST.json").read_text(encoding="utf-8"))
    gold = pd.read_csv(ROOT / latest["path"], parse_dates=["published_at"])
    rubric = pd.read_csv(ROOT / "data" / "silver" / "gemini_video" / "gemini_video_features.csv")
    silver = pd.read_csv(ROOT / "data" / "silver" / "eligible_shorts" / "20261002T013000Z" / "eligible_shorts.csv")
    df = (pred.merge(gold[["video_id", "sector", "duration_seconds", "channel_prior_posts"]], on="video_id")
              .merge(rubric[["video_id", "tom_narracao", "tem_rosto", "potencial_viral_0a10"]], on="video_id", how="left")
              .merge(silver[["video_id", "title"]], on="video_id", how="left"))
    df = df.rename(columns={"interval_low": "lo", "interval_high": "hi"})
    df["err"] = df[PRED] - df[TARGET]
    df["faixa_real"] = band(df[TARGET] - df[BASELINE])
    df["faixa_prevista"] = band(df[PRED] - df[BASELINE])
    collected = pd.Timestamp("2026-10-02", tz="UTC")
    df["idade_dias"] = (collected - df["published_at"]).dt.days
    df["idade"] = pd.cut(df["idade_dias"], [0, 60, 120, 240, 10_000], labels=["30-60 d", "60-120 d", "120-240 d", "> 240 d"])

    out = {
        "geral": summary(df),
        "por_setor": {k: summary(g) for k, g in df.groupby("sector")},
        "por_tipo_de_canal": {k: summary(g) for k, g in df.groupby("channel_type")},
        "por_faixa_real": {k: summary(g) for k, g in df.groupby("faixa_real")},
        "por_idade_na_coleta": {str(k): summary(g) for k, g in df.groupby("idade", observed=True)},
        "por_tom_narracao": {k: summary(g) for k, g in df.groupby("tom_narracao")},
        "matriz_faixas": pd.crosstab(df["faixa_real"], df["faixa_prevista"]).reindex(index=LABELS, columns=LABELS, fill_value=0).to_dict(),
    }
    per_channel = (df.groupby("channel_name")
                     .apply(lambda g: pd.Series(summary(g)), include_groups=False)
                     .assign(ganho_vs_b0=lambda t: t["mae_b0"] - t["mae"]).sort_values("ganho_vs_b0"))
    out["por_canal"] = per_channel.reset_index().to_dict("records")
    worst = df.reindex(df["err"].abs().sort_values(ascending=False).index).head(10)
    out["maiores_erros"] = [{"canal": r.channel_name, "titulo": str(r.title)[:80], "curtidas_reais": int(round(np.expm1(r.target_log_likes))),
                             "previstas": int(round(np.expm1(r.pred_CB_FINAL))), "mediana_canal": int(round(np.expm1(r.baseline_log_likes))),
                             "erro_log": round(float(r.err), 2)} for r in worst.itertuples()]
    (TEST / "error_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def table(title, d):
        print(f"\n**{title}**\n\n| Grupo | n | MAE | MAE B0 | Vies | Cobertura 80% |\n| --- | ---: | ---: | ---: | ---: | ---: |")
        for k, v in d.items():
            print(f"| {k} | {v['n']} | {v['mae']} | {v['mae_b0']} | {v['vies']:+} | {v['cobertura_80']:.0%} |")

    table("Geral", {"todos": out["geral"]})
    table("Setor", out["por_setor"])
    table("Tipo de canal", out["por_tipo_de_canal"])
    table("Faixa real (desvio sobre a mediana do canal)", {k: out["por_faixa_real"][k] for k in LABELS if k in out["por_faixa_real"]})
    table("Idade do Short na coleta", out["por_idade_na_coleta"])
    table("Tom da narracao (Gemini)", out["por_tom_narracao"])
    print("\n**Canais (pior -> melhor ganho sobre o B0)**")
    for r in out["por_canal"]:
        print(f"- {r['channel_name']}: n={int(r['n'])} MAE {r['mae']} vs B0 {r['mae_b0']} (vies {r['vies']:+})")
    print("\n**Matriz de faixas (linha = real, coluna = prevista)**")
    print(pd.crosstab(df["faixa_real"], df["faixa_prevista"]).reindex(index=LABELS, columns=LABELS, fill_value=0))
    print("\n**Maiores erros**")
    for r in out["maiores_erros"]:
        print(r)


if __name__ == "__main__":
    main()
