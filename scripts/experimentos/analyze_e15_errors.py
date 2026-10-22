"""Analise de onde o modelo erra no teste do E-15 (Gold v2.1, ja aberto; nao orienta escolha de modelo).

Mesmo recorte do analyze_e13_errors.py, com os grupos do v2 (pessoa / marca / midia e origem do Short).
Intervalo de 80%: quantis por tipo de canal dos residuos fora da amostra das origens 1 e 2 do rolling-origin do
pacote tool_v3 (janelas 50-80% de cada canal, anteriores ao teste), aplicados a previsao do E-15.

Le  artifacts/feature_experiments/e14_e15_gold_v2/e15/predictions.csv
Sai artifacts/feature_experiments/e14_e15_gold_v2/e15/error_analysis.json e um resumo em Markdown no terminal.

    python scripts/experimentos/analyze_e15_errors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e13_errors import LABELS, band  # noqa: E402  (mesmas faixas do E-13)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "feature_experiments" / "e14_e15_gold_v2" / "e15"
TARGET, BASELINE, PRED = "target_log_likes", "baseline_log_likes", "pred_CB_FINAL_v2"
COLLECTED = pd.Timestamp("2026-10-21", tz="UTC")  # coleta do Gold v2 (20-21/10/2026)
INTERVAL = (0.10, 0.90)


def calibration() -> dict:
    latest = json.loads((ROOT / "artifacts" / "tool_model" / "LATEST.json").read_text(encoding="utf-8"))
    rolling = pd.read_csv(ROOT / latest["path"] / "calibration_rolling_origin.csv")
    cal = rolling[rolling["origin"].isin(["origin_1", "origin_2"])]
    error = cal[TARGET] - cal["pred_final"]
    out = {"global": np.quantile(error, INTERVAL)}
    for kind, idx in cal.groupby("channel_type").groups.items():
        out[kind] = np.quantile(error.loc[idx], INTERVAL)
    return out


def summary(frame: pd.DataFrame) -> dict:
    err = frame[PRED] - frame[TARGET]
    b0 = frame[BASELINE] - frame[TARGET]
    return {"n": int(len(frame)), "canais": int(frame["channel_id"].nunique()),
            "mae": round(float(err.abs().mean()), 3), "mae_b0": round(float(b0.abs().mean()), 3),
            "vies": round(float(err.mean()), 3),
            "cobertura_80": round(float(((frame[TARGET] >= frame["lo"]) & (frame[TARGET] <= frame["hi"])).mean()), 3)}


def main() -> None:
    df = pd.read_csv(OUT / "predictions.csv", parse_dates=["published_at"], dtype={"video_id": str})
    cal = calibration()
    q = df["channel_type"].map(lambda k: cal.get(k, cal["global"]))
    df["lo"] = df[PRED] + q.map(lambda v: v[0])
    df["hi"] = df[PRED] + q.map(lambda v: v[1])
    df["err"] = df[PRED] - df[TARGET]
    df["faixa_real"] = band(df[TARGET] - df[BASELINE])
    df["faixa_prevista"] = band(df[PRED] - df[BASELINE])
    df["idade_dias"] = (COLLECTED - df["published_at"]).dt.days
    df["idade"] = pd.cut(df["idade_dias"], [0, 60, 120, 240, 10_000], labels=["30-60 d", "60-120 d", "120-240 d", "> 240 d"])
    df["tom_narracao"] = df["tom_narracao"].fillna("<sem_rubrica>")

    groups = {"por_setor": "sector", "por_grupo": "grupo", "por_tipo_de_canal": "channel_type", "por_origem": "origem_v2",
              "por_faixa_real": "faixa_real", "por_tom_narracao": "tom_narracao"}
    out = {"geral": summary(df), "intervalo": {"metodo": "quantis 10%/90% dos residuos das origens 1-2 do rolling-origin do tool_v3, por tipo de canal",
                                               "quantis": {k: [round(float(x), 4) for x in v] for k, v in cal.items()}}}
    for key, col in groups.items():
        out[key] = {str(k): summary(g) for k, g in df.groupby(col)}
    out["por_idade_na_coleta"] = {str(k): summary(g) for k, g in df.groupby("idade", observed=True)}
    matrix = pd.crosstab(df["faixa_real"], df["faixa_prevista"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
    out["matriz_faixas"] = matrix.to_dict()
    per_channel = (df.groupby(["channel_name", "sector", "grupo"])
                     .apply(lambda g: pd.Series(summary(g)), include_groups=False)
                     .assign(ganho_vs_b0=lambda t: t["mae_b0"] - t["mae"]).sort_values("ganho_vs_b0"))
    out["por_canal"] = per_channel.reset_index().to_dict("records")
    out["canais_melhores_que_b0"] = int((per_channel["ganho_vs_b0"] > 0).sum())
    worst = df.reindex(df["err"].abs().sort_values(ascending=False).index).head(10)
    out["maiores_erros"] = [{"canal": r.channel_name, "grupo": r.grupo, "titulo": str(r.title)[:80],
                             "curtidas_reais": int(round(np.expm1(r.target_log_likes))),
                             "previstas": int(round(np.expm1(getattr(r, PRED)))),
                             "mediana_canal": int(round(np.expm1(r.baseline_log_likes))),
                             "erro_log": round(float(r.err), 2)} for r in worst.itertuples()]
    (OUT / "error_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def table(title, d):
        print(f"\n**{title}**\n\n| Grupo | n | MAE | MAE B0 | Vies | Cobertura 80% |\n| --- | ---: | ---: | ---: | ---: | ---: |")
        for k, v in d.items():
            print(f"| {k} | {v['n']} | {v['mae']} | {v['mae_b0']} | {v['vies']:+} | {v['cobertura_80']:.0%} |")

    table("Geral", {"todos": out["geral"]})
    for title, key in [("Setor", "por_setor"), ("Grupo", "por_grupo"), ("Origem do Short", "por_origem"),
                       ("Idade do Short na coleta", "por_idade_na_coleta"), ("Tom da narracao (Gemini)", "por_tom_narracao")]:
        table(title, out[key])
    table("Faixa real (desvio sobre a mediana do canal)", {k: out["por_faixa_real"][k] for k in LABELS if k in out["por_faixa_real"]})
    print(f"\n**Canais melhores que o B0: {out['canais_melhores_que_b0']} de {len(per_channel)}** (piores ganhos primeiro)")
    for r in out["por_canal"][:12]:
        print(f"- {r['channel_name']} ({r['sector']}/{r['grupo']}): n={int(r['n'])} MAE {r['mae']} vs B0 {r['mae_b0']} (vies {r['vies']:+})")
    print("\n**Matriz de faixas (linha = real, coluna = prevista)**")
    print(matrix)
    print("\n**Maiores erros**")
    for r in out["maiores_erros"]:
        print(r)


if __name__ == "__main__":
    main()
