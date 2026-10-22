"""Como deve ser um Short: o que pesa no modelo servido e como cada caracteristica se associa as curtidas.

Descritivo (nao causal, nao escolhe modelo). Sai em artifacts/feature_experiments/what_works/:
  importancia.json  SHAP medio absoluto do modelo final do pacote servido (LATEST.json) no Gold v2.1, por bloco de feature
  atributos.json    para cada caracteristica, curtidas dos Shorts em cada nivel em relacao a mediana do proprio canal
                    (media geometrica de exp(alvo - B0), IC 95% pela media do log), so niveis com >= 80 Shorts
Quadros do video (E-17): variacao entre quadros e semelhanca capa x video, em quintis.

    python scripts/experimentos/analyze_what_works.py --frames data/silver/features/video_frames_oar_v1/<run>/frames_oar.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "modelo"))
import run_e17_video_frames as e17  # noqa: E402
import train_tool_model as ttm  # noqa: E402
import train_tool_model_v2 as ttm2  # noqa: E402

ROOT = ttm.ROOT
OUT = ROOT / "artifacts" / "feature_experiments" / "what_works"
TARGET, BASELINE = ttm.TARGET, ttm.BASELINE
MIN_N = 80
BLOCKS = {  # bloco legivel: features do modelo final
    "Historico do canal": ["baseline_log_likes", "channel_prior_posts", "days_since_previous_post", "channel_id"],
    "Estilo visual da thumbnail": ttm.EMB_COLS,
    "Comparaveis do proprio canal": ["knn5_resid_mean", "knn5_sim_mean", "visual_novelty"],
    "Gancho e ritmo (Gemini)": ["forca_hook_0a10", "ritmo_cortes_0a10", "energia_musica_0a10"],
    "Pessoas e produto (Gemini)": ["tem_rosto", "pessoa_fala_para_camera", "close_up_produto", "before_after_ou_transformacao",
                                   "try_on_ou_demonstracao", "clareza_beneficio_produto_0a10"],
    "Texto, estetica e CTA (Gemini)": ["texto_na_tela", "densidade_texto_0a10", "brilho_saturacao_0a10", "estetica_geral_0a10",
                                       "chamada_para_acao", "tem_musica"],
    "Fala e narracao": ["tem_fala", "fala_palavras_log", "fala_palavras_por_segundo", "tom_narracao"],
    "Nota de potencial viral (Gemini)": ["potencial_viral_0a10"],
    "Duracao": ["duration_seconds"],
    "Titulo, descricao e tags": ["title_length", "description_length", "tag_count"],
    "Data e hora de publicacao": ["publish_hour_utc", "publish_weekday"],
    "Setor e tipo de canal": ["sector", "channel_type", "gemini_coverage"],
}


def ratio(resid: pd.Series) -> dict:
    m, s, n = float(resid.mean()), float(resid.std(ddof=1)), int(len(resid))
    half = 1.96 * s / np.sqrt(n)
    return {"shorts": n, "x_mediana": round(float(np.exp(m)), 3),
            "ic_baixo": round(float(np.exp(m - half)), 3), "ic_alto": round(float(np.exp(m + half)), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(ttm2.GOLD, parse_dates=["published_at"], dtype={"video_id": str})
    data = data.sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)
    emb = ttm2.merge_embeddings()
    matrix = np.vstack([emb[v] for v in data["video_id"]])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    ttm.add_knn(data, matrix)
    ttm.add_rubric(data)
    latest = json.loads((ttm.OUT_BASE / "LATEST.json").read_text(encoding="utf-8"))
    pkg = ROOT / latest["path"]
    meta = json.loads((pkg / "metadata.json").read_text(encoding="utf-8"))
    pca = np.load(pkg / "pca.npz")
    data[ttm.EMB_COLS] = (matrix - pca["mean"]) @ pca["components"].T

    spec = meta["models"]["final"]
    model = CatBoostRegressor().load_model(str(pkg / "model_final.cbm"))
    pool = ttm.to_pool(data, spec["numeric"], spec["categorical"], label=False)
    shap = np.abs(model.get_feature_importance(pool, type="ShapValues")[:, :-1]).mean(0)
    per_feature = dict(zip(spec["numeric"] + spec["categorical"], shap))
    total = sum(per_feature.values())
    used = set()
    importance = []
    for block, cols in BLOCKS.items():
        value = sum(per_feature.get(c, 0.0) for c in cols)
        used |= set(cols)
        importance.append({"bloco": block, "shap_medio": round(float(value), 4), "participacao": round(float(value / total), 4)})
    missing = set(per_feature) - used
    if missing:
        raise SystemExit(f"features sem bloco: {sorted(missing)}")
    importance.sort(key=lambda r: -r["shap_medio"])

    resid = data[TARGET] - data[BASELINE]
    _, scalars, _ = e17.frame_features(data["video_id"], matrix, args.frames)
    data[e17.FR_SCALARS] = scalars
    levels = []

    def add(attr, grupo, nivel, ordem, mask):
        if mask.sum() >= MIN_N:
            levels.append({"atributo": attr, "grupo": grupo, "nivel": nivel, "ordem": ordem, "chave": f"{attr}|{nivel}", **ratio(resid[mask])})

    for col, name in (("tem_rosto", "Rosto no video"), ("pessoa_fala_para_camera", "Pessoa fala para a camera"),
                      ("texto_na_tela", "Texto na tela"), ("tem_musica", "Musica"), ("chamada_para_acao", "Chamada para acao"),
                      ("close_up_produto", "Close no produto"), ("try_on_ou_demonstracao", "Try-on ou demonstracao")):
        add(name, "Conteudo (Gemini)", "Nao", 0, data[col].eq(0))
        add(name, "Conteudo (Gemini)", "Sim", 1, data[col].eq(1))
    for col, name in (("forca_hook_0a10", "Forca do gancho (0-10)"), ("ritmo_cortes_0a10", "Ritmo de cortes (0-10)"),
                      ("estetica_geral_0a10", "Estetica geral (0-10)")):
        for lo, hi, lab, o in ((0, 4, "0-4", 0), (5, 6, "5-6", 1), (7, 8, "7-8", 2), (9, 10, "9-10", 3)):
            add(name, "Notas do Gemini", lab, o, data[col].between(lo, hi))
    for i, tom in enumerate(["ausente", "calmo", "animado", "instrucional"]):
        add("Tom da narracao", "Conteudo (Gemini)", tom, i, data["tom_narracao"].eq(tom))
    for lo, hi, lab, o in ((0, 15, "ate 15 s", 0), (15, 30, "15-30 s", 1), (30, 60, "30-60 s", 2), (60, 181, "60-180 s", 3)):
        add("Duracao", "Formato", lab, o, (data["duration_seconds"] > lo) & (data["duration_seconds"] <= hi))
    for col, name in (("frames_variacao", "Troca de cena entre quadros"), ("thumb_vs_frames", "Capa parecida com o video")):
        q = pd.qcut(data[col], 5, labels=False)
        for k, lab in enumerate(["muito baixa", "baixa", "media", "alta", "muito alta"]):
            add(name, "Quadros do video (E-17)", lab, k, q.eq(k))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "importancia.json").write_text(json.dumps(importance, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "atributos.json").write_text(json.dumps(levels, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(importance, ensure_ascii=False))
    for r in levels:
        print(r["atributo"], r["nivel"], r["shorts"], r["x_mediana"], (r["ic_baixo"], r["ic_alto"]))


if __name__ == "__main__":
    main()
