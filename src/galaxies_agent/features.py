"""Features do modelo final (E-13) compartilhadas entre o treino (scripts/modelo/train_tool_model.py) e a tool.

Blocos acima do CB_EMB:
- comparaveis (KNN visual): desvio medio dos 5 Shorts ANTERIORES do mesmo canal com thumbnail mais
  parecida, a similaridade media desses 5 e a novidade visual (distancia ao centroide do canal);
- rubrica Gemini: notas 0-10, sinais sim/nao, tom de narracao e fala derivada da transcricao.

Mesmas definicoes de scripts/experimentos/run_e13_final.py (o experimento que aprovou o modelo).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

KNN_COLS = ["knn5_resid_mean", "knn5_sim_mean", "visual_novelty"]
SCORES = ["forca_hook_0a10", "densidade_texto_0a10", "ritmo_cortes_0a10", "energia_musica_0a10",
          "brilho_saturacao_0a10", "estetica_geral_0a10", "clareza_beneficio_produto_0a10", "potencial_viral_0a10"]
FLAGS = ["tem_rosto", "pessoa_fala_para_camera", "close_up_produto", "before_after_ou_transformacao",
         "try_on_ou_demonstracao", "texto_na_tela", "tem_musica", "chamada_para_acao"]
SPEECH = ["tem_fala", "fala_palavras_log", "fala_palavras_por_segundo"]
GEMINI_NUM = SCORES + FLAGS + SPEECH + ["gemini_coverage"]
GEMINI_CAT = ["tom_narracao"]
NO_RUBRIC = "<sem_rubrica>"
K = 5


def knn_features(prior_emb: np.ndarray, prior_resid: np.ndarray, vector: Optional[np.ndarray]) -> dict:
    """prior_emb: embeddings L2-normalizados dos Shorts anteriores do canal; prior_resid: desvio log de cada um."""
    out = {c: float("nan") for c in KNN_COLS}
    if vector is None or len(prior_emb) < K:
        return out
    sims = prior_emb @ vector
    top = np.argsort(-sims)[:K]
    centroid = prior_emb.mean(0)
    out["knn5_resid_mean"] = float(np.mean(prior_resid[top]))
    out["knn5_sim_mean"] = float(np.mean(sims[top]))
    out["visual_novelty"] = float(1 - float(centroid @ vector) / (np.linalg.norm(centroid) + 1e-9))
    return out


def _num(value) -> float:
    if value is None or value == "":
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return float(value.strip().lower() == "true")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rubric_features(rubric: Optional[dict], duration_seconds: float) -> dict:
    """Converte a rubrica (linha da Silver ou resposta ao vivo do Gemini) nas colunas do modelo."""
    if not rubric:
        out = {c: float("nan") for c in SCORES + FLAGS + SPEECH}
        out.update({"gemini_coverage": 0.0, "tom_narracao": NO_RUBRIC})
        return out
    out = {c: _num(rubric.get(c)) for c in SCORES + FLAGS}
    text = rubric.get("transcricao") or ""
    words = len(str(text).split()) if not (isinstance(text, float) and math.isnan(text)) else 0
    out["tem_fala"] = float(words > 0)
    out["fala_palavras_log"] = float(np.log1p(words))
    out["fala_palavras_por_segundo"] = float(words / max(float(duration_seconds or 1.0), 1.0))
    out["gemini_coverage"] = 1.0
    tom = rubric.get("tom_narracao")
    out["tom_narracao"] = str(tom) if tom not in (None, "") and not (isinstance(tom, float) and math.isnan(tom)) else NO_RUBRIC
    return out


LABELS = {
    "knn5_resid_mean": "desempenho dos Shorts anteriores do canal com visual parecido",
    "knn5_sim_mean": "quanto o visual se parece com Shorts anteriores do canal",
    "visual_novelty": "novidade visual em relacao ao canal",
    "forca_hook_0a10": "forca do gancho nos 3 primeiros segundos (Gemini)",
    "densidade_texto_0a10": "densidade de texto na tela (Gemini)",
    "ritmo_cortes_0a10": "ritmo de cortes (Gemini)",
    "energia_musica_0a10": "energia da musica (Gemini)",
    "brilho_saturacao_0a10": "brilho e saturacao (Gemini)",
    "estetica_geral_0a10": "estetica geral (Gemini)",
    "clareza_beneficio_produto_0a10": "clareza do beneficio do produto (Gemini)",
    "potencial_viral_0a10": "potencial viral avaliado pelo Gemini",
    "tem_rosto": "rosto no video (Gemini)",
    "pessoa_fala_para_camera": "pessoa falando para a camera (Gemini)",
    "close_up_produto": "close no produto (Gemini)",
    "before_after_ou_transformacao": "antes/depois ou transformacao (Gemini)",
    "try_on_ou_demonstracao": "try-on ou demonstracao (Gemini)",
    "texto_na_tela": "texto na tela (Gemini)",
    "tem_musica": "musica (Gemini)",
    "chamada_para_acao": "chamada para acao (Gemini)",
    "tem_fala": "fala no video",
    "fala_palavras_log": "quantidade de fala",
    "fala_palavras_por_segundo": "ritmo da fala (palavras por segundo)",
    "gemini_coverage": "video analisado pelo Gemini",
    "tom_narracao": "tom da narracao (Gemini)",
}


def display(key: str, value) -> Optional[str]:
    """Texto curto do valor de uma feature da rubrica/comparaveis; None se a chave nao for deste modulo."""
    if key not in LABELS:
        return None
    if isinstance(value, float) and math.isnan(value):
        return "sem dado"
    if key in SCORES:
        return f"{value:.0f}/10"
    if key in FLAGS or key in ("tem_fala", "gemini_coverage"):
        return "sim" if value >= 0.5 else "nao"
    if key == "fala_palavras_log":
        return f"{math.expm1(value):.0f} palavras"
    if key == "fala_palavras_por_segundo":
        return f"{value:.1f} palavras/s"
    if key == "knn5_resid_mean":
        return f"{math.exp(value):.2f}x a mediana do canal"
    if key in ("knn5_sim_mean", "visual_novelty"):
        return f"{value:.2f}"
    return str(value)
