"""Servidor MCP (FastMCP, transporte stdio) que expoe o agente preditivo como tools.

    python -m galaxies_agent.mcp_server          # com PYTHONPATH=src, na raiz do repo

Tools:
- prever_criativo: previsao, intervalo, confianca, drivers SHAP, comparaveis e explicacao de um criativo.
- ranquear_criativos: mesma saida para 2 a 5 criativos, ordenados pelo desempenho relativo ao canal.
- historico_canal: resumo do historico de um canal da base (para o agente contextualizar a resposta).
- info_modelo: versao, dados de treino, alvo, metodo do intervalo (inspecao exigida pelo TAP).
"""

from __future__ import annotations

from functools import lru_cache

try:  # SDK mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # SDK mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .predictor import Predictor
from .schemas import CreativeInput

mcp = _Server("galaxies-agente-preditivo")


@lru_cache(maxsize=1)
def predictor() -> Predictor:
    return Predictor()


@mcp.tool()
def prever_criativo(criativo: CreativeInput) -> dict:
    """Preve curtidas de um YouTube Short ANTES da publicacao, com intervalo de 80%, confianca,
    desempenho relativo ao canal, fatores (SHAP) e Shorts comparaveis. O numero vem do modelo treinado;
    nao o altere ao explicar."""
    return predictor().predict(criativo).model_dump(mode="json")


@mcp.tool()
def ranquear_criativos(criativos: list[CreativeInput]) -> dict:
    """Compara de 2 a 5 criativos e devolve o ranking pelo desempenho previsto relativo ao canal."""
    if not 2 <= len(criativos) <= 5:
        raise ValueError("envie de 2 a 5 criativos")
    return predictor().rank(criativos).model_dump(mode="json")


@mcp.tool()
def historico_canal(channel_id: str, ultimos: int = 10) -> dict:
    """Resumo do historico de curtidas de um canal da base e seus ultimos Shorts."""
    return predictor().channel_summary(channel_id, ultimos)


@mcp.tool()
def info_modelo() -> dict:
    """Versao do modelo, dados de treino, alvo e metodo do intervalo."""
    return predictor().model_info()


if __name__ == "__main__":
    mcp.run()
