"""Agente orquestrador em LangGraph (secao 7 do TAP).

Fluxo: interpretar -> contextualizar canal -> atributos do video -> prever/ranquear -> explicar -> verificar.

- `atributos` processa o video: rubrica da secao 6.2 do TAP (gancho, rosto, produto, texto na tela, ritmo,
  audio, estetica) e transcricao da fala, pelo Gemini a partir da URL publica (ou do cache da base).
  A rubrica entra no modelo final (E-13); sem ela, a tool usa o modelo reserva e marca confianca baixa.

- O LLM (Gemini via langchain-google-genai, ou qualquer chat model LangChain) so faz duas coisas:
  transformar um pedido em linguagem natural em `CreativeInput` e escrever a explicacao para o publico.
- O numero vem SEMPRE do modelo treinado (tools da `Predictor`). O no `verificar` confere que todo numero
  citado na explicacao existe nas saidas das tools; se o LLM inventar um numero, a explicacao e
  descartada e a versao deterministica (montada do SHAP) e usada.
- Sem LLM (sem chave ou sem rede), o agente funciona em modo deterministico com entrada estruturada.

    from galaxies_agent.agent import build_agent
    agent = build_agent()                         # usa GEMINI_API_KEY se houver
    out = agent.invoke({"pedido": "Qual destes 2 Shorts vai melhor no canal UC... sexta 18h?",
                        "criativos": [{"creative": "https://youtube.com/shorts/A"}, {"creative": "..."}]})
    print(out["resposta"])
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from .predictor import ROOT, Predictor
from .schemas import CreativeInput

PARSE_PROMPT = """Converta o pedido em JSON para a tool de previsao de YouTube Shorts.
Responda SOMENTE um JSON: {{"criativos": [{{"creative": str|null, "creative_id": str|null, "caption": str,
"description": str, "tags": [str], "planned_datetime": "AAAA-MM-DDTHH:MM", "channel_id": str|null, "sector":
"moda"|"beleza"|"automotivo", "duration_seconds": number|null}}]}}
Use os valores padrao quando o pedido nao disser: {defaults}
Criativos ja informados (complete, nao invente URLs): {criativos}
Pedido: {pedido}"""

EXPLAIN_PROMPT = """Voce explica a previsao de um modelo de machine learning para um time de marketing.
Regras obrigatorias:
1. Use SOMENTE os numeros que aparecem nos DADOS abaixo; nao calcule, arredonde de outro jeito nem invente numeros.
2. Diga que e uma estimativa antes da publicacao, cite o intervalo de 80% e a confianca (e os motivos se for baixa).
3. Explique os principais fatores (top_drivers) em linguagem simples e use os Shorts comparaveis como evidencia.
4. Os atributos do video (rubrica do Gemini, em video_attributes) entram no modelo; so diga que um atributo pesou
   se ele estiver em top_drivers. Os demais podem ser citados como descricao do video.
5. Em ranking, diga qual criativo priorizar e se a diferenca e decisiva ou nao (veja a nota do ranking).
6. Portugues do Brasil, no maximo 180 palavras, sem titulos.
DADOS:
{dados}"""


class AgentState(TypedDict, total=False):
    pedido: str
    criativos: list[dict]
    defaults: dict
    usar_llm: bool
    entradas: list[CreativeInput]
    historico: dict
    atributos: dict
    atributos_lista: list
    resultado: dict
    explicacao: str
    explicacao_fonte: str
    verificacao: dict
    resposta: str
    erros: list[str]


# ------------------------------------------------------------------ utilidades

def default_llm():
    """Gemini via LangChain se houver chave; senao None (modo deterministico)."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    env = ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=os.environ.get("GALAXIES_AGENT_MODEL", "gemini-3.5-flash-lite"),
                                      google_api_key=key, temperature=0)
    except Exception:
        return None


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _json_block(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("resposta do LLM sem JSON")
    return json.loads(match.group(0))


NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d.,]*\d|\b\d\b")


def _candidates(token: str) -> set[float]:
    token = token.strip(".,")
    out = set()
    for variant in (token.replace(".", "").replace(",", "."),  # 1.383 -> 1383 ; 1,84 -> 1.84
                    token.replace(",", "")):                  # 1,383 -> 1383 ; 1.84 -> 1.84
        try:
            out.add(float(variant))
        except ValueError:
            pass
    return out


def _allowed_numbers(*objects: Any) -> list[float]:
    found: list[float] = []

    def walk(obj):
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            found.append(float(obj))
        elif isinstance(obj, dict):
            for value in obj.values():
                walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value)
        elif isinstance(obj, str):
            for token in NUMBER.findall(obj):
                found.extend(_candidates(token))
    for obj in objects:
        walk(obj)
    return found


def verify_numbers(text: str, *sources: Any) -> dict:
    """Todo numero do texto precisa existir nas fontes (tolerancia de arredondamento)."""
    allowed = _allowed_numbers(*sources)
    unknown = []
    for token in NUMBER.findall(text):
        values = _candidates(token)
        if any(0 <= v <= 10 and v == int(v) for v in values):  # contagens pequenas ("3 Shorts", "top 5")
            continue
        ok = any(abs(v - a) <= max(0.011, 0.006 * abs(a)) for v in values for a in allowed)
        if not ok:
            unknown.append(token)
    return {"ok": not unknown, "numeros_nao_encontrados": unknown}


# ------------------------------------------------------------------ grafo

def build_agent(predictor: Optional[Predictor] = None, llm: Any = "auto"):
    predictor = predictor or Predictor()
    llm = default_llm() if llm == "auto" else llm

    def interpretar(state: AgentState) -> AgentState:
        defaults = state.get("defaults", {})
        raw = [dict(defaults, **c) for c in state.get("criativos", [])]
        try:
            if raw and all("planned_datetime" in c and "sector" in c and ("channel_id" in c or "channel_stats" in c) for c in raw):
                return {"entradas": [CreativeInput(**c) for c in raw]}
            if llm is None or not state.get("usar_llm", True):
                return {"erros": ["entrada incompleta e sem LLM para interpretar o pedido: informe planned_datetime, "
                                  "sector e channel_id/channel_stats de cada criativo"]}
            prompt = PARSE_PROMPT.format(defaults=json.dumps(defaults, ensure_ascii=False),
                                         criativos=json.dumps(raw, ensure_ascii=False), pedido=state.get("pedido", ""))
            parsed = _json_block(_text(llm.invoke(prompt)))
            items = [dict(defaults, **c) for c in parsed.get("criativos", [])]
            return {"entradas": [CreativeInput(**c) for c in items]}
        except (ValidationError, ValueError, json.JSONDecodeError) as error:
            return {"erros": [f"nao consegui montar a entrada da tool: {error}"[:500]]}

    def contextualizar(state: AgentState) -> AgentState:
        historico = {}
        for item in state["entradas"]:
            if item.channel_id and item.channel_id not in historico:
                try:
                    historico[item.channel_id] = predictor.channel_summary(item.channel_id, 5)
                except ValueError:
                    pass
        return {"historico": historico}

    def atributos(state: AgentState) -> AgentState:
        """Analisa o video de cada criativo (Gemini ao vivo ou cache) antes da previsao."""
        lista, resumo = [], {}
        for pos, item in enumerate(state["entradas"]):
            rubric, fonte = predictor.video_attributes(item)
            lista.append((rubric, fonte))
            key = item.creative_id or item.creative or f"criativo_{pos + 1}"
            resumo[key] = {"fonte": fonte, "gancho": (rubric or {}).get("hook_primeiros_3s"),
                           "tom_narracao": (rubric or {}).get("tom_narracao")}
        return {"atributos_lista": lista, "atributos": resumo}

    def prever(state: AgentState) -> AgentState:
        entradas, attrs = state["entradas"], state.get("atributos_lista")
        if len(entradas) == 1:
            out = predictor.predict(entradas[0], attrs[0] if attrs else None)
            return {"resultado": {"tipo": "previsao", **out.model_dump(mode="json")}}
        return {"resultado": {"tipo": "ranking", **predictor.rank(entradas, attrs).model_dump(mode="json")}}

    def deterministic(resultado: dict) -> str:
        if resultado["tipo"] == "previsao":
            return resultado["explanation"]
        lines = [f"{r['rank']}. {r['creative_id']}: {r['explanation']}" for r in resultado["ranking"]]
        return "\n".join(lines + [resultado["note"]])

    def explicar(state: AgentState) -> AgentState:
        resultado = state["resultado"]
        if llm is None or not state.get("usar_llm", True):
            return {"explicacao": deterministic(resultado), "explicacao_fonte": "deterministica"}
        dados = {"resultado": resultado, "historico_canal": state.get("historico", {})}
        try:
            text = _text(llm.invoke(EXPLAIN_PROMPT.format(dados=json.dumps(dados, ensure_ascii=False))))
        except Exception as error:
            return {"explicacao": deterministic(resultado), "explicacao_fonte": f"deterministica (LLM falhou: {type(error).__name__})"}
        return {"explicacao": text.strip(), "explicacao_fonte": "llm"}

    def verificar(state: AgentState) -> AgentState:
        check = verify_numbers(state["explicacao"], state["resultado"], state.get("atributos", {}), state.get("historico", {}))
        explicacao, fonte = state["explicacao"], state["explicacao_fonte"]
        if not check["ok"]:
            explicacao, fonte = deterministic(state["resultado"]), "deterministica (LLM citou numero fora das tools)"
        return {"verificacao": check, "explicacao": explicacao, "explicacao_fonte": fonte, "resposta": explicacao}

    def falha(state: AgentState) -> AgentState:
        return {"resposta": "Nao foi possivel prever: " + "; ".join(state.get("erros", []))}

    graph = StateGraph(AgentState)
    for name, fn in (("interpretar", interpretar), ("contextualizar", contextualizar), ("atributos", atributos),
                     ("prever", prever), ("explicar", explicar), ("verificar", verificar), ("falha", falha)):
        graph.add_node(name, fn)
    graph.add_edge(START, "interpretar")
    graph.add_conditional_edges("interpretar", lambda s: "falha" if s.get("erros") else "contextualizar")
    graph.add_edge("contextualizar", "atributos")
    graph.add_edge("atributos", "prever")
    graph.add_edge("prever", "explicar")
    graph.add_edge("explicar", "verificar")
    graph.add_edge("verificar", END)
    graph.add_edge("falha", END)
    return graph.compile()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agente preditivo (LangGraph)")
    parser.add_argument("pedido")
    parser.add_argument("--criativos", help="JSON com lista de criativos", default="[]")
    parser.add_argument("--sem-llm", action="store_true")
    args = parser.parse_args()
    out = build_agent(llm=None if args.sem_llm else "auto").invoke(
        {"pedido": args.pedido, "criativos": json.loads(args.criativos), "usar_llm": not args.sem_llm})
    print(out["resposta"])
    print("\n[fonte da explicacao]", out.get("explicacao_fonte"), "| verificacao:", out.get("verificacao"))
