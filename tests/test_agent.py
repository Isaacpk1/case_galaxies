"""Testes do agente LangGraph com LLM falso (offline).

    PYTHONPATH=src python -m pytest -q tests/test_agent.py
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from galaxies_agent import Predictor
from galaxies_agent.agent import build_agent, verify_numbers
from galaxies_agent.predictor import ROOT


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor()


@pytest.fixture(scope="module")
def creatives() -> list[dict]:
    latest = json.loads((ROOT / "data" / "gold" / "snapshots" / "LATEST.json").read_text(encoding="utf-8"))
    gold = pd.read_csv(ROOT / latest["path"])
    rows = gold[(gold["split"] == "train") & (gold["channel_name"] == "Carros com Camanzi")].tail(2)
    return [{"creative": f"https://www.youtube.com/shorts/{v}", "caption": "teste", "planned_datetime": "2026-07-01T18:00",
             "channel_id": c, "sector": "automotivo"} for v, c in zip(rows["video_id"], rows["channel_id"])]


def test_deterministic_single(predictor, creatives):
    out = build_agent(predictor, llm=None).invoke({"criativos": creatives[:1]})
    assert out["resultado"]["tipo"] == "previsao"
    assert out["explicacao_fonte"] == "deterministica"
    assert out["verificacao"]["ok"], out["verificacao"]  # o texto deterministico so usa numeros das tools


def test_deterministic_ranking(predictor, creatives):
    out = build_agent(predictor, llm=None).invoke({"criativos": creatives})
    assert out["resultado"]["tipo"] == "ranking"
    assert [r["rank"] for r in out["resultado"]["ranking"]] == [1, 2]
    assert out["verificacao"]["ok"], out["verificacao"]


def test_llm_parses_request_and_explains(predictor, creatives):
    base = creatives[0]
    parsed = json.dumps({"criativos": [{"creative": base["creative"], "caption": "Novo SUV", "description": "",
                                        "tags": [], "planned_datetime": "2026-07-01T18:00", "channel_id": base["channel_id"],
                                        "sector": "automotivo", "duration_seconds": None}]})
    pred = predictor.predict(__import__("galaxies_agent").CreativeInput(**dict(base, caption="Novo SUV")))
    honest = (f"Estimativa antes da publicacao: {pred.predicted_metric} curtidas, intervalo de 80% entre "
              f"{pred.interval_low} e {pred.interval_high}.")
    llm = FakeListChatModel(responses=[parsed, honest])
    out = build_agent(predictor, llm=llm).invoke({"pedido": "Quanto esse Short vai render no Camanzi?",
                                                  "criativos": [{"creative": base["creative"]}]})
    assert out["explicacao_fonte"] == "llm"
    assert out["resposta"] == honest


def test_hallucinated_number_is_rejected(predictor, creatives):
    llm = FakeListChatModel(responses=["Vai viralizar com 987654 curtidas, certeza."])
    out = build_agent(predictor, llm=llm).invoke({"criativos": creatives[:1]})
    assert not out["verificacao"]["ok"]
    assert "987654" in out["verificacao"]["numeros_nao_encontrados"]
    assert out["explicacao_fonte"].startswith("deterministica")
    assert "987654" not in out["resposta"]


def test_missing_context_without_llm_fails_cleanly(predictor):
    out = build_agent(predictor, llm=None).invoke({"pedido": "preve isso", "criativos": [{"creative": "abc"}]})
    assert out["resposta"].startswith("Nao foi possivel prever")


def test_verify_numbers_formats():
    src = {"predicted_metric": 1383, "relative_to_channel": 1.841, "interval_low": 550}
    assert verify_numbers("1.383 curtidas, 1,84x a mediana, minimo 550", src)["ok"]
    assert not verify_numbers("1.500 curtidas", src)["ok"]
