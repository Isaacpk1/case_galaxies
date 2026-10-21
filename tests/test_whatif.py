"""What-if: variacoes de um fator re-pontuadas pelo modelo.

    PYTHONPATH=src python -m pytest -q tests/test_whatif.py
"""

from __future__ import annotations

import pandas as pd
import pytest

from galaxies_agent import CreativeInput, Predictor
from galaxies_agent.predictor import ROOT


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(live_rubric=False)


@pytest.fixture(scope="module")
def item(predictor) -> CreativeInput:
    gold = pd.read_csv(ROOT / predictor.meta["inputs"]["gold"]["path"], parse_dates=["published_at"],
                       dtype={"video_id": str})  # Gold do pacote servido
    row = next(gold[gold["split"] == "train"].sample(1, random_state=11).itertuples())
    return CreativeInput(creative=row.video_id, caption="Titulo de teste", planned_datetime=row.published_at.to_pydatetime(),
                         channel_id=row.channel_id, sector=row.sector, duration_seconds=float(row.duration_seconds))


def test_default_grid(predictor, item):
    out = predictor.what_if(item)
    factors = [v.factor for v in out.variants]
    assert factors.count("horario") == 5 and factors.count("dia_da_semana") == 7
    assert set(out.best_by_factor) == {"horario", "dia_da_semana"}
    assert all(v.interval_low <= v.predicted_metric <= v.interval_high for v in out.variants)
    assert out.base.model_variant == "final" and all(v.model_variant == "final" for v in out.variants)
    assert "tamanho" in out.note


def test_same_value_is_zero_delta(predictor, item):
    local_hour = item.planned_datetime.astimezone(__import__("zoneinfo").ZoneInfo("America/Sao_Paulo"))
    same = predictor.what_if(item, hours=[local_hour.hour], durations=[item.duration_seconds])
    for v in same.variants:
        if v.factor == "duracao" or local_hour.minute == 0:
            assert v.delta_vs_base_pct == pytest.approx(0.0, abs=0.05)


def test_caption_only_changes_through_length(predictor, item):
    out = predictor.what_if(item, captions=["A" * len(item.caption), "Um titulo bem mais comprido para o mesmo video"])
    same_len, longer = out.variants
    assert same_len.delta_vs_base_pct == pytest.approx(0.0, abs=0.05)  # mesmo tamanho -> mesma previsao
    assert longer.factor == "titulo"
